# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fitz, hashlib, os, shutil, re, time, glob, json, httpx, asyncio, logging, traceback, unicodedata
from fuzzywuzzy import fuzz
import re, unicodedata
# ---------------- Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log", encoding="utf-8"), logging.StreamHandler()]
)

# ---------------- Config -----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# default path (thư mục inputs_demo bên trong cv-monitoring-pipeline)
DEFAULT_BASE_INPUTS = os.path.join(BASE_DIR, "inputs_demo")

# cho phép override bằng biến môi trường nếu cần (ví dụ: trong systemd EnvironmentFile)
BASE_INPUTS_DIR = os.getenv("BASE_INPUTS_DIR", DEFAULT_BASE_INPUTS)
# BASE_INPUTS_DIR = "/home/root1/project_ai_cv/cv-monitoring-pipeline/inputs_demo"
HASH_STORE = os.path.join(BASE_INPUTS_DIR, "uploaded_hashes.txt")
ES_HOST = "http://localhost:9200"
VECTOR_DIMS = 384
EMBED_URL = "http://10.0.3.54:8004/embed"
LLM_API = "http://10.0.3.54:8001/v1/chat/completions"

# ---------------- FastAPI -----------------
app = FastAPI(title="Unified CV/Job API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ---------------- Elasticsearch -----------------
es = Elasticsearch(ES_HOST)

# ---------------- Models -----------------
class IndexRequest(BaseModel):
    index_name: str
    date_days: Optional[int] = None
    page: Optional[int] = 1
    page_size: Optional[int] = 50

class SectionInfo(BaseModel):
    key: str
    text: str
    fuzzy_score: int

class FileInfo(BaseModel):
    id: str
    filename: str
    timestamp: str
    sections: List[SectionInfo]
    details: Optional[Dict[str, Any]] = None

class IndexResult(BaseModel):
    index_name: str
    total: int
    files: List[FileInfo]

class IndexResponse(BaseModel):
    index_matches: List[IndexResult]

class JDField(BaseModel):
    key: str
    value: str
    weight: float = 1.0

class JDRequest(BaseModel):
    id: str
    job_name: str
    fields: List[JDField]
    top_k: int = 5
    value_cosine_threshold: float = 0.5
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    other_jobs: str = "Tất cả"
    Searchin_jobs: List[str] = []

# ---------------- Helpers -----------------
def extract_text_pymupdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc: text += page.get_text("text")
    return text

def normalize_text(text: str) -> str:
    text = text.lower(); text = re.sub(r'\s+', ' ', text); text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def text_hash(text: str) -> str: return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()

def is_duplicate_text(text: str) -> bool:
    h = text_hash(text)
    if not os.path.exists(HASH_STORE): return False
    with open(HASH_STORE, "r", encoding="utf-8") as f: existing = set(line.strip() for line in f)
    return h in existing

def save_text_hash(text: str):
    with open(HASH_STORE, "a", encoding="utf-8") as f: f.write(text_hash(text) + "\n")

def detect_doc_type(text: str) -> str:
    cv_keywords = ["curriculum vitae","kinh nghiệm","học vấn","kỹ năng","ứng viên"]
    jd_keywords = ["mô tả công việc","yêu cầu công việc","quyền lợi","phúc lợi","mức lương"]
    cv_score = sum(kw in text.lower() for kw in cv_keywords)
    jd_score = sum(kw in text.lower() for kw in jd_keywords)
    return "cv" if cv_score>jd_score else "jd" if jd_score>cv_score else "unknown"

async def embed_texts_async(texts: List[str]) -> np.ndarray:
    if not texts: return np.zeros((0,VECTOR_DIMS),dtype=np.float32)
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(EMBED_URL,json={"texts":texts,"type":"query","normalize":True}); res.raise_for_status()
            data=res.json()
            if "embeddings" in data: return np.array(data["embeddings"],dtype=np.float32)
            elif "data" in data: return np.array([d["embedding"] for d in data["data"]],dtype=np.float32)
            elif isinstance(data,list): return np.array(data,dtype=np.float32)
    except Exception as e: logging.error(f"[embed_texts_async] {e}")
    return np.zeros((len(texts),VECTOR_DIMS),dtype=np.float32)

EXPERIENCE_SYNONYMS = ["kinh nghiệm","Kinh nghiệm làm việc","kinh_nghiem","thâm niên","experience","work_experience","lịch sử công việc"]
FUZZY_THRESHOLD=60

def normalize_key(text: str) -> str:
    """Chuẩn hóa key: bỏ dấu tiếng Việt, chuyển về thường, thay khoảng trắng và dấu chấm thành '_' """
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")  # bỏ dấu
    text = text.lower().strip()
    text = re.sub(r"[\s\.]+", "_", text)
    return text

def extract_basic_info(sections: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    synonyms = {
        "name": ["ho_ten", "fullname", "name", "candidate_name", "applicant", "ten", "Tên","tên","họ_tên"],
        "position": ["chuc_danh", "vi_tri", "position", "job", "title", "role", "designation"],
        "phone": ["dien_thoai", "phone", "mobile", "tel", "so_dien_thoai", "Số điện thoại","sđt"],
        "email": ["email", "e_mail", "mail"]
    }

    name = position = phone = email = None

    for sec in sections:
        key = normalize_key(sec.get("key") or "")
        val = (sec.get("text") or sec.get("value") or "").strip()

        # --- name ---
        if not name and any(k in key for k in synonyms["name"]):
            name = val

        # --- position ---
        if not position and any(k in key for k in synonyms["position"]):
            position = val

        # --- phone ---
        if not phone:
            # Bắt các định dạng như: 0383228849 / 0383 22 88 49 / 0383.22.88.49 / +84 383 228 849
            m = re.search(r"(\+84|0)(?:[\s\.\-]?\d){8,10}", val)
            if m:
                phone = m.group(0).strip()

        # --- email ---
        if not email:
            m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", val)
            if m:
                email = m.group(0).strip()

    return {
        "candidate_name": name,
        "job_position": position,
        "phone": phone,
        "email": email
    }

def extract_all_experience_fields(sections: List[Dict[str, Any]]) -> List[Dict[str,str]]:
    results=[]
    for s in sections:
        key=(s.get("key")or "").lower(); value=s.get("text") or s.get("value","")
        if not value: continue
        if any(k in key for k in EXPERIENCE_SYNONYMS): results.append({"key":s.get("key",""),"value":value}); continue
        max_score=max(fuzz.partial_ratio(key,syn.lower()) for syn in EXPERIENCE_SYNONYMS)
        if max_score>=FUZZY_THRESHOLD: results.append({"key":s.get("key",""),"value":value})
    return results

ANCHOR_KEYWORDS=["yêu cầu năng lực","yeu_cau_nang_luc","yêu_cầu.năng_lực","y/c năng lực","năng lực cần có","Kỹ năng & Kinh nghiệm yêu cầu","Yêu cầu","Yêu_cầu"]
def fuzzy_match_key(key:str,threshold:int=71)->Optional[int]:
    scores=[fuzz.partial_ratio(key.lower(),a.lower()) for a in ANCHOR_KEYWORDS]; max_score=max(scores) if scores else 0
    return int(max_score) if max_score>=threshold else None

# ---------------- Endpoints -----------------
@app.get("/")
def root(): return {"status":"API is running"}
@app.get("/robots.txt")
def robots(): return PlainTextResponse("User-agent: *\nDisallow:")

# upload-pdf, getid, search_v1 giữ nguyên logic như bản bạn gửi


# ---------------- API Endpoints -----------------
# --- API Upload PDF ---
@app.post("/upload-pdf/")
async def upload_document(
    request: Request,
    files: List[UploadFile],
    metadata: List[str] = Form(...),
    doc_type: str = Form(...),
    position_name: str = Form(...),
    overwrite: str = Form("false")
):
    results = []
    duplicate_files = []
    client_host = request.client.host
    doc_type = doc_type.lower()
    overwrite_flag = overwrite.lower() == "true"

    # --- Giữ nguyên tiếng Việt cho position_name, chỉ thêm "_" thay khoảng trắng ---
    position_name_clean = position_name.strip()
    position_name_clean = re.sub(r'\s+', '_', position_name_clean)  # khoảng trắng → "_"
    position_name_clean = re.sub(r'[\/\\:*?"<>|]', "_", position_name_clean)  # ký tự bất hợp pháp

    logging.info(f"[UPLOAD-PDF] position_name_clean: '{position_name_clean}' from client {client_host}")

    try:
        for idx, file in enumerate(files):
            meta = {}
            try:
                if idx < len(metadata):
                    meta = json.loads(metadata[idx])
            except Exception as e:
                logging.warning(f"[WARN] Metadata parse failed for {file.filename}: {e}")

            source = (meta.get("source") or "").strip()

            # --- Folder lưu file ---
            save_dir = os.path.join(BASE_INPUTS_DIR, doc_type, "processed", position_name_clean)
            os.makedirs(save_dir, exist_ok=True)

            logging.info(f"[UPLOAD-PDF] Saving '{file.filename}' to folder: {save_dir}")

            base_name, ext = os.path.splitext(file.filename)
            ext = ext.lower()
            timestamp = int(time.time())
            temp_path = os.path.join(save_dir, f"temp_{timestamp}{ext}")

            # --- Lưu tạm file ---
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # --- Trích xuất nội dung ---
            if ext == ".pdf":
                text_content = extract_text_pymupdf(temp_path)
            elif ext == ".txt":
                text_content = open(temp_path, "r", encoding="utf-8").read()
            else:
                os.remove(temp_path)
                results.append({"file": file.filename, "status": "error", "message": "Unsupported file type"})
                continue

            # --- Kiểm tra duplicate ---
            if is_duplicate_text(text_content):
                if overwrite_flag:
                    for of in glob.glob(os.path.join(save_dir, f"{base_name}_*.*")):
                        os.remove(of)
                    logging.info(f"[OVERWRITE] Deleted old files for {base_name}")
                else:
                    duplicate_files.append(file.filename)
                    os.remove(temp_path)
                    logging.info(f"[DUPLICATE] Skipping duplicate file: {file.filename}")
                    continue

            final_filename = f"{base_name}_{timestamp}{ext}"
            final_path = os.path.join(save_dir, final_filename)
            shutil.move(temp_path, final_path)

            # --- Lưu hash text ---
            save_text_hash(text_content)

            # --- Lưu nội dung ra TXT ---
            txt_filename = f"{base_name}_{timestamp}.txt"
            txt_path = os.path.join(save_dir, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text_content)

            detected_type = detect_doc_type(text_content)
            type_warning = None
            if detected_type != "unknown" and detected_type != doc_type:
                type_warning = f"File có vẻ là {detected_type.upper()}, nhưng bạn chọn {doc_type.upper()}."

            results.append({
                "filename": file.filename,
                "source": source,
                "position_name": position_name_clean,
                "folder_name": position_name_clean,
                "saved_file": final_path,
                "text_saved": txt_path,
                "detected_type": detected_type,
                "type_warning": type_warning,
                "text_preview": text_content[:300],
                "upload_time": timestamp,
            })

        response = {
            "status": "success",
            "client": client_host,
            "doc_type": doc_type,
            "position_name": position_name_clean,
            "results": results
        }
        if duplicate_files:
            response["duplicates"] = duplicate_files
        return response

    except Exception as e:
        logging.error(f"[FATAL ERROR] {e}")
        return {"status": "error", "message": str(e)}

@app.post("/getid", response_model=IndexResponse)
def get_files_all(req: IndexRequest):
    try:
        index_name_param=req.index_name.strip().lower()
        date_days=req.date_days; page=max(req.page,1); page_size=max(req.page_size,1); from_=(page-1)*page_size
        date_from=date_to=None
        if date_days and date_days>0:
            today=datetime.utcnow()
            date_from=(today-timedelta(days=date_days-1)).strftime("%Y-%m-%d")
            date_to=today.strftime("%Y-%m-%d")
        all_indices=[idx["index"] for idx in es.cat.indices(format="json") if idx["index"].startswith("jd_")]
        matching_indices=[idx for idx in all_indices if index_name_param in idx.lower()]
        if not matching_indices: return IndexResponse(index_matches=[])
        index_results=[]
        for idx in matching_indices:
            query={"match_all": {}}
            if date_from and date_to:
                query={"bool":{"must":[{"match_all":{}}],"filter":[{"range":{"timestamp":{"gte":f"{date_from}T00:00:00","lte":f"{date_to}T23:59:59"}}}]}}
            res=es.search(index=idx,body={"query":query,"_source":{"excludes":["sections.embedding"]},"from":from_,"size":page_size})
            hits=res.get("hits",{}).get("hits",[]); total=res.get("hits",{}).get("total",{}).get("value",0)
            files=[]
            for h in hits:
                src=h["_source"].copy()
                sections_filtered=[]
                for s in src.get("sections",[]):
                    key=s.get("key",""); score=fuzzy_match_key(key)
                    if score: sections_filtered.append(SectionInfo(key=key,text=s.get("text",""),fuzzy_score=score))
                files.append(FileInfo(id=h["_id"],filename=src.get("filename",""),timestamp=src.get("timestamp",""),sections=sections_filtered,details=src))
            index_results.append(IndexResult(index_name=idx[3:] if idx.startswith("jd_") else idx,total=total,files=files))
        return IndexResponse(index_matches=index_results)
    except Exception: logging.error(traceback.format_exc()); raise HTTPException(status_code=500,detail="Internal server error")

@app.post("/search_v1")
async def search(request: JDRequest):
    try:
        jd_values = [f.value for f in request.fields]
        jd_vecs = await embed_texts_async(jd_values)
        if jd_vecs.shape[0] == 0:
            return {"top_results": []}

        # ---------------- Xác định index Elasticsearch ----------------
        indices = []
        if request.other_jobs.lower() == "tất cả":
            indices = list(es.indices.get(index="cv_*").keys())
        elif request.other_jobs.lower() == "1 vị trí":
            indices = [f"cv_{request.job_name}"]
        elif request.other_jobs.lower() == "vài vị trí":
            indices = [f"cv_{request.job_name}"] + [f"cv_{j.strip()}" for j in request.Searchin_jobs if j.strip()]
        if not indices:
            indices = [f"cv_{request.job_name}"]

        # ---------------- Lấy dữ liệu từ Elasticsearch ----------------
        hits = []
        es_query = {"match_all": {}}
        if request.start_timestamp and request.end_timestamp:
            es_query = {"range": {"timestamp": {"gte": request.start_timestamp, "lte": request.end_timestamp}}}

        for idx in indices:
            if not es.indices.exists(index=idx):
                continue
            res = es.search(index=idx, query=es_query, size=200)
            hits.extend(res["hits"]["hits"])

        if not hits:
            return {"top_results": []}

        # ---------------- Hàm gom nhóm experience_fields động ----------------
        def reformat_experience_fields_dynamic(experience_fields: list) -> list:
            """
            Gom nhóm tất cả các trường trong experience_fields (dạng key[index].subkey).
            Không loại bỏ bất kỳ key nào — giữ toàn bộ dữ liệu gốc.
            """
            import re
            grouped = {}

            for item in experience_fields:
                key = item.get("key")
                value = item.get("value")
                if not key:
                    continue

                # Tìm index trong key, ví dụ: kinh_nghiệm[0].công_ty
                match = re.match(r".*?\[(\d+)\]\.(.+)", key)
                if not match:
                    continue
                idx = int(match.group(1))
                subkey = match.group(2)

                if idx not in grouped:
                    grouped[idx] = {}

                # Nếu là mảng con: trách_nhiệm[0], nhiệm_vụ[1], ...
                arr_match = re.match(r"([^\[\]]+)\[(\d+)\]", subkey)
                if arr_match:
                    field = arr_match.group(1)
                    grouped[idx].setdefault(field, [])
                    grouped[idx][field].append(value)
                else:
                    # Nếu key đã tồn tại và là string khác → chuyển thành list để giữ tất cả
                    if subkey in grouped[idx]:
                        existing = grouped[idx][subkey]
                        if isinstance(existing, list):
                            existing.append(value)
                        else:
                            grouped[idx][subkey] = [existing, value]
                    else:
                        grouped[idx][subkey] = value

            # Giữ nguyên thứ tự index xuất hiện
            items = [grouped[i] for i in sorted(grouped.keys())]
            return items

        # ---------------- Xử lý từng CV ----------------
        async def process_hit(h):
            src = h["_source"]
            sections = src.get("sections", [])
            texts = [s.get("text") or s.get("value", "") for s in sections]
            vecs = await embed_texts_async(texts)
            return h, sections, vecs

        processed = await asyncio.gather(*(process_hit(h) for h in hits))
        results = []

        for h, sections, cv_vecs in processed:
            if cv_vecs.shape[0] == 0:
                continue

            cosine_matrix = cosine_similarity(jd_vecs, cv_vecs)
            jd_scores = {}
            field_details_list = []

            for jd_idx, f in enumerate(request.fields):
                jd_key, jd_value, jd_weight = f.key, f.value, f.weight
                jd_value_score = 0.0
                matched_count = 0

                for sec_idx, sec in enumerate(sections):
                    cos_val = float(cosine_matrix[jd_idx, sec_idx])
                    #if cos_val >= request.value_cosine_threshold:
                    if cos_val >= 0.78:
                        field_score = cos_val * jd_weight
                        jd_value_score += field_score
                        matched_count += 1
                        field_details_list.append({
                            "jd_key": jd_key,
                            "jd_value": jd_value,
                            "section_key": sec.get("key", ""),
                            "section_value": sec.get("text") or sec.get("value", ""),
                            "metrics": {
                                "cosine_value": cos_val,
                                "weight": jd_weight,
                                "field_score": field_score,
                                "explanation": "field_score = cosine_value × weight"
                            }
                        })
                jd_scores[jd_key] = {
                    "jd_value": jd_value,
                    "total_score": jd_value_score,
                    "explanation": f"total_score = tổng field_score cho JD field '{jd_key}' (số fields match: {matched_count})"
                }

            total_score = sum(s['total_score'] for s in jd_scores.values())
            basic_info = extract_basic_info(sections)
            experience_fields = extract_all_experience_fields(sections)
            experience_fields = reformat_experience_fields_dynamic(experience_fields)

            cv_result = {
                "id": h["_id"],
                "filename": h["_source"].get("filename"),
                "candidate_name": basic_info.get("candidate_name"),
                "job_position": h["_index"].replace("cv_", ""),
                "phone": basic_info.get("phone"),
                "email": basic_info.get("email"),
                "total_score": total_score,
                "jd_scores": jd_scores,
                "field_details": field_details_list,
                "pdf_link": h["_source"].get("pdf_link"),
                "source": h["_source"].get("source"),
                "experience_fields": experience_fields
            }
            results.append(cv_result)

        # ---------------- Sắp xếp theo tổng điểm ----------------
        results = sorted(results, key=lambda x: x["total_score"], reverse=True)[:request.top_k]
        return {"top_results": results}

    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

