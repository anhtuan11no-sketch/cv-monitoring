# -*- coding: utf-8 -*-
import os
import re
import json
import json5
import time
import shutil
import hashlib
import requests
from datetime import datetime
from rich.console import Console
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from elasticsearch import Elasticsearch
from queue import Queue
from threading import Thread
import unicodedata

# ---------------- Config ----------------
BASE_INPUT_DIR = "/home/root1/project_ai_cv/cv-monitoring-pipeline/inputs_demo"
EXTRACT_URL = "http://10.0.3.54:8001/v1/chat/completions"
EMBED_URL = "http://10.0.3.54:8004/embed"
VECTOR_DIMS = 384
ES_HOST = "http://localhost:9200"
PDF_STORAGE_DIR = "/home/root1/project_ai_cv/pdf_storage"
PUBLIC_IP = "10.0.3.54"

console = Console()
es = Elasticsearch([ES_HOST], request_timeout=60)
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# ---------------- Elasticsearch ----------------
def create_index_if_not_exists(es_index):
    mapping = {
        "mappings": {
            "properties": {
                "filename": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "pdf_link": {"type": "keyword"},
                "source": {"type": "keyword"},
                "position": {"type": "keyword"},
                "sections": {
                    "type": "nested",
                    "properties": {
                        "key": {"type": "keyword"},
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": VECTOR_DIMS,
                            "index": True,
                            "similarity": "cosine",
                        },
                    },
                },
            }
        }
    }
    try:
        if not es.indices.exists(index=es_index):
            es.indices.create(index=es_index, body=mapping, ignore=400)
            console.print(f"[OK] Created index: {es_index}", style="bold green")
        else:
            console.print(f"[INFO] Index '{es_index}' already exists", style="yellow")
    except Exception as e:
        console.print(f"[ERROR] Index create failed: {e}", style="red")


# ---------------- Elasticsearch ----------------
def sanitize_index_name(name: str) -> str:
    """
    Chuyển tiếng Việt sang không dấu, thay ký tự không hợp lệ thành _,
    dùng được làm index Elasticsearch.
    """
    # 1. Chuyển sang không dấu
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))

    # 2. Chuyển thành chữ thường
    name = name.lower()

    # 3. Thay tất cả ký tự không hợp lệ (khoảng trắng, /, \, *, ?, ", <, >, |, ,) bằng _
    # name = re.sub(r'[^a-z0-9_\-+]', '_', name)

    # 4. Loại bỏ _ trùng lặp
    name = re.sub(r'_+', '_', name)

    # 5. Bỏ _ đầu và cuối
    name = name.strip('_')

    # Nếu trống, trả về default_index
    return name or "default_index"

def get_index_from_folder(folder_path):
    parts = folder_path.split(os.sep)
    if "cv" in parts:
        raw_index = f"cv_{parts[-1]}"
    elif "jd" in parts:
        raw_index = f"jd_{parts[-1]}"
    else:
        raw_index = "default_index"
    return sanitize_index_name(raw_index)

def get_doc_type_from_path(path):
    parts = path.split(os.sep)
    if "cv" in parts:
        return "CV"
    elif "jd" in parts:
        return "JD"
    return "UNKNOWN"

def get_source_and_position(path):
    """Tách nguồn và vị trí từ tên thư mục có dạng: nguon__vitri"""
    parts = path.split(os.sep)
    for p in parts:
        if "__" in p:
            tokens = p.split("__", 1)
            return tokens[0], tokens[1]
    return None, None

# ---------------- Extract ----------------
def clean_json_response(raw_text: str) -> str:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned)
    cleaned = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", cleaned)
    cleaned = cleaned.replace("'", '"')
    cleaned = re.sub(r'[\u2018\u2019\u201C\u201D]', '"', cleaned)
    return cleaned.strip()

def extract_content_to_json(text: str, doc_type: str) -> dict:
    if doc_type == "CV":
        system_prompt = """
        Hãy phân tích và bóc tách toàn bộ thông tin trong CV thành JSON có cấu trúc chuẩn, đầy đủ và chi tiết.
        Không được bỏ sót bất kỳ thông tin nào.
        Nếu thông tin không có, hãy gán giá trị null.
        Các trường cơ bản cần có:
            Thông tin cá nhân
            Học vấn
            Kinh nghiệm làm việc
            Kỹ năng
            Mục tiêu nghề nghiệp / Tóm tắt
        Nếu CV có các thông tin khác như Chứng chỉ, Dự án, Hoạt động ngoại khóa, Ngôn ngữ, Sở thích, v.v., cũng phải được bóc tách và đưa vào JSON.
        Lưu ý: Trả về DUY NHẤT một JSON hợp lệ, bắt đầu bằng { và kết thúc bằng }, không thêm bất kỳ chữ nào khác.
        """
    else:
        system_prompt = """
        Hãy phân tích và bóc tách toàn bộ thông tin trong JD thành JSON có cấu trúc chuẩn, đầy đủ và chi tiết.
        Không được bỏ sót bất kỳ thông tin nào.
        Nếu thông tin không có, hãy gán giá trị null.
        Các trường cơ bản cần có:
            Công ty
            Vị trí tuyển dụng
            Yêu cầu công việc
            Kỹ năng & Kinh nghiệm yêu cầu
            Phúc lợi
            Địa điểm
            Mức lương
        Nếu JD có các thông tin khác như Thời gian làm việc, Cách thức ứng tuyển, Liên hệ, Chỉ tiêu tuyển dụng, v.v., cũng phải được bóc tách và đưa vào JSON.
        Lưu ý: Trả về DUY NHẤT một JSON hợp lệ, bắt đầu bằng { và kết thúc bằng }, không thêm bất kỳ chữ nào khác.
        """

    payload = {
        "model": "aya-expanse-8b-Q4_K_M.gguf",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.15,
        "top_p": 0.9,
        "presence_penalty": 0.3,
        "frequency_penalty": 0.5,
    }

    try:
        response = requests.post(EXTRACT_URL, headers={"Content-Type": "application/json"}, json=payload, timeout=600)
        response.raise_for_status()
        raw_json = response.json()
        raw_content = raw_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        console.print(f"[RESULT] Raw model output ({doc_type}):\n{raw_content}\n", style="magenta")

        cleaned = clean_json_response(raw_content)

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                return json5.loads(cleaned)
            except Exception as e:
                debug_folder = os.path.join("/tmp/json_debug")
                os.makedirs(debug_folder, exist_ok=True)
                debug_file = os.path.join(debug_folder, f"raw_{int(time.time())}.json")
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                console.print(f"[ERROR] JSON parse failed. Raw saved: {debug_file}", style="red")
                return {"error": "invalid_json", "raw_content_file": debug_file, "fallback_text": cleaned}

    except Exception as e:
        console.print(f"[ERROR] Extract API failed: {e}", style="red")
        return {"error": "api_failed", "message": str(e)}

# ---------------- Embed ----------------
def embed_text_batch(texts):
    if not texts:
        return []
    payload = {"texts": texts, "type": "query", "normalize": True}
    try:
        response = requests.post(EMBED_URL, headers={"Content-Type": "application/json"}, json=payload, timeout=300)
        response.raise_for_status()
        embedding_result = response.json()
        vectors = []
        if isinstance(embedding_result, list):
            vectors = embedding_result
        elif "embeddings" in embedding_result:
            vectors = embedding_result["embeddings"]
        elif "data" in embedding_result:
            vectors = [item["embedding"] for item in embedding_result["data"]]
        return [[float(x) for x in vec] for vec in vectors]
    except Exception as e:
        console.print(f"[ERROR] Embed API failed: {e}")
        return [[0.0] * VECTOR_DIMS for _ in texts]

def flatten_json_to_sections(json_obj, parent_key=""):
    sections = []
    texts_to_embed = []
    keys = []

    def collect_texts(obj, key_prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                full_key = f"{key_prefix}.{k}" if key_prefix else k
                collect_texts(v, full_key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                full_key = f"{key_prefix}[{i}]"
                collect_texts(item, full_key)
        else:
            text = str(obj).strip()
            if text:
                texts_to_embed.append(text)
                keys.append(key_prefix)

    collect_texts(json_obj)
    vectors = embed_text_batch(texts_to_embed)
    for k, t, v in zip(keys, texts_to_embed, vectors):
        sections.append({"key": k, "text": t, "embedding": v})
    return sections

# ---------------- Elasticsearch Insert ----------------
def insert_to_elasticsearch(sections, filename, timestamp, es_index, 
                            pdf_link=None, filepath=None, source=None, position=None):
    doc_id = hashlib.md5(filepath.encode()).hexdigest() if filepath else f"{es_index}_{filename}"
    doc = {
        "filename": filename,
        "timestamp": timestamp,
        "pdf_link": pdf_link,
        "source": source,
        "position": position,
        "sections": sections
    }
    try:
        es.index(index=es_index, document=doc, id=doc_id)
        console.print(f"[ES] Inserted: {doc_id} → {es_index}", style="green")
    except Exception as e:
        console.print(f"[ERROR] ES insert failed for {filename}: {e}", style="red")

# ---------------- Processing ----------------
def process_file(filepath):
    if not filepath.endswith(".txt"):
        return

    filename = os.path.basename(filepath)
    folder = os.path.dirname(filepath)
    output_dir = os.path.join(folder, "processed_results")
    os.makedirs(output_dir, exist_ok=True)

    es_index = get_index_from_folder(folder)
    create_index_if_not_exists(es_index)

    json_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
    if os.path.exists(json_file):
        console.print(f"[SKIP] Already processed: {filename}", style="yellow")
        return

    console.print(f"\n[START] Processing: {filename}", style="bold cyan")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        doc_type = get_doc_type_from_path(folder)
        source, position = get_source_and_position(folder)
        extracted_json = extract_content_to_json(content, doc_type)

        sections = flatten_json_to_sections(extracted_json)
        if "fallback_text" in extracted_json:
            sections.extend(flatten_json_to_sections(extracted_json["fallback_text"], parent_key="fallback_text"))

        # ---------------- PDF ----------------
        pdf_filename = os.path.splitext(filename)[0] + ".pdf"
        pdf_src_path = os.path.join(folder, pdf_filename)
        pdf_dst_path = os.path.join(PDF_STORAGE_DIR, pdf_filename)

        if os.path.exists(pdf_src_path):
            shutil.copy(pdf_src_path, pdf_dst_path)
            pdf_link = f"http://{PUBLIC_IP}/pdf_storage/{pdf_filename}"
            console.print(f"[PDF] Saved original PDF → {pdf_link}", style="bold green")
        else:
            pdf_link = None
            console.print(f"[WARN] No PDF found for {filename}", style="yellow")

        result = {
            "filename": filename,
            "timestamp": datetime.utcnow().isoformat(),
            "doc_type": doc_type,
            "source": source,
            "position": position,
            "original_json": extracted_json,
            "sections": sections,
            "pdf_link": pdf_link
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        console.print(f"[SAVED] {json_file}", style="bold blue")

        insert_to_elasticsearch(sections, filename, result["timestamp"], es_index,
                                pdf_link=pdf_link, filepath=filepath,
                                source=source, position=position)

    except Exception as e:
        console.print(f"[ERROR] Failed processing {filename}: {e}", style="red")

# ---------------- Watchdog ----------------
def is_in_cv_or_jd(path):
    parts = path.split(os.sep)
    return "cv" in parts or "jd" in parts

class CVJDWatcher(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            if is_in_cv_or_jd(event.src_path):
                console.print(f"[INFO] New folder detected: {event.src_path}", style="bold cyan")
                for root, _, files in os.walk(event.src_path):
                    for fname in files:
                        if fname.endswith(".txt"):
                            file_queue.put(os.path.join(root, fname))
        else:
            if event.src_path.endswith(".txt") and is_in_cv_or_jd(event.src_path):
                console.print(f"[INFO] New file detected: {event.src_path}", style="bold cyan")
                file_queue.put(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            if is_in_cv_or_jd(event.dest_path):
                console.print(f"[INFO] Folder moved in: {event.dest_path}", style="bold cyan")
                for root, _, files in os.walk(event.dest_path):
                    for fname in files:
                        if fname.endswith(".txt"):
                            file_queue.put(os.path.join(root, fname))
        else:
            if event.dest_path.endswith(".txt") and is_in_cv_or_jd(event.dest_path):
                console.print(f"[INFO] File moved in: {event.dest_path}", style="bold cyan")
                file_queue.put(event.dest_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".txt") and is_in_cv_or_jd(event.src_path):
            console.print(f"[INFO] File modified: {event.src_path}", style="bold cyan")
            file_queue.put(event.src_path)

# ---------------- Main ----------------
if __name__ == "__main__":
    file_queue = Queue()

    def worker():
        while True:
            filepath = file_queue.get()
            if filepath is None:
                break
            try:
                process_file(filepath)
            except Exception as e:
                console.print(f"[ERROR] Worker failed for {filepath}: {e}", style="red")
            file_queue.task_done()

    NUM_WORKERS = 4
    workers = []
    for _ in range(NUM_WORKERS):
        t = Thread(target=worker, daemon=True)
        t.start()
        workers.append(t)

    # Initial scan
    for doc_type in ["cv", "jd"]:
        type_folder = os.path.join(BASE_INPUT_DIR, doc_type)
        if os.path.exists(type_folder):
            for root, _, files in os.walk(type_folder):
                for fname in files:
                    if fname.endswith(".txt"):
                        file_queue.put(os.path.join(root, fname))

    observer = Observer()
    for doc_type in ["cv", "jd"]:
        type_folder = os.path.join(BASE_INPUT_DIR, doc_type)
        if os.path.exists(type_folder):
            observer.schedule(CVJDWatcher(), path=type_folder, recursive=True)

    observer.start()
    console.print(f"[SYSTEM] Watching {BASE_INPUT_DIR}/cv and {BASE_INPUT_DIR}/jd ...", style="bold magenta")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("[SYSTEM] Stopping...", style="red")
        observer.stop()

    observer.join()

    for _ in workers:
        file_queue.put(None)
    for t in workers:
        t.join()

    console.print("[SYSTEM] Stopped.", style="red")
