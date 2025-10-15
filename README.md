# 🧠 CV Monitoring Pipeline

Hệ thống **tự động theo dõi thư mục chứa CV và JD**, trích xuất thông tin bằng mô hình ngôn ngữ (LLM), tạo embedding và index vào **Elasticsearch** để phục vụ tìm kiếm, so khớp hồ sơ và vị trí.

---

## 📁 Cấu trúc dự án

```
cv-monitoring/
├── main.py                 # Chương trình chính
├── api_main.py             # Các api chính
├── requirements.txt        # Danh sách thư viện Python
├── Dockerfile              # Build image cho Docker
├── docker-compose.yml      # (Tùy chọn) Chạy kèm Elasticsearch
├── .env.example            # Mẫu biến môi trường
├── README.md               # Tài liệu hướng dẫn này
```

---

## ⚙️ Yêu cầu hệ thống

| Thành phần | Phiên bản khuyến nghị |
|-------------|------------------------|
| Python      | ≥ 3.9 (3.10 khuyến nghị) |
| pip         | Mới nhất |
| Elasticsearch | 8.x |
| Docker (tuỳ chọn) | 24+ |
| Git         | Mới nhất |

---

## 🔧 Cấu hình chính (sửa trong `.env` hoặc `main.py`)

```bash
BASE_INPUT_DIR=/home/root1/project_ai_cv/inputs_demo
EXTRACT_URL=http://10.0.3.54:8001/v1/chat/completions
EMBED_URL=http://10.0.3.54:8004/embed
VECTOR_DIMS=384
ES_HOST=http://localhost:9200
PDF_STORAGE_DIR=/home/root1/project_ai_cv/pdf_storage
PUBLIC_IP=10.0.3.54
```

> Nếu bạn dùng `.env`, có thể load tự động bằng thư viện `python-dotenv`.

---

## 🚀 Cài đặt & chạy (Local)

### 1️⃣ Clone repository

```bash
git clone https://github.com/anhtuan11no-sketch/cv-monitoring.git
cd cv-monitoring
```

### 2️⃣ Tạo môi trường ảo

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Nếu chưa có file `requirements.txt`, tạo nhanh bằng:
```bash
pip install watchdog requests elasticsearch rich python-dotenv json5
pip freeze > requirements.txt
```

### 4️⃣ Chạy chương trình

```bash
python api_main.py
python main.py
```

Khi chương trình chạy:
- Theo dõi thư mục `/inputs_demo` (hoặc đường dẫn theo cấu hình)
- Phát hiện file mới (CV, JD)
- Gửi yêu cầu đến API extract → tạo JSON → gọi embed → lưu vào Elasticsearch
- Copy PDF sang thư mục lưu trữ (`pdf_storage`)

---

## ⚙️ Chạy nền (Background Mode)
Tạo file `/etc/systemd/system/cv-monitor.service`:

```ini
[Unit]
Description=CV Monitoring Pipeline
After=network.target

[Service]
User=root1
WorkingDirectory=/home/root1/project_ai_cv/cv-monitoring
Environment="PATH=/home/root1/project_ai_cv/cv-monitoring/venv/bin"
ExecStart=/home/root1/project_ai_cv/cv-monitoring/venv/bin/python /home/root1/project_ai_cv/cv-monitoring/api_main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Kích hoạt dịch vụ:

```bash
sudo systemctl daemon-reload
sudo systemctl enable cv-monitor
sudo systemctl start cv-monitor
sudo journalctl -u cv-monitor -f
```

---

## 🐳 Chạy bằng Docker

### Dockerfile mẫu

```dockerfile
FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### Build & Run

```bash
docker build -t cv-monitor:latest .
docker run -d --name cv-monitor   -v /home/root1/project_ai_cv/inputs_demo:/app/inputs_demo   -v /home/root1/project_ai_cv/pdf_storage:/app/pdf_storage   --network host   cv-monitor:latest
```

> ⚠️ Dùng `--network host` nếu Elasticsearch chạy trên cùng máy (`localhost:9200`).

