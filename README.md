# CV Monitoring Pipeline

This project automatically watches for new CV and JD text files, extracts structured information using a local LLM API, embeds data, and indexes it into Elasticsearch.

## 🚀 Features
- Auto-detect new CV/JD files (via Watchdog)
- Extract structured JSON using LLM (local API)
- Generate embeddings and push to Elasticsearch
- Supports PDF linking and JSON output
- Multi-threaded file processing

## 🛠️ Run locally
```bash
python api_main.py


