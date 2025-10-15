# CV Monitoring & Ingestion Pipeline

## 🧠 Overview
This project watches a folder for new or modified `.txt` or `.pdf` files, extracts structured information using an LLM, embeds the text into vectors, and indexes the data into Elasticsearch.

## 🚀 Run
```bash
pip install -r requirements.txt
python src/watcher.py

