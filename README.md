# Crowd Detector (YOLOv8 + OpenCV)

Детекция людей на видео `crowd.mp4` с отрисовкой боксов, класса и уверенности.
Кросс‑платформенно (Linux / macOS / Windows), без скрытой магии: чтение видео,
загрузка весов, инференс, отрисовка и сохранение — прописаны явно.

## Быстрый старт

1) **Python 3.9–3.12** (рекомендуется `venv`):

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
