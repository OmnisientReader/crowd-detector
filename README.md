## Установка и запуск

Требования: Python 3.10–3.13, 64‑битная ОС (Linux/macOS/Windows). Рекомендуется изолированное окружение (`venv`).

### 1) Создать окружение и обновить pip
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
```

### 2) Запуск программы
```bash
python main.py --input crowd.mp4 --output outputs/crowd_annotated.mp4 \
  --model yolov8n.pt --conf 0.30 --imgsz 960
