
# 🏃‍♂️ How to Run – Accident Detection System

## 0) Prereqs
- Python 3.9+ recommended
- (Optional) NVIDIA GPU + CUDA for faster inference
- PostgreSQL running if you’ll use the API persistence


## 1) Create & activate a virtualenv
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate

## 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Download YOLOv8 weights

**Windows (PowerShell)**

```powershell
New-Item -ItemType Directory -Force -Path weights | Out-Null
Invoke-WebRequest "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt" -OutFile "weights\yolov8n.pt"
```

## 4) Point YAML to your weights & video

## 5) Quick desktop run (CLI)

```bash
python -m scripts.infer_accidents --config configs/infer.yaml
```

* ESC to exit.
* Creates `outputs/accidents.mp4` and `outputs/events.csv` if enabled.

## 6) Run API server (with DB + JWT)

1. Create `.env` (already provided) and ensure Postgres is running.
2. Start FastAPI:

```bash
uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000
```

3. Endpoints:

* Health: [http://localhost:8000/health](http://localhost:8000/health)
* Latest frame: [http://localhost:8000/frame](http://localhost:8000/frame)
* Login (POST): [http://localhost:8000/login](http://localhost:8000/login)  → returns JWT
* List persisted events (GET, JWT): [http://localhost:8000/events](http://localhost:8000/events)
* Live events (WebSocket): `ws://localhost:8000/events/ws`

## 7) Run dashboard (Next.js)

```bash
cd dashboard
npm install
npm run dev
# open http://localhost:3001
```

Set `NEXT_PUBLIC_API` if your API isn’t on localhost:8000.

✅ That’s it!

