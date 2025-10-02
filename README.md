# 🚦 Accident Detection System (YOLOv8 + FastAPI + Next.js)

An end-to-end **AI-powered traffic accident detection** platform.

The system uses **YOLOv8** for vehicle detection, **DeepSORT** for multi-object tracking, and a custom **Accident Engine** to flag collisions, sudden deceleration, and near-miss events in live traffic feeds or videos.  

A **FastAPI** backend streams detection frames and accident events over WebSocket to a **Next.js dashboard**, which shows live video, highlights crashes, and triggers an **audio alarm + popup alert**.

---

## ✨ Features
- 🟢 **Real-time detection & tracking** of cars, buses, trucks, bikes.
- 🚨 **Accident alerts** – collision, sudden-stop, or near-miss events detected automatically.
- 🔊 **Siren alarm & popup banner** on the dashboard when an accident is detected.
- 📈 **REST APIs & WebSocket** for live frames (`/frame`) and event stream (`/events/ws`).
- 🔐 **JWT-based auth** with login endpoint and Swagger docs (`/docs`) including “Authorize” button.
- 🗄️ **PostgreSQL/SQLite support** for event history & user accounts.
- ⚡ GPU-accelerated inference (YOLOv8 on CUDA if available).

---

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

pip install --upgrade pip
pip install -r requirements.txt

## 3) Download YOLOv8 weights

**Windows (PowerShell)**

```powershell
New-Item -ItemType Directory -Force -Path weights | Out-Null
Invoke-WebRequest "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt" -OutFile "weights\yolov8n.pt"
```

## 4) Quick desktop run (CLI)

python -m scripts.infer_accidents --config configs/infer.yaml

* ESC to exit.
* Creates `outputs/accidents.mp4` and `outputs/events.csv` if enabled.

## 5) Run API server (with DB + JWT)

uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000

Endpoints:

* Health: [http://localhost:8000/health](http://localhost:8000/health)
* Latest frame: [http://localhost:8000/frame](http://localhost:8000/frame)
* Login (POST): [http://localhost:8000/login](http://localhost:8000/login)  → returns JWT
* List persisted events (GET, JWT): [http://localhost:8000/events](http://localhost:8000/events)
* Live events (WebSocket): `ws://localhost:8000/events/ws`

## 6) Run dashboard (Next.js)


cd dashboard
npm install
npm run dev
# open http://localhost:3001

✅ That’s it!

