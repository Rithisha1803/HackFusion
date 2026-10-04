# HACKFUSION - OPEN INNOVATION - AI/ML
# AuthentiHire :  AI-Powered Remote Interview Proctoring Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-000000?style=for-the-badge&logo=flask&logoColor=white)
![WebRTC](https://img.shields.io/badge/WebRTC-Live%20Video-333333?style=for-the-badge&logo=webrtc&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

**A real-time AI proctoring system for remote interviews — combining WebRTC video, YOLOv8 object detection, MediaPipe gaze tracking, FER emotion recognition, DeepFace identity verification, and Librosa voice analysis into one seamless hiring pipeline.**

[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-FF6B35?style=flat-square)](https://ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Gaze%20Tracking-00BFA5?style=flat-square)](https://mediapipe.dev/)
[![DeepFace](https://img.shields.io/badge/DeepFace-Identity%20Verification-E53935?style=flat-square)](https://github.com/serengil/deepface)
[![FER](https://img.shields.io/badge/FER-Emotion%20Recognition-7C4DFF?style=flat-square)](https://github.com/justinshenk/fer)
[![Librosa](https://img.shields.io/badge/Librosa-Voice%20Analysis-43A047?style=flat-square)](https://librosa.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Frame%20Extraction-5C3EE8?style=flat-square)](https://opencv.org/)

</div>

---

##  Overview

**AuthentiHire** is a fully automated remote hiring platform that conducts live video interviews via WebRTC and simultaneously runs a multi-model AI surveillance pipeline. Every session is proctored in real time — detecting proxy candidates, monitoring gaze and attention, analyzing emotions and voice, and verifying identity — then compiled into a detailed analytical report linked to a unique Meeting ID.

> No plugins. No manual review. Just intelligent, trustworthy remote hiring.

---

## ✨ Features

| Feature | Details |
|--------|---------|
| Live Video Interviews | WebRTC-based in-browser video calls — no setup needed |
| Identity Verification | DeepFace matches candidate against pre-registered profile photo |
| Emotion Recognition | FER tracks 7 emotions across the full session timeline |
|  Gaze Detection | `gaze_detector.py` + MediaPipe detects off-screen attention |
| Multi-Person Detection | YOLOv8 (`yolov8n.pt`) flags unauthorized persons in frame |
| Voice Analysis | Librosa detects stress, pace, and audio anomalies |
| Post-Interview Report | AI signals aggregated by Meeting ID → `view_results.html` |
| Dual-Portal Access | Separate flows for candidates and clients (recruiters) |
| Local Databases | `candidates.db` for profiles, `ai_results.db` for analysis data |

---

##  System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     CANDIDATE BROWSER                         │
│              WebRTC Live Video  (meeting.html)                │
└────────────────────────┬──────────────────────────────────────┘
                         │  A/V Stream
                         ▼
┌───────────────────────────────────────────────────────────────┐
│                      app.py  (Flask)                          │
│                                                               │
│   ┌─────────────────────────────────────────────────────┐     │
│   │         OpenCV — Frame Extraction Engine            │     │
│   └──────────┬──────────────────────────────────────────┘     │
│              │  Frames dispatched in parallel                 │
│    ┌─────────┼──────────┬─────────────┬───────────┐           │
│    ▼         ▼          ▼             ▼           ▼           │
│ ┌──────┐ ┌───────┐ ┌────────┐ ┌──────────┐ ┌──────────┐      │
│ │ YOLO │ │Gaze   │ │  FER   │ │DeepFace  │ │ Librosa  │      │
│ │v8n.pt│ │Detect.│ │Emotions│ │ Identity │ │  Audio   │      │
│ └──┬───┘ └───┬───┘ └───┬────┘ └────┬─────┘ └────┬─────┘      │
│    └─────────┴──────────┴───────────┴────────────┘            │
│                         │                                     │
│              ┌──────────▼──────────┐                          │
│              │  Aggregation Engine │                          │
│              │  (Meeting ID index) │                          │
│              └──────────┬──────────┘                          │
│                         │                                     │
│         ┌───────────────▼──────────────────┐                  │
│         │  ai_results.db  ←→  candidates.db│                  │
│         └───────────────┬──────────────────┘                  │
│                         │                                     │
│              ┌──────────▼───────────┐                         │
│              │   view_results.html  │  (Client Dashboard)     │
│              └──────────────────────┘                         │
└───────────────────────────────────────────────────────────────┘
```

---


##  AI Models

### `gaze — Gaze & Attention Tracking
- Built on **MediaPipe Face Mesh** with 468 facial landmarks
- Computes iris position to determine gaze direction (left / right / up / down / center)
- Calculates head yaw and pitch for off-screen attention detection
- Raises flags when the candidate looks away for longer than a configurable threshold

###  YOLOv8 (`yolov8n.pt`) — Multi-Person & Object Detection
- Lightweight YOLOv8 Nano model for real-time inference
- Detects multiple persons → triggers proxy candidate alert
- Identifies unauthorized objects: phones, earphones, printed notes
- Every detection is timestamped and stored in `ai_results.db`

### FER — Facial Emotion Recognition
- Classifies 7 emotions: `Happy` `Sad` `Angry` `Fear` `Disgust` `Surprise` `Neutral`
- Runs per-frame and builds an emotion timeline for the full session
- Sudden stress spikes or inconsistency patterns are flagged as anomalies

###  DeepFace — Identity Verification
- Compares live frames against the candidate's pre-registered photo in `uploads/`
- Supports multiple backends: `ArcFace`, `VGG-Face`, `Facenet512`
- Identity mismatch beyond the configured threshold → flagged immediately

### 🎙️ Librosa — Voice & Audio Analysis
- Extracts **MFCCs**, pitch, spectral centroid, and zero-crossing rate
- Detects speech stress, unnatural pauses, and anomalous tone shifts
- Flags possible pre-recorded audio playback attempts

---

##  User Portals

### 👤 Candidate Portal
| Page | Purpose |
|------|---------|
| `candidatesignup.html` | Create account with photo upload |
| `candidatelogin.html` | Authenticate and access dashboard |
| `candidateportal.html` | View scheduled interviews, join meeting |
| `edit_profile.html` | Update personal details and photo |
| `profile.html` | View profile summary |
| `meeting.html` | Live WebRTC interview room |

### 🏢 Client / Recruiter Portal
| Page | Purpose |
|------|---------|
| `clientsignup.html` | Recruiter registration |
| `clientlogin.html` | Recruiter authentication |
| `clientportal.html` | Manage candidates and schedule interviews |
| `analytics.html` | Real-time and historical AI analytics |
| `view_results.html` | Post-interview reports by Meeting ID |

---



---

##  Getting Started

### Prerequisites
- Python 3.10+
- Webcam & Microphone
- Modern browser with WebRTC support (Chrome / Edge /Safari recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/trivector.git
cd trivector
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

##  Key Dependencies

```
flask
opencv-python
mediapipe
ultralytics          # YOLOv8
deepface
fer
librosa
torch
numpy
Pillow
```

>  A CUDA-compatible GPU is strongly recommended for real-time multi-model inference.

---

##  Post-Interview Report (`view_results.html`)

Each report is uniquely indexed by **Meeting ID** and contains:

- **Session Timeline** — full chronological event log
-  **Emotion Distribution** — percentage breakdown across 7 emotions
-  **Gaze Attention Score** — attention tracking over interview duration
-  **Identity Match Score** — per-frame confidence with flagged drops
-  **Multi-Person Events** — timestamps of unauthorized presence
-  **Voice Anomaly Markers** — audio stress points with timestamps
- **Anomaly Summary** — all flagged events with severity levels
-  **Trust Score** — composite integrity score (0–100)

---

##  Privacy & Data Handling

- All video communication is encrypted via **DTLS-SRTP** (WebRTC standard)
- Candidate photos and session data are stored locally — never sent to third-party APIs
- Upload folders are namespaced per candidate (`uploads/<candidate_id>/`)
- Session data in `ai_results.db` can be purged post-review at admin discretion

---

##  Roadmap

- [ ] Whisper integration for real-time speech-to-text transcription
- [ ] LLM-based answer quality scoring
- [ ] Export reports as PDF
- [ ] Live recruiter alert dashboard during interviews
- [ ] Multi-interview scheduling calendar
- [ ] Role-based admin panel

---





*Secure hiring. Smart proctoring. Zero compromise.*


</div>
