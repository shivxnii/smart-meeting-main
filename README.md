
# 🎙️ Smart Meeting — AI-Powered Active Speaker Detection & Live Subtitles

> **Real-time AI meeting assistant that detects who is speaking, tracks participants, automatically focuses the camera on the active speaker, and generates live multilingual subtitles.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Server-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Whisper](https://img.shields.io/badge/Whisper-Speech%20Recognition-412991?style=for-the-badge)](https://github.com/openai/whisper)

---

## 📌 Overview

**Smart Meeting** is an AI-powered real-time meeting monitoring system designed to understand **who is speaking and what is being said**.

The system combines computer vision, audio processing, active speaker detection, face tracking, and speech recognition into a single pipeline.

It continuously captures:

* 🎥 **Video** from a webcam
* 🎤 **Audio** from a microphone
* 👤 **Faces** from the video stream
* 🗣️ **Active speaker information** using TalkNet
* 📝 **Speech-to-text** using Whisper

The system then dynamically identifies the active speaker, focuses the output frame around that person, and displays their speech as live subtitles.

---

## ✨ Key Features

### 👥 Real-Time Person Detection & Tracking

Detects multiple faces in the camera frame and maintains persistent speaker/participant IDs while they remain visible.

### 🗣️ Active Speaker Detection

Uses **TalkNet-ASD** to determine which detected face is actually speaking by analyzing the relationship between:

* Facial/lip movement
* Audio features
* Temporal audio-video alignment

### 🎯 Automatic Speaker Focus

When a person starts speaking, the system automatically reframes the video around the active speaker instead of keeping the entire meeting frame fixed.

### 📝 Live Speech-to-Text

Uses **OpenAI Whisper** to transcribe speech in real time.

The current configuration supports:

* Hindi
* English
* Hinglish
* Automatic language detection
* English subtitle translation

### 📊 Meeting Analytics

The dashboard tracks useful meeting-level statistics such as:

* Number of participants
* Current active speaker
* Speaker speaking time
* Number of speaker changes
* Meeting duration
* Speaker names

### 🎙️ Speaker Identification

Detected participants initially receive IDs such as:

```text
Person 1
Person 2
Person 3
```

Users can rename them from the dashboard.

Example:

```text
Person 1 → Shivani
Person 2 → Rahul
Person 3 → Aman
```

### 📜 Meeting Transcript

The system maintains a speaker-labelled transcript and provides an option to download the transcript as a `.txt` file.

### 🎥 Meeting Recording

The processed video stream can be recorded directly from the dashboard.

### ⚡ CPU-Compatible Processing

The system automatically selects the available PyTorch device:

```python
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Therefore, it can operate on CPU-only systems as well as CUDA-enabled machines.

---

# 🧠 System Architecture

```text
                    ┌─────────────────────┐
                    │      Webcam         │
                    │   Video Stream      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Face Detection    │
                    │       S3FD          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Face Tracking &     │
                    │ Participant IDs     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      TalkNet        │
                    │ Active Speaker      │
                    │     Detection       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Active Speaker      │
                    │     Selection       │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐        ┌──────────────────┐
       │ Speaker-Focused  │        │      Whisper     │
       │ Video Framing    │        │ Speech-to-Text   │
       └────────┬─────────┘        └────────┬─────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  Smart Meeting      │
                    │     Dashboard       │
                    ├─────────────────────┤
                    │ Live Video          │
                    │ Active Speaker      │
                    │ Subtitles           │
                    │ Participants        │
                    │ Analytics            │
                    │ Transcript           │
                    │ Recording            │
                    └─────────────────────┘
```

---

# 🔬 AI Pipeline

The complete processing pipeline works approximately as follows:

### 1. Video Capture

OpenCV continuously captures frames from the webcam.

```text
Webcam → Frame Buffer → AI Processing
```

The default configuration uses:

```text
Resolution: 640 × 480
Target FPS: 10
```

### 2. Face Detection

The system uses **S3FD** to detect faces in the camera stream.

Each detected face is assigned a persistent participant ID through the tracking system.

### 3. Active Speaker Detection

TalkNet analyzes synchronized audio and facial information to determine whether a particular detected person is speaking.

The system uses a temporal window rather than making an isolated decision from a single frame.

### 4. Speaker Stabilization

Speaker switching uses hysteresis and temporal logic to reduce rapid speaker flickering caused by noisy predictions.

### 5. Speaker-Focused Framing

Once the active speaker is identified, the renderer calculates a centered crop around the speaker and smoothly adjusts the camera framing.

```text
Full Meeting View
       ↓
Active Speaker Detection
       ↓
Speaker Bounding Box
       ↓
Smooth Reframing
       ↓
Focused Speaker View
```

### 6. Speech Recognition

Audio is processed independently by Whisper so that transcription does not block the active-speaker pipeline.

The project currently uses:

```text
Whisper Base
16 kHz Audio
CPU Inference
```

### 7. Speaker-Aware Subtitles

The system combines the speaker timeline with Whisper transcription.

Example:

```text
[10:42:31] Shivani: We need to complete the project today.
[10:42:36] Rahul: I'll handle the backend integration.
```

---

# 🛠️ Technology Stack

| Component                | Technology              |
| ------------------------ | ------------------------ |
| Programming Language     | Python                  |
| Computer Vision          | OpenCV                  |
| Face Detection           | S3FD                    |
| Active Speaker Detection | TalkNet-ASD             |
| Deep Learning            | PyTorch                 |
| Speech Recognition       | OpenAI Whisper          |
| Audio Capture            | SoundDevice             |
| Numerical Processing     | NumPy                   |
| Web Backend              | Flask                   |
| Frontend                 | HTML / CSS / JavaScript |
| Video Processing         | OpenCV                  |
| Hardware Input           | Webcam + Microphone     |

---

# 📁 Project Structure

```text
smart-meeting-main/
│
├── app.py
│
├── smart_meeting/
│   ├── config.py
│   ├── speaker_engine.py
│   ├── talknet_cpu.py
│   ├── subtitle_engine.py
│   └── ...
│
├── model/
│   └── ...
│
├── utils/
│   └── ...
│
├── TalkNet-ASD-main/
│   └── ...
│
├── _original/
│   └── TalkNet-ASD-main/
│
├── README.md
├── .gitignore
└── ...
```

The main application entry point is:

```text
app.py
```

The repository also contains the TalkNet implementation, model components, utility modules, and supporting files.

---

# ⚙️ Configuration

Most runtime parameters are centralized inside:

```text
smart_meeting/config.py
```

Important configuration parameters include:

```python
CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

TARGET_FPS = 10

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

WHISPER_MODEL = "base"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000
```

The system automatically checks whether CUDA is available and otherwise falls back to CPU processing.

---

# 📦 Model Weights

The configuration expects the following model files:

```text
weights/
├── pretrain_TalkSet.model
└── sfd_face.pth
```

These paths are defined in `smart_meeting/config.py`.

> **Note:** Model weights are not included in this repository when they are too large for normal GitHub storage. Download the required weights separately and place them inside the `weights/` directory.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/shivxnii/smart-meeting-main.git
cd smart-meeting-main
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

Install the required Python packages used by the project:

```bash
pip install numpy
pip install opencv-python
pip install flask
pip install sounddevice
pip install torch
pip install openai-whisper
pip install python_speech_features
```

Depending on your TalkNet/S3FD environment, additional dependencies may be required by the model implementation.

---

# 🎧 Microphone Configuration

By default, the system uses the operating system's default microphone:

```python
AUDIO_DEVICE = None
```

To see available audio devices:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Then specify the desired microphone index in:

```text
smart_meeting/config.py
```

For example:

```python
AUDIO_DEVICE = 1
```

---

# ▶️ Run the Application

Start the application using:

```bash
python app.py
```

When initialization is complete, the Flask server will start at:

```text
http://127.0.0.1:5000
```

Open the URL in your browser.

---

# 🖥️ Dashboard

The application provides a browser-based dashboard containing:

### Live Video

Displays the processed camera feed with active-speaker framing.

### Active Speaker

Shows the participant currently identified as speaking.

### Participants

Displays the number of detected participants and their IDs/names.

### Live Subtitles

Displays the latest Whisper transcription.

### Meeting Analytics

Provides:

```text
Meeting Duration
Participants
Speaker Changes
Speaker Speaking Time
```

### Transcript

Provides access to the accumulated speaker-labelled transcript.

### Recording

Allows the processed meeting video to be started and stopped from the dashboard.

---

# 🔌 API Endpoints

The Flask application exposes several endpoints.

| Endpoint               | Method | Purpose                            |
| ----------------------- | -----: | ----------------------------------- |
| `/`                     |    GET | Main dashboard                     |
| `/video_feed`           |    GET | Live processed video stream        |
| `/status`               |    GET | Current speaker/participant status |
| `/subtitle_text`        |    GET | Latest subtitle                    |
| `/transcript`           |    GET | Meeting transcript history         |
| `/meeting_stats`        |    GET | Meeting analytics                  |
| `/rename_speaker`       |   POST | Rename participant                 |
| `/download_transcript`  |    GET | Download transcript                |
| `/recording/start`      |   POST | Start recording                    |
| `/recording/stop`       |   POST | Stop recording                     |

These routes are implemented directly in `app.py`.

---

# 🌐 Example API Response

### `/status`

```json
{
  "persons_detected": 2,
  "active_speaker_id": 1,
  "speaking_score": 0.82,
  "is_speaking": true,
  "attendees": []
}
```

### `/meeting_stats`

```json
{
  "duration_formatted": "12:42",
  "speaker_changes": 7,
  "participants": 3,
  "speaker_times": []
}
```

---

# 🌍 Multilingual Speech Processing

Smart Meeting is designed for multilingual meetings.

The current Whisper configuration automatically detects the spoken language and translates the resulting speech into English subtitles.

Supported scenarios include:

```text
English
Hindi
Hinglish
```

For example:

```text
Spoken:
"Hum kal project complete kar lenge."

Subtitle:
"We will complete the project tomorrow."
```

The subtitle engine also includes silence filtering and safeguards against common Whisper hallucinations caused by background noise or silent audio.

---

# ⚡ Performance Design

The system separates major operations into independent processing threads:

```text
Camera Thread
      │
      ├──► Frame Capture
      │
      ▼
Director Thread
      │
      ├──► Face Detection
      ├──► Tracking
      └──► TalkNet
      │
      ▼
Render Thread
      │
      └──► Speaker-Focused Video
      │
      ▼
Analysis Thread
      │
      └──► Meeting Analytics
      │
      ▼
Whisper Thread
      │
      └──► Speech-to-Text
```

This architecture prevents slow Whisper transcription from blocking active-speaker detection and video processing.

---

# 🎯 Use Cases

Smart Meeting can be used for:

* 🏢 Corporate meetings
* 🎓 Classrooms and lectures
* 💻 Online/hybrid meetings
* 🎤 Panel discussions
* 🧑‍💼 Interview rooms
* 🏫 Smart classrooms
* 📝 Automatic meeting transcription
* 🎥 Speaker-focused video recording
* ♿ Accessibility and live captioning

---

# 🔮 Future Improvements

Planned/improvable areas include:

* [ ] GPU acceleration optimization
* [ ] Improved multi-person tracking
* [ ] Speaker re-identification across longer meetings
* [ ] Face recognition with user profiles
* [ ] Persistent meeting database
* [ ] Searchable meeting transcripts
* [ ] Automatic meeting summaries
* [ ] Action-item extraction
* [ ] Emotion/context analysis
* [ ] Cloud deployment
* [ ] WebSocket-based real-time updates
* [ ] Multi-camera support
* [ ] Better low-light face detection
* [ ] Advanced speaker-focused camera control

---

# ⚠️ Current Limitations

Because the application combines several computationally expensive AI components, CPU-only systems may experience lower real-time performance.

In particular:

* TalkNet + S3FD + Whisper can be computationally demanding.
* Whisper transcription is intentionally processed at intervals rather than on every frame.
* Webcam and microphone synchronization can affect active-speaker accuracy.
* Performance depends on camera quality, lighting, microphone quality, and hardware.
* Large model weights need to be downloaded separately.

---

# 🔐 Privacy

Smart Meeting is designed around local processing.

The core pipeline captures the webcam and microphone locally and processes the streams through the local application.

Before deploying in real meetings, obtain appropriate consent from participants and follow applicable organizational privacy policies and recording laws.

---

# 👨‍💻 Author

**Shivani Sharma**

B.Tech CSE — Artificial Intelligence & Machine Learning

GitHub: [@shivxnii](https://github.com/shivxnii)

LinkedIn: [Shivani Sharma](https://www.linkedin.com/in/shivani-sharma-1a7737289/)

---

# ⭐ Project

If you find this project useful or interesting, consider giving it a ⭐ on GitHub.

**Repository:**
https://github.com/shivxnii/smart-meeting-main

---

## 📄 License

Add the appropriate license for your project before distributing it publicly.

---

> **Smart Meeting** — *Making meetings more intelligent, focused, and accessible with AI.*
