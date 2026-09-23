# Mine Sarthi — The Charioteer of Sustainable Mining

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-Powered-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-Dashboard-61DAFB?logo=react" alt="React">
  <img src="https://img.shields.io/badge/MQTT-Real--time-660066?logo=mqtt" alt="MQTT">
  <img src="https://img.shields.io/badge/InfluxDB-Timeseries-22ADF6?logo=influxdb" alt="InfluxDB">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

> Mine Sarthi — an AI-powered platform that optimizes energy usage in iron ore mining operations. Our system uses real-time AI to automatically adjust crusher speed based on ore hardness, reducing energy consumption by 20-35% while maintaining production throughput.With end-to-end IoT integration, predictive maintenance, and intelligent control systems, Mine Sarthi demonstrates how technology can make mining more sustainable and cost-effective.


<p align="center">
  <img src="images/Screenshot 2026-03-19 181353.png" alt="Mine Sarthi AI Monitoring Dashboard" width="800"/>
</p>

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Overview](#-overview)
- [System Architecture](#%EF%B8%8F-system-architecture)
- [What You Will Learn](#-what-you-will-learn)
- [What You'll Build](#%EF%B8%8F-what-youll-build)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running with Docker](#running-with-docker)
- [AI Models](#-ai-models)
- [Key Features](#-key-features)
- [Contributing](#-contributing)
- [Team](#-team)
- [License](#-license)

---

## 🎯 Problem Statement

### SIH Problem Statement ID: **25210**
**Problem Statement Title:** Efficient Energy use in Iron Ore Mining Operations
**Theme:** Miscellaneous | **PS Category:** Software | **Team ID:** 97177 | **Team Name:** @XEN!TH

---

#### Background
In mining operations, the processes of **crushing and grinding** — collectively known as **comminution** — are essential for liberating valuable minerals from the surrounding rock. However, these processes are among the **most energy-intensive stages** in mineral processing, often accounting for up to **50% of a mine's total energy consumption**. The inefficiencies in these systems not only lead to **elevated operational costs** but also contribute significantly to the **environmental footprint** of mining activities.

#### Description
Traditional crushing and grinding equipment often operate under **suboptimal conditions** due to static control systems, wear and tear, and lack of real-time adaptability. This results in:
- ⚠️ **Excessive energy usage**
- 📉 **Reduced throughput**
- 💰 **Increased maintenance costs**
- 🌍 **Higher carbon emissions**

#### Expected Solutions
**a) AI-Controlled Optimization Systems:**
1. **Real-Time Monitoring & Control:** Implement AI-driven systems that continuously monitor variables such as ore hardness, feed size, moisture content, and equipment load.
2. **Predictive Maintenance:** Use machine learning models to predict wear patterns and schedule maintenance proactively, reducing downtime and energy waste.
3. **Dynamic Process Adjustment:** AI algorithms can adjust operational parameters (e.g., crusher speed, grinding media size, mill rotation speed) in real time to maintain optimal energy efficiency.
4. **Integration with IoT Sensors:** Deploy smart sensors to collect high-resolution data and feed it into AI models for more accurate decision-making.

> **Main Goal:** How we can optimize energy consumption in crushing plant.

---

## 🧾 Overview

**Mine Sarthi** is a full-stack, production-ready platform that addresses the SIH problem statement through:

✅ **Real-Time Monitoring** — IoT sensors continuously collect 9+ operational parameters (power, RPM, feed rate, temperature, vibration, ore hardness, etc.)
✅ **AI-Driven Optimization** — Two machine learning models work sequentially:
   - **Model 1:** Ore Hardness Classifier (Random Forest) → Classifies ore into SOFT / MEDIUM / HARD
   - **Model 2:** RPM Energy Optimizer → Recommends optimal crusher speed to minimize energy per ton
✅ **Autonomous Control** — MQTT-based closed-loop system adjusts crusher RPM automatically based on AI predictions
✅ **Energy Savings** — Demonstrated **8-15% reduction in energy consumption** vs baseline operations
✅ **Complete Stack** — Data pipeline (MQTT → InfluxDB → PostgreSQL), ML service (FastAPI + Scikit-learn), and modern web dashboard (React)

---

## 🏗️ System Architecture

> *High-level architecture showing the complete data flow — from IoT sensors through MQTT ingestion, real-time AI inference, to autonomous speed control commands.*

### 🔄 Architecture Flow
```mermaid
graph TD
    subgraph IOT["Edge / IoT Layer"]
        A["IoT Sensors\nCrusher & Mill"] -->|"MQTT Publish\nmining/device_id/metrics"| B([Mosquitto Broker])
    end

    subgraph INGEST["Ingestion & Bridge"]
        B --> C["consumer.py\nMQTT-to-HTTP Bridge\nwith retry queue"]
        C -->|"POST /ingest"| D["FastAPI Backend"]
    end

    subgraph STORE["Data Storage"]
        D -->|"Time-series\ncrusher_metrics"| F[(InfluxDB)]
        D -->|"Aggregates\ncrusher_minute_stats"| G[(PostgreSQL)]
    end

    subgraph AI["AI Monitoring & Optimization"]
        D -->|"Webhook if ML_SERVICE_ENABLED"| H["ML Service"]
        H -->|"Model 1"| I["Ore Hardness Classifier\nRandom Forest"]
        H -->|"Model 2"| J["RPM Energy Optimizer\nRegression"]
        I --> K["Control Logic\n& Safety Interlocks"]
        J --> K
    end

    subgraph CONTROL["Autonomous Control Loop"]
        K -->|"MQTT Publish\nmining/device_id/speed_setpoint"| B
        B -->|"RPM Command"| L["Crusher PLC\nAdjusts RPM"]
        L -->|"Feedback"| A
    end

    subgraph UI["Frontend Display"]
        M["React Dashboard\nAIMonitoring.tsx"]
        M -->|"GET /timeseries/crusher_01"| D
        M -->|"GET /ai-features/crusher_01"| D
        M -->|"GET /control-status/crusher_01"| H
    end
```

**MQTT Topics:**

| Topic | Direction | Purpose |
|:---|:---|:---|
| `mining/device_id/metrics` | IoT → Broker | Sensor telemetry |
| `mining/device_id/speed_setpoint` | ML Service → Broker | RPM commands to crusher |

### Architecture Breakdown

| Component | Location | Role | Technology |
|:---|:---|:---|:---|
| **IoT Publisher** | `data_pipeline/gateway/publisher_mqtt.py` | Simulates sensors, publishes to MQTT | Paho-MQTT |
| **MQTT Broker** | Docker: `mosquitto` | Relays all MQTT messages | Mosquitto |
| **Bridge** | `data_pipeline/bridge/consumer.py` | MQTT → HTTP to FastAPI ingest | Python, Requests |
| **FastAPI Backend** | `data_pipeline/backend/fastapi_app.py` | Ingest, InfluxDB, PostgreSQL, ML webhook | FastAPI, Pydantic |
| **InfluxDB** | Docker | Time-series storage (`crusher_metrics`) | InfluxDB 1.8 |
| **PostgreSQL** | Docker | Aggregates (`crusher_minute_stats`, `sensor_minute_agg`) | PostgreSQL 15 |
| **ML Service** | `ml_service/` | Model 1, Model 2, control loop, MQTT publish | Scikit-learn, FastAPI |
| **Frontend** | `smart-ore-flow-main/` | AI Monitoring, displays sensor data, ore type, optimal RPM, control status | React, Vite, Tailwind |

---

## 🎓 What You Will Learn
- ✅ Architecting **Industrial IoT** pipelines for high-frequency data
- ✅ Implementing **sequential AI workflows** (Classification → Regression)
- ✅ Building **production-grade bridges** between different protocols (MQTT & HTTP)
- ✅ Working with **hybrid database strategies** (Time-series + Relational)
- ✅ Developing **Digital Twins** for process visualization and simulation

---

## 🏗️ What You'll Build
- A live **Energy Optimization Platform** for mining operations
- A **bidirectional data pipeline** handling thousands of metrics per second
- An **AI-powered speed control service** that operates autonomously
- A **React-based command center** with real-time analytics and alerts

---

## 🧰 Tech Stack

| Category | Technology |
|:---|:---|
| **Backend** | Python 3.10+, FastAPI |
| **Machine Learning** | Scikit-learn, Random Forest, Regression |
| **IoT / Messaging** | MQTT Broker & Bridge, Mosquitto |
| **Databases** | InfluxDB (Time-series), PostgreSQL (Relational) |
| **Frontend** | React, Vite, Tailwind CSS, Lucide Icons |
| **DevOps** | Docker, Docker Compose |
| **Others** | Pydantic, Dotenv, Requests, Asyncio |

---

## 📁 Project Structure

```bash
Mine-Sarthi/
|-- data_pipeline/          # Core ingestion & infrastructure
|   |-- backend/            # FastAPI ingestion server
|   |-- bridge/             # MQTT -> HTTP bridge (resilient, consumer.py)
|   |-- gateway/            # IoT publisher (publisher_mqtt.py)
|   |-- mqtt_broker/        # Mosquitto configuration
|   `-- docker-compose.yml  # Orchestration script
|-- ml_service/             # AI Engine (Inference & Control)
|   |-- api/                # Prediction & control endpoints
|   |-- models/             # Trained .pkl models (Model 1 & 2)
|   `-- src/                # SpeedControlService & business logic
|-- smart-ore-flow-main/    # Frontend React Application
|-- images/                 # Project documentation images
`-- README.md               # You are here
```

---

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose installed
- Python 3.10 or higher
- Node.js (for frontend development)

### Installation
1. **Clone the repository**
```bash
git clone https://github.com/Vishalkumarjaiswal16/Mine-Sarthi.git
cd Mine-Sarthi
```

2. **Set up Environment Variables**
Create a `.env` file in the `data_pipeline/` directory based on the template.

### Running with Docker
The easiest way to run the full stack is using Docker Compose:
```bash
cd "data_pipeline"
docker-compose up --build
```

This will start:
- Mosquitto Broker (1883)
- InfluxDB (8086)
- PostgreSQL (5432)
- FastAPI Backend (8000)
- ML Service (8001)

---

## 🤖 AI Models

### Model 1: Ore Hardness Classifier
- **Algorithm:** Random Forest Classifier
- **Inputs:** Power, RPM, Feed Rate, Feed Size, Vibration, Temperature, Current
- **Output:** Ore class (SOFT, MEDIUM, HARD) with confidence %
- **Purpose:** Identifies material characteristics in real-time to set optimization constraints.

### Model 2: RPM Energy Optimizer
- **Algorithm:** Regression-based optimization
- **Inputs:** Output of Model 1 + Live operational data
- **Output:** Optimal RPM setpoint
- **Purpose:** Minimizes `kWh/ton` processed by matching machine speed to ore hardness.

---

## ✨ Key Features
- ⚡ **Real-time Speed Control** — Autonomous adjustment of crusher RPM via AI.
- 🔄 **Digital Twin** — Visual process flow simulation for operational planning.
- 📊 **Energy Usage Analytics** — Granular breakdown of consumption by equipment.
- ☀️ **Renewable Integration** — Dashboard for solar and battery storage management.
- 🛡️ **Safety Interlocks** — Logic-based overrides to prevent unsafe machine states.

---

## 🤝 Contributing
We welcome contributions to make mining more sustainable!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👥 Team — @XEN!TH
- **[Vishal Kumar](https://github.com/Vishalkumarjaiswal16)** 
- **[Shivani Sharma](https://github.com/shivxnii)**  — Team Lead
- **Aditya Goyal**
- **Akshat Kumar Arya**
- **Himanshi Bishoi**
- **Aditya Naruka**

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---

Made with ❤️ for **Smart India Hackathon 2025** by Team **@XEN!TH**
