#  Real-Time Violence Detection & Threat Escalation System

An automated, edge-optimized CCTV surveillance pipeline that detects public violence in real-time using skeletal tracking, optical flow verification, and concurrent weapon detection, instantly dispatching forensic video evidence to security personnel via Telegram.

![Accuracy](https://img.shields.io/badge/Accuracy-98.5%25-success)
![FPS](https://img.shields.io/badge/FPS-30-blue)
![Python](https://img.shields.io/badge/Python-3.10-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-FP16-orange)

##  Overview
Traditional surveillance systems rely on passive recording and human monitoring. Existing AI solutions often use heavy 3D-CNNs that process massive raw pixel volumes, leading to lag and high false-alarm rates in cluttered environments. 

This project introduces a **novel, lightweight multi-stage architecture** that shifts from heavy pixel processing to mathematical skeletal abstraction. By integrating YOLOv8, MediaPipe, and an LSTM network with asynchronous processing, the system achieves state-of-the-art accuracy at real-time speeds (25-30 FPS) on edge hardware. Upon detecting a threat, the system automatically archives the forensic clip and alerts authorities in real-time.

##  Key Innovations & Novelty
- **Skeletal Abstraction:** Extracts a 132-dimensional topological vector per frame, making the LSTM immune to background clutter and drastically reducing computational load.
- **Dynamic 30% Subject Cropping:** Automatically expands YOLO bounding boxes by 30% to ensure high-velocity, erratic limbs (punches/kicks) remain tracked during violent acts.
- **Optical Flow Dual-Verification:** Cross-verifies LSTM predictions with Farneback Optical Flow pixel displacement, successfully eliminating false alarms caused by static overlapping skeletons (e.g., hugging).
- **Context-Aware Threat Escalation:** Fuses behavioral prediction with concurrent weapon detection (bats/knives) to dynamically escalate the alert status (`NORMAL` ➡️ `VIOLENCE` ➡️ `CRITICAL`).
- **Automated Forensic Alerting:** Instantly captures and locally stores video evidence of confirmed violent incidents, seamlessly dispatching real-time, severity-graded alerts directly to investigative officers via a Telegram Bot API.
- **Asynchronous Frame-Skipping:** Orchestrates heavy AI models on alternating frames, cutting computational overhead by >50%.

##  Performance Metrics
Tested on a dataset of ~1,335 continuous video sequences.
- **Accuracy:** 98.50%
- **Recall (Sensitivity):** 99.25% *(Extremely low False Negative rate)*
- **Precision:** 97.78%
- **F1-Score:** 98.51%
- **Inference Speed:** - GPU (NVIDIA RTX 5060 + CUDA): **25–30 FPS**
  - CPU-Only (Edge Simulation): **15–22 FPS**

##  System Architecture
*(Insert your architecture diagram image here. Upload the image to your repository and replace this text with: `![Architecture Diagram](link_to_your_image.png)`)*

1. **Subject Localization:** YOLOv8s (Class 0, 34, 43) localizes humans and weapons.
2. **Dynamic Crop:** Bounding boxes expanded by 30% and resized to 256x256.
3. **Pose Extraction:** MediaPipe extracts 33 keypoints (132-D vector).
4. **Temporal Classification:** 30-frame sequence passed through a 2-Layer LSTM (Hidden: 256).
5. **Context Fusion:** Output probability modulated by Optical Flow and YOLO weapon buffer.
6. **Dispatch & Evidence Storage:** Logs the incident, saves a forensic video clip, and transmits a real-time Telegram alert to security personnel.

##  Installation & Usage

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU (Highly Recommended for 30 FPS)
- Telegram Bot Token (for alert dispatch)

### Setup
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/Violence-Detection-System.git](https://github.com/YOUR_USERNAME/Violence-Detection-System.git)
cd Violence-Detection-System

# Install dependencies
pip install -r requirements.txt

Running the System
Bash
# To run on live webcam
python src/main.py --source 0

# To run on a specific video file
python src/main.py --source path/to/cctv_footage.mp4

 Future Scope
Privacy Compliance (GDPR): Automated face-blurring pipeline on forensic video clips before they are dispatched or stored.
Edge Hardware Scaling: TensorRT optimization for decentralized deployment on specialized edge AI hardware (e.g., NVIDIA Jetson).
Multimodal AI: Acoustic analysis (detecting breaking glass, screams, or gunshots) fused with visual data for a denser threat verification matrix.

 Contributors
Adithya Kiran
Asraya Ajay
Chethan P
Sreya S

Developed at the Department of Artificial Intelligence & Data Science, Muthoot Institute of Technology and Science, Kochi.
