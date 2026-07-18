# 🚶 Gait Analysis

A computer-vision-based project for analyzing human walking patterns using pose estimation and movement tracking.

The project processes video data to identify body landmarks and analyze gait-related characteristics such as posture, joint movement, walking patterns, and body alignment.

## ✨ Features

* Human pose detection
* Body landmark tracking
* Walking-pattern analysis
* Joint movement analysis
* Gait visualization
* Video-based motion analysis
* Computer-vision-based processing

## 🛠️ Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* Computer Vision
* Pose Estimation
* Human Motion Analysis

## ⚙️ How It Works

```text
Walking Video
      │
      ▼
Video Frame Extraction
      │
      ▼
Human Pose Detection
      │
      ▼
Body Landmark Tracking
      │
      ▼
Joint and Movement Analysis
      │
      ▼
Gait Analysis Results
```

The system processes video frames, detects body landmarks, and analyzes movement patterns to provide information about a person's gait.

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/SahilAcharya13/gait-analysis.git

cd gait-analysis
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is unavailable, install the commonly required packages:

```bash
pip install opencv-python mediapipe numpy
```

## ▶️ Usage

Run the main Python application:

```bash
python main.py
```

> If the main file has a different name, replace `main.py` with the appropriate filename.

Provide a walking video or enable camera input to begin gait analysis.

## 📊 Applications

* Human movement analysis
* Sports-performance analysis
* Walking-pattern research
* Rehabilitation research
* Posture and mobility assessment

## 🔮 Future Improvements

* Add real-time gait analysis
* Calculate joint angles
* Add step and stride detection
* Measure walking speed
* Generate gait-analysis reports
* Improve movement visualization

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. The generated results should not be considered a medical diagnosis or used as a replacement for professional medical assessment.
