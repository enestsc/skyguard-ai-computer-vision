# 🦅 SkyGuard: AI Computer Vision

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-Object_Detection-yellow)
![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-purple)
![Computer Vision](https://img.shields.io/badge/AI-Computer_Vision-success)

SkyGuard is an AI-powered computer vision project focused on real-time **aeroplane type object detection**. Leveraging the power of the YOLO (You Only Look Once) architecture, this system is designed to accurately detect and classify airplanes within visual data.

All datasets used for training and validation have been heavily curated, annotated, and augmented using **Roboflow**.

## ✨ Features
* **Real-time Object Detection:** High-speed inference using YOLO models tailored for aviation detection.
* **Heavy Data Augmentation:** The dataset was expanded to over 9,000 images using custom noise, brightness, and exposure adjustments to improve model robustness in various weather and lighting conditions.
* **Custom Dataset Integration:** Seamlessly trained on datasets structured via Roboflow.

## 🛠️ Technologies & Tools
* **Programming Language:** Python
* **Computer Vision:** YOLO, OpenCV
* **Data Management & Augmentation:** Roboflow

## 📂 Dataset
The dataset for this project was expertly annotated and generated using Roboflow. 
* **Target:** Aeroplane Types
* **Size:** 9,000+ images (after augmentation)
* **Dataset Link:** You can view/download the original dataset format via Roboflow [here](https://universe.roboflow.com/studia-lavg4/aeroplane-type-object-detection).

## 🚀 Installation & Usage

**Step 1: Clone the repository**
```bash
git clone https://github.com/enestsc/skyguard-ai-computer-vision.git
cd skyguard-ai-computer-vision
```

**2. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the detection script:**
```bash
python detect.py --source data/images --weights best.pt
```

## 📊 Results / Inference Examples
![Demo](demo.png)
