# Lung Disease Detector (Pneumonia Detection)

A deep learning-based web application that detects **Pneumonia** from chest X-ray images using a Convolutional Neural Network (CNN).  
Built with **TensorFlow/Keras** and deployed using **Streamlit**.

---

## Features

- Upload chest X-ray images (JPG, PNG, JPEG)
- AI prediction (Normal / Pneumonia)
- Confidence score display
- Fast and lightweight Streamlit interface

---

## Model Details

- Model: Convolutional Neural Network (CNN)
- Framework: TensorFlow (Keras API)
- Input Size: 100x100 grayscale images
- Classes:
  - Normal
  - Pneumonia

---

## Project Structure

```
lung-disease-project/
│
├── app.py                  # Streamlit web app
├── lung_disease_model.h5   # Trained model
├── requirements.txt        # Dependencies
├── README.md
│
└── dataset/ (NOT INCLUDED - ignored via gitignore)
```

---

## Installation

### Clone repo
```bash
git clone https://github.com/daniyalhussayin/Lung-Disease-Detector.git
cd Lung-Disease-Detector
```

---

### Create virtual environment (optional)
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Usage

1. Open app in browser  
2. Upload chest X-ray image  
3. Click **Predict**  
4. Get result with confidence score  

---

## Disclaimer

This project is for **educational purposes only** and should not be used for medical diagnosis.

---

## Author

Daniyal Hussain  
GitHub: https://github.com/daniyalhussayin

---

⭐ If you like this project, give it a star!
