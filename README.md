# 🎧 Speech Emotion Recognition with Deep Learning

This project performs emotion classification from human speech using a deep learning model. It uses audio features extracted from .wav files and classifies them into one of 8 emotions. A Streamlit web application allows users to upload their own audio and receive real-time emotion predictions.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Features Used](#features-used)
- [Emotions Covered](#emotions-covered)
- [Streamlit App Workflow](#streamlit-app-workflow)
- [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
- [Sample Output](#sample-output)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## 🧠 Overview

This end-to-end pipeline performs emotion recognition from speech. The system uses MFCCs and temporal features to train a CNN + BiLSTM + Attention-based deep learning model that achieves strong performance across 8 emotion classes.

The web app is implemented using Streamlit and allows users to upload a .wav file and receive emotion predictions instantly.

---

## 🏗️ Model Architecture

- Conv1D layers for local time-frequency pattern extraction
- BiLSTM to capture temporal dependencies
- Custom Attention layer to focus on emotionally relevant frames
- Dense output layer with softmax activation

Training was performed with:
- Random oversampling for class balancing
- Class weighting
- Learning rate scheduler
- Early stopping

📁 Model file: Emotion_classifier.h5

---

## 🎚️ Features Used

Each audio clip is transformed into a fixed-size 2D feature matrix:

- MFCC (40 coefficients)
- Delta MFCC (40)
- Delta-Delta MFCC (40)
- RMS Energy (1)

Total = 121 features per time frame  
Input shape to the model: (248, 121)

---

## 😊 Emotions Covered
01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)

| Label | Emotion     |
|-------|-------------|    |
| 1     | Neutral     |
| 2     | Calm        |
| 3     | Happy       |
| 4     | Sad         |
| 5     | Angry       |
| 6     | Fearful     |
| 7     | Disgust     |
| 8     | Surprised   |
-----------------------

## 🌐 Streamlit App Workflow

1. User uploads a `.wav` file.
2. The audio is saved temporarily and passed to the feature extractor.
3. The model loads and predicts emotion.
4. The predicted label is shown on the UI.

💡 The model is loaded using a custom AttentionLayer class to support saved .h5 format.

📄 app file: web_app.py

---

## ⚙️ Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/audio-emotion-classification.git
   cd audio-emotion-classification
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure your Python version is between 3.8 and 3.10 for TensorFlow compatibility.

---

## ▶️ How to Use

To launch the Streamlit app:

```bash
streamlit run web_app.py
```

Then open your browser at http://localhost:8501  
Upload a .wav file and the predicted emotion will be displayed.

---

## ✅ Sample Output

After uploading a file, you’ll see:

🧠 Predicted Emotion: happy

---

## 📈 Results

- Achieved class-wise F1-scores ≥ 75% on 7/8 emotions
- Used RandomOversampling and Attention for boosting low-recall classes (e.g., sad, fearful)
- Fully deployable as a web app

---

## 🔭 Future Improvements

- Add multi-language emotion recognition
- Incorporate noise robustness (e.g., real-time audio, phone recordings)
- Deploy on Hugging Face Spaces or Streamlit Cloud
- Export model to ONNX or TFLite for mobile apps

---

## 🙌 Acknowledgements

- TensorFlow and Keras for model training
- Librosa for audio processing
- Streamlit for UI
- scikit-learn + imbalanced-learn for oversampling
- Your effort in carefully balancing performance and deployment!
