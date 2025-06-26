import streamlit as st
import numpy as np
import librosa
import os
import tensorflow as tf
from keras.models import load_model
#from utils import extract_features  # or use inline function
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Load model
model = tf.keras.models.load_model('Emotion_classifier.h5',custom_objects={'AttentionLayer': AttentionLayer})

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Your original feature extraction method
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    rms = librosa.feature.rms(y=audio)

    combined = np.vstack([mfcc, delta, delta2, rms])
    
    # Transpose to (time_steps, features)
    combined = combined.T

    # Pad or truncate to exactly 248 frames
    required_frames = 248
    if combined.shape[0] < required_frames:
        pad_width = required_frames - combined.shape[0]
        combined = np.pad(combined, ((0, pad_width), (0, 0)), mode='constant')
    else:
        combined = combined[:required_frames, :]

    # Final shape should be (1, 248, 121)
    combined = np.expand_dims(combined, axis=0)

    return combined

# Define emotion labels
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful','disgust' ,'surprised']  # update if needed

# App UI
st.title("🎙️ Speech Emotion Recognition")
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Save temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Extract features
    features = extract_features("temp.wav")
    #features = np.expand_dims(features, axis=0)  # or match the shape used in your model

    # Predict
    prediction = model.predict(features)
    predicted_label = emotion_labels[np.argmax(prediction)]

    st.markdown(f"### 🧠 Predicted Emotion: `{predicted_label}`")
    os.remove("temp.wav")