import os
import joblib
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import keyboard
import pyttsx3
from datetime import datetime
from tensorflow import keras
from collections import deque


model_path = "dog_sound_classifier.h5"
label_encoder_path = "label_encoder.pkl"
sample_rate = 22050
chunk_duration = 0.5  # seconds

# Human-friendly label messages 
label_to_message = {
    "dog_bark": "I am alert",
    "dog_growl": "I am angry",
    "dog_grunt": "I am asking for food"
}

# Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 130)  # Lower the value the slower it will speak

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load Model & Encoder
if not os.path.exists(model_path) or not os.path.exists(label_encoder_path):
    raise FileNotFoundError("Model or label encoder not found.")

model = keras.models.load_model(model_path)
label_encoder = joblib.load(label_encoder_path)

# Feature Extraction 
def extract_features(file_path, max_pad_len=100):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print("Error extracting features:", e)
        return None

# Predict & Speak 
def predict_sound(file_path):
    feature = extract_features(file_path)
    if feature is None:
        return "Could not extract features"
    feature = feature[np.newaxis, ..., np.newaxis]
    prediction = model.predict(feature)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    meaning = label_to_message.get(predicted_label, "Unknown behavior")
    message = f"{predicted_label} - {meaning}"
    speak(meaning)  # Speak the interpretation
    return message

#Classification 
def controlled_classification():
    buffer = deque()
    print("Press [G] to start listening. Press [S] to stop and classify. Press [Q] to quit.")

    def callback(indata, frames, time, status):
        if status:
            print("Error:", status)
        buffer.append(indata.copy())

    stream = None
    listening = False

    try:
        while True:
            if keyboard.is_pressed("g") and not listening:
                print("Started listening...")
                buffer.clear()
                stream = sd.InputStream(callback=callback, samplerate=sample_rate, channels=1)
                stream.start()
                listening = True

            elif keyboard.is_pressed("s") and listening:
                print("Stopped. Classifying...")
                stream.stop()
                stream.close()
                listening = False

                # Save audio
                audio_data = np.concatenate(list(buffer), axis=0)
                os.makedirs("recordings", exist_ok=True)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                file_path = f"recordings/dog_sound_{timestamp}.wav"
                wav.write(file_path, sample_rate, (audio_data * 32767).astype(np.int16))

                # Predict and speak
                result = predict_sound(file_path)
                print(f"File saved: {file_path}")
                print(f"Prediction: {result}")

                # Log prediction
                with open("prediction_log.txt", "a") as log:
                    log.write(f"{timestamp} - {file_path} - {result}\n")

                print("\n Press [G] to record again or [Q] to quit.")

            elif keyboard.is_pressed("q"):
                print("Exiting.")
                if stream and stream.active:
                    stream.stop()
                    stream.close()
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        if stream and stream.active:
            stream.stop()
            stream.close()

# Run the real time classifier
controlled_classification()
