# animalsoundclassification

Dog Sound Classifier — How to Run

Step 1: Install Required Libraries
Run this in your terminal or command prompt:
pip install tensorflow numpy sounddevice scipy librosa matplotlib seaborn scikit-learn joblib pyttsx3 keyboard

Step 2: Dataset Folder Structure
Make sure your Dataset/ folder is arranged like this:
Dataset/
├── dog_bark_train/
├── dog_bark_test/
├── dog_growl_train/
├── dog_growl_test/
├── dog_grunt_train/
├── dog_grunt_test/
Each folder should contain .wav audio files.

Step 3: Train the Model
Run the training script (only once unless you update the dataset):
python cnn_classifier.py

This will:
- Train a CNN model
- Save the model as dog_sound_classifier.h5
- Save label encoder as label_encoder.pkl

Step 4: Real-Time Sound Classification
Run the real-time classifier with TTS (text-to-speech):
python translator.py

Controls During Real-Time Classification:
G - Start listening
S - Stop & classify the sound
Q - Quit the program

Output Files:
- Each recording is saved in the recordings/ folder
- All predictions are logged in prediction_log.txt
- The prediction is spoken out loud using your system voice
