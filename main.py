import numpy as np
import sounddevice as sd
import librosa
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.lang import Builder
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Parameters
fs = 44100  # Target sampling rate
n_mfcc = 13  # Number of MFCC features
n_fft = 2048  # Interval to apply FFT
hop_length = 512  # Sliding window for FFT
num_segments = 2  # Number of segments to divide each audio file
max_length = 300  # Maximum length of MFCC features (number of frames)
model_save_path = 'trained_model.h5'  # Path to the trained model

# Load and compile the trained model
model = load_model(model_save_path)
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Kivy Layouts
Builder.load_string("""
<RootWidget>:
    orientation: 'vertical'
    padding: 10
    spacing: 10
    Label:
        id: title_label
        text: 'Heart Rate Estimation'
        font_size: 30
    Button:
        text: 'Record Audio'
        size_hint_y: None
        height: 50
        on_press: root.record_audio()
    Label:
        id: result_label
        text: 'Estimated Heart Rate: '
        font_size: 20
""")

class RootWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_audio(self):
        duration = 5  # Record for 5 seconds
        self.ids.result_label.text = "Recording..."
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        audio_data = audio_data.flatten()

        # Ensure the audio data is at 44100 Hz
        audio_data_resampled = librosa.resample(audio_data, orig_sr=fs, target_sr=fs)

        estimated_heart_rate = self.predict_heart_rate(audio_data_resampled, fs)
        self.ids.result_label.text = f'Estimated Heart Rate: {estimated_heart_rate:.2f} bpm'

    def predict_heart_rate(self, audio_data, sample_rate):
        samples_per_segment = int(len(audio_data) / num_segments)
        num_mfcc_vectors_per_segment = int(np.ceil(samples_per_segment / hop_length))

        all_mfcc_features = []
        
        for segment in range(num_segments):
            start = samples_per_segment * segment
            finish = start + samples_per_segment

            mfcc_features = self.extract_mfcc_segment(audio_data, sample_rate, start, finish)
            mfcc_features = self.pad_or_truncate(mfcc_features, max_length)
            
            all_mfcc_features.append(mfcc_features)
        
        all_mfcc_features = np.array(all_mfcc_features)

        predictions = model.predict(all_mfcc_features)
        avg_prediction = np.mean(predictions)
        
        return avg_prediction

    def extract_mfcc_segment(self, signal, sample_rate, start, finish):
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return mfcc.T

    def pad_or_truncate(self, mfcc, max_length):
        if len(mfcc) < max_length:
            pad_width = max_length - len(mfcc)
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:max_length]
        return mfcc

class HeartRateApp(App):
    def build(self):
        return RootWidget()

if __name__ == '__main__':
    HeartRateApp().run()