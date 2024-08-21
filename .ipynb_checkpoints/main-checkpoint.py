import os
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.core.audio import SoundLoader
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import StringProperty
from tensorflow.keras.models import load_model

# Parameters
fs = 44100  # Target sampling rate
n_mfcc = 13  # Number of MFCC features
n_fft = 2048  # Interval to apply FFT
hop_length = 512  # Sliding window for FFT
num_segments = 2  # Number of segments to divide each audio file
max_length = 300  # Maximum length of MFCC features (number of frames)
model_save_path = 'trained_model.h5'  # Path to the trained model

# Load the trained model
model = load_model(model_save_path)

class LoadDialog(FloatLayout):
    load = StringProperty(None)
    cancel = StringProperty(None)

class RootWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        # Add buttons and labels
        self.label = Label(text='Heart Rate Estimation', font_size=30)
        self.add_widget(self.label)

        self.record_button = Button(text='Record Audio', size_hint=(1, 0.1))
        self.record_button.bind(on_press=self.record_audio)
        self.add_widget(self.record_button)

        self.load_button = Button(text='Load Pre-recorded Audio', size_hint=(1, 0.1))
        self.load_button.bind(on_press=self.show_load_dialog)
        self.add_widget(self.load_button)

        self.result_label = Label(text='Estimated Heart Rate: ', font_size=20)
        self.add_widget(self.result_label)

    def record_audio(self, instance):
        duration = 5  # Record for 5 seconds
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        audio_data = audio_data.flatten()
        estimated_heart_rate = self.predict_heart_rate(audio_data, fs)
        self.result_label.text = f'Estimated Heart Rate: {estimated_heart_rate:.2f} bpm'

    def show_load_dialog(self, instance):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load Pre-recorded Audio", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        self.dismiss_popup()
        file_path = os.path.join(path, filename[0])
        audio_data, sample_rate = librosa.load(file_path, sr=fs)
        estimated_heart_rate = self.predict_heart_rate(audio_data, sample_rate)
        self.result_label.text = f'Estimated Heart Rate: {estimated_heart_rate:.2f} bpm'

    def dismiss_popup(self):
        self._popup.dismiss()

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
