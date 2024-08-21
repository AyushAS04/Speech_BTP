# Heart Rate Detection Using Speech

This project is an advanced system designed to estimate heart rate from speech recordings using a neural network model. By analyzing the Mel-frequency cepstral coefficients (MFCCs) of the audio, the model accurately predicts heart rate, providing a non-invasive and innovative approach to biometric monitoring.

## Introduction

This project aims to explore the intersection of speech processing and biometric analysis by estimating heart rate directly from audio recordings. The system leverages machine learning techniques, specifically a neural network model, to analyze speech and predict the heart rate (in beats per minute). This approach has potential applications in health monitoring, stress analysis, and fitness tracking.

### Key Features

- **Speech-Based Heart Rate Estimation**: Predicts heart rate using only a short speech recording, making it a convenient and non-invasive method.
- **Advanced Signal Processing**: Utilizes Mel-frequency cepstral coefficients (MFCCs), a robust feature set commonly used in speech recognition, to extract meaningful patterns from audio data.
- **Neural Network Model**: Employs a well-trained neural network model to map MFCC features to heart rate values with high accuracy.

## Prerequisites

Before running the project, make sure your environment is set up with the following software:

- **Python 3.x**: The core programming language used for the project.
- **Required Python Libraries**:
  - `numpy`: For numerical operations and data manipulation.
  - `librosa`: For audio processing, particularly MFCC feature extraction.
  - `soundfile`: For reading and writing audio files.
  - `sounddevice`: For recording audio directly from the microphone.
  - `tensorflow`: For building and running the neural network model.

## Project Overview
### Problem Statement
The project addresses the challenge of estimating heart rate using speech recordings. Traditional methods of heart rate measurement, such as ECG or pulse oximetry, require physical contact or specialized equipment. This project proposes an alternative by analyzing speech, which can be recorded with any standard microphone.

### Methodology
__Data Collection:__ Speech data, potentially containing corresponding heart rate measurements, is used to train the model.

__Feature Extraction:__

__MFCC Extraction:__ The speech signal is processed to extract MFCC features, capturing essential characteristics of the audio that correlate with physiological states.

__Segmentation:__ The audio is divided into segments to enhance the model’s ability to capture temporal variations in speech.

__Model Training:__ A neural network is trained on the extracted MFCC features and corresponding heart rate labels. The training process involves optimizing the model to minimize prediction errors.

__Prediction:__ For a given speech input, the model processes the audio, extracts the MFCC features, and predicts the heart rate. The system outputs the estimated heart rate in beats per minute (bpm).

## Usage
The system is designed for ease of use:

__Recording Audio:__ The system records a 5-second audio clip from the user's microphone. This recording is used as the input for heart rate prediction.

__Feature Extraction:__ The recorded audio is divided into segments, and MFCC features are extracted from each segment.

__Prediction:__ The extracted features are fed into the neural network model, which outputs an estimate of the heart rate.

__Output:__ The estimated heart rate is displayed in beats per minute (bpm), providing an immediate and actionable result.

### Example Workflow
__Start the Script:__ The script will automatically begin recording audio from your microphone.
__Process the Audio:__ The system processes the recording to extract MFCC features.
__Predict Heart Rate:__ The neural network model predicts the heart rate based on the processed audio.
__Display Result:__ The estimated heart rate is printed, e.g., "Estimated Heart Rate (bpm): 72.45".
## Conclusion
This project demonstrates the feasibility of using speech as a biometric indicator for heart rate estimation. It combines advanced audio processing techniques with machine learning to deliver a novel solution that could be applied in various fields, from healthcare to fitness.

By completing this project, significant skills were developed in areas such as signal processing, machine learning, and software development. The project showcases an innovative approach to biometric analysis and reflects the practical application of theoretical knowledge.
