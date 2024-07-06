import numpy as np
import librosa 
import os
import matplotlib.pyplot as plt
import sounddevice as sd
import pandas as pd

def mu_law_quantize(samples, mu=255):
    companded_x = np.sign(samples) * np.log(1 + mu * np.abs(samples)) / np.log(1 + mu)
    quantized_x = ((companded_x + 1.0) / 2.0 * mu).astype(np.int32)
    return quantized_x

def mu_law_recover(mu_law_samples, mu=255):
    return np.sign(mu_law_samples) * (np.exp(np.abs(mu_law_samples) * np.log(1 + mu)) - 1) / mu

if __name__ == "__main__":
    data_path = "data"
    audio_files = os.listdir(data_path)
    print(f"audio_files:{audio_files}")
    data_processed_path = "data_processed"

    for audio_file in audio_files:
        samples, sr = librosa.load(f"{data_path}/{audio_file}", sr=16000)
        mu_samples = mu_law_quantize(samples)
        np.save(f"{data_processed_path}/{audio_file}.npy", mu_samples)


    # sample_rate = 16000
    # samples = np.load(f"{data_processed_path}/{audio_files[0]}.npy")
    # audio_data = samples[0:sample_rate * 20].astype(np.float32)  
    # audio_data = (audio_data / (255/2.0)) - 1
    # sd.play(audio_data, samplerate=sample_rate)
    # sd.wait()
