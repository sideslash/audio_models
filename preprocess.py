import numpy as np
import librosa 
import os
import matplotlib.pyplot as plt

def mu_law_quantize(samples, mu=255):
    companded_x = np.sign(samples) * np.log(1 + mu * np.abs(samples)) / np.log(1 + mu)
    quantized_x = ((companded_x + 1.0) / 2.0 * mu).astype(np.int32)
    # print(quantized_x.shape)
    # print(samples[100000:100100])
    # print(quantized_x[100000:100100])
    return companded_x

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
