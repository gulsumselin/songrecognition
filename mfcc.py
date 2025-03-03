import os
import librosa
import numpy as np
import pandas as pd
import time

# İşlem süresi başlat
start_time = time.time()

def extract_features(file_path):
    try:
        # Ses dosyasını yükle
        y, sr = librosa.load(file_path, sr=None)
        
        # Özellikler
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y).mean()
        
        # Özellikleri birleştir
        features = np.hstack([mfcc, chroma, spectral_contrast, spectral_bandwidth, zero_crossing_rate])
        return features
    except Exception as e:
        # Hataları log dosyasına yaz
        with open("error_log.txt", "a") as log_file:
            log_file.write(f"Hata: {file_path} - {e}\n")
        return None

def process_audio_dataset(root_dir):
    all_features = []
    for artist in os.listdir(root_dir):
        artist_path = os.path.join(root_dir, artist)
        if os.path.isdir(artist_path):
            for file_name in os.listdir(artist_path):
                file_path = os.path.join(artist_path, file_name)
                features = extract_features(file_path)
                if features is not None:
                    feature_row = {
                        "file_name": file_name,
                        "folder": artist,
                    }
                    for i, value in enumerate(features[:13]):  # MFCC özellikleri
                        feature_row[f"mfcc_{i+1}"] = value
                    for i, value in enumerate(features[13:25]):  # Chroma özellikleri
                        feature_row[f"chroma_{i+1}"] = value
                    for i, value in enumerate(features[25:31]):  # Spectral Contrast özellikleri
                        feature_row[f"spectral_contrast_{i+1}"] = value
                    feature_row["spectral_bandwidth"] = features[31]
                    feature_row["zero_crossing_rate"] = features[32]
                    all_features.append(feature_row)
                    
                    # Her 100 dosyada bir CSV'ye kaydet
                    if len(all_features) % 100 == 0:
                        pd.DataFrame(all_features).to_csv("audio_partial.csv", index=False, mode="a", header=not os.path.exists("audio_partial.csv"))
                        all_features = []  # Belleği temizle
    return pd.DataFrame(all_features)

# Verilerin bulunduğu klasör yolu
root_directory = "audio/"

# Özellikleri çıkar ve DataFrame oluştur
df = process_audio_dataset(root_directory)

# Kalan verileri kaydet
if not df.empty:
    df.to_csv("audio.csv", index=False)

# İşlem süresi yazdır
print(f"Şarkı özellikleri çıkarıldı ve CSV dosyasına kaydedildi.")
print(f"İşlem süresi: {time.time() - start_time:.2f} saniye")
