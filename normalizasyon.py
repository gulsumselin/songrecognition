from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib

# CSV dosyasını yükle
file_path = "audio.csv"
df = pd.read_csv(file_path)

# Hedef sütunlar dışındaki özellikleri seçme
features = [col for col in df.columns if col not in ["file_name", "folder"]]

# MinMaxScaler ile özellikleri 0-1 aralığına ölçeklendirme
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Normalleştirilmiş veriyi kaydetme
normalized_file_path = "normalized_audio.csv"
df.to_csv(normalized_file_path, index=False)

# Scaler nesnesini kaydetme
joblib.dump(scaler, "minmax_scaler.pkl")

print(f"Normalleştirilmiş veri başarıyla kaydedildi: {normalized_file_path}")
print("Scaler kaydedildi: minmax_scaler.pkl")
