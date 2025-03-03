from pydub import AudioSegment

# WAV dosyasını yükleme
file_path = "3.wav"  # Orijinal dosya
output_path = "33.wav"  # Kesilmiş dosyanın kaydedileceği yol

# Ses dosyasını yükle
audio = AudioSegment.from_wav(file_path)

# Kesilecek süreler (milisaniye cinsinden)
start_time = 80 * 1000  # Başlangıç: 0 saniye
end_time = 115 * 1000   # Bitiş: 60 saniye (örnek: 1 dakika)

# Ses dosyasını kesme
trimmed_audio = audio[start_time:end_time]

# Kesilen dosyayı kaydetme
trimmed_audio.export(output_path, format="wav")

print(f"WAV dosyası başarıyla kesildi ve kaydedildi: {output_path}")
