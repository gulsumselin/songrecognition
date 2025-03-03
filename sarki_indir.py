import json
import requests
import os

# JSON dosyasını yükleme
with open('trackss.json', 'r', encoding='utf-8') as file:
    songs = json.load(file)

# Şarkıların kaydedileceği klasör
os.makedirs('all_audio_files/cleaned', exist_ok=True)

# Çalışmayan URL'leri kaydetmek için bir dosya
unplayable_urls_file = 'unplayable_urls.txt'
unplayable_urls = []

# Şarkıları indirme
for song in songs:
    # Şarkı bilgilerini al
    artist = song['artist']
    track_name = song['track_name']
    preview_url = song['preview_url']

    # Geçerli URL kontrolü
    if not preview_url:
        print(f"Geçersiz URL: {artist} - {track_name}")
        unplayable_urls.append(preview_url)
        continue

    # Şarkının kaydedileceği dosya yolu
    # Dosya adlarında geçersiz karakterleri temizle

    # Artist ismindeki boşlukları kaldırıyoruz ve birleşik dosya adı oluşturuyoruz.
    audio_file_path = f"{artist.replace(' ', '')}_{track_name}.wav".replace("/", "_").replace("\\", "_")

    # Şarkıyı indir
    if not os.path.exists(audio_file_path):
        try:
            response = requests.get(preview_url, timeout=10)  # Zaman aşımı 10 saniye
            response.raise_for_status()  # HTTP hatalarını yakala
            # Dosyayı kaydet
            with open(audio_file_path, 'wb') as audio_file:
                audio_file.write(response.content)
            print(f"İndirildi: {artist} - {track_name}")
        except requests.exceptions.RequestException as e:
            print(f"İndirilemedi: {artist} - {track_name}, Hata: {e}")
            unplayable_urls.append(preview_url)  # Çalışmayan URL'yi listeye ekle

# Çalışmayan URL'leri kaydet
if unplayable_urls:
    with open(unplayable_urls_file, 'w') as file:
        for url in unplayable_urls:
            file.write(url + '\n')
    print(f"Çalışmayan URL'ler {unplayable_urls_file} dosyasına kaydedildi.")

print("İndirme işlemi tamamlandı!")
