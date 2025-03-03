import requests
import json

def get_artist_top_tracks(artist_name, limit=25):
    # Deezer Artist Search Endpoint
    search_url = f"https://api.deezer.com/search/artist?q={artist_name}"
    response = requests.get(search_url)
    
    if response.status_code != 200:
        print(f"Error: Unable to fetch artist data for {artist_name}")
        return []
    
    artist_data = response.json().get("data")
    if not artist_data:
        print(f"No data found for artist: {artist_name}")
        return []
    
    artist_id = artist_data[0].get("id")  # Get the first matching artist ID
    tracks_url = f"https://api.deezer.com/artist/{artist_id}/top?limit={limit}"
    response = requests.get(tracks_url)
    
    if response.status_code != 200:
        print(f"Error: Unable to fetch top tracks for {artist_name}")
        return []
    
    tracks_data = response.json().get("data", [])
    track_list = []
    for track in tracks_data:
        track_info = {
            "artist": artist_name,
            "track_name": track.get("title"),
            "track_id": track.get("id"),
            "album": track.get("album", {}).get("title"),
            "preview_url": track.get("preview")
        }
        track_list.append(track_info)
    
    return track_list

def fetch_tracks_for_artists(artists):
    all_tracks = []
    for artist in artists:
        print(f"Fetching tracks for artist: {artist}")
        artist_tracks = get_artist_top_tracks(artist)
        all_tracks.extend(artist_tracks)
    return all_tracks

# List of 30 artists
artists = [
    "Yüksek Sadakat","TNK","Gece Yolcuları","Batuhan Kordel",
    "Raviş","Kalben","Yaşlı Amca"
]



# Fetch tracks for all artists
all_tracks = fetch_tracks_for_artists(artists)

# Save to JSON file
with open("trackss.json", "w", encoding="utf-8") as file:
    json.dump(all_tracks, file, ensure_ascii=False, indent=4)

print("Tracks saved to tracks.json")
