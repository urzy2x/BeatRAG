import os
import json
import librosa
import numpy as np
import multiprocessing
from rich.console import Console
from rich.panel import Panel

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples") 
OUTPUT_JSON = os.path.join(BASE_DIR, "sample_database.json")

def get_key(y, sr):
    # 1. Extract the chromagram using Chroma CENS
    try:
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    except Exception:
        # Fallback to STFT for very short samples where CQT might fail
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
    # Sum the energy of each note over the entire loop
    chroma_sum = np.sum(chroma, axis=1) 
    
    # 2. Krumhansl-Schmuckler key profiles
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    maj_corrs = []
    min_corrs = []
    
    # 3. Test the audio's notes against all 12 keys
    for i in range(12):
        shifted_maj = np.roll(maj_profile, i)
        shifted_min = np.roll(min_profile, i)
        
        maj_corrs.append(np.corrcoef(chroma_sum, shifted_maj)[0, 1])
        min_corrs.append(np.corrcoef(chroma_sum, shifted_min)[0, 1])
        
    # 4. Find the best match
    best_maj_idx = np.argmax(maj_corrs)
    best_min_idx = np.argmax(min_corrs)
    
    if maj_corrs[best_maj_idx] > min_corrs[best_min_idx]:
        return f"{keys[best_maj_idx]} Major"
    else:
        return f"{keys[best_min_idx]} Minor"
    
def extract_features(file_path):
    console = Console()
    console.print(f"Analyzing: {os.path.basename(file_path)}...")
    
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        duration_sec = len(y) / sr
        if duration_sec > 30:
            mid_point = int((duration_sec / 2) * sr)
            half_window = int(15 * sr)
            y_analysis = y[mid_point - half_window : mid_point + half_window]
        else:
            y_analysis = y
            
        y_harmonic, y_percussive = librosa.effects.hpss(y_analysis)

        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=140.0)
        bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        if 0 < bpm < 100:
            bpm *= 2
        
        centroid = librosa.feature.spectral_centroid(y=y_analysis, sr=sr)
        mean_centroid = float(np.mean(centroid))
        
        detected_key = get_key(y_harmonic, sr)
        
        flatness = librosa.feature.spectral_flatness(y=y_analysis)
        mean_flatness = float(np.mean(flatness))

        rms = librosa.feature.rms(y=y_analysis)
        mean_rms = float(np.mean(rms))
        
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        duration_analysis = len(y_analysis) / sr
        onset_rate = len(onsets) / duration_analysis

        is_minor = "Minor" in detected_key
        is_high_energy = mean_rms > 0.35
        is_fast = bpm >= 130
        
        if is_minor:
            if is_high_energy and is_fast:
                mood = "Aggressive, Intense, Dark"
            elif is_high_energy:
                mood = "Epic, Serious, Moody"
            elif is_fast:
                mood = "Restless, Tense, Driving"
            else:
                mood = "Melancholy, Ambient, Somber"
        else:
            if is_high_energy and is_fast:
                mood = "Energetic, Euphoric, Triumphant"
            elif is_high_energy:
                mood = "Uplifting, Bright, Confident"
            elif is_fast:
                mood = "Bouncy, Playful, Upbeat"
            else:
                mood = "Chill, Relaxed, Peaceful"

        return {
            "filename": os.path.basename(file_path),
            "filepath": file_path,
            "bpm": round(bpm),
            "key": detected_key,
            "mood": mood,
            "spectral_centroid": round(mean_centroid, 2),
            "spectral_variance": round(float(np.var(centroid)), 2),
            "spectral_flatness": round(mean_flatness, 5),
            "overall_energy": round(mean_rms, 3),
            "bounciness": round(onset_rate, 2)
        }
        
    except Exception as e:
        console.print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

def main():
    console = Console()
    console.print(Panel("[bold green]Extracting features from your beats[/bold green]"))

    if not os.path.exists(SAMPLES_DIR):
        console.print(f"[bold red]⚠️ Folder not found: {SAMPLES_DIR}[/bold red]")
        console.print("Create a 'samples' folder in your BeatRAG directory and drop some beats in there!")
        return

    file_paths = []
    for filename in os.listdir(SAMPLES_DIR):
        if filename.lower().endswith(('.wav', '.mp3')):
            file_paths.append(os.path.join(SAMPLES_DIR, filename))
            
    safe_cores = min(max(1, multiprocessing.cpu_count() // 2), 4)
    
    console.print(f"Starting parallel extraction on {len(file_paths)} files using {safe_cores} CPU cores (Safe Mode)...")
    with multiprocessing.Pool(processes=safe_cores) as pool:
        results = pool.map(extract_features, file_paths)
        
    database = [res for res in results if res is not None]

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(database, f, indent=4)
        
    console.print(f"\n[bold green]Done! Extracted features for {len(database)} samples.[/bold green]")
    console.print(f"Saved database to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()