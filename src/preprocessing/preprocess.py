"""Preprocessing: data/wav -> clips -> spectrograms (siehe data_format_spec.md).

Voraussetzung: src/tools/SoundFileConverter.py hat data/raw nach data/wav konvertiert.

Stufen (Originale in raw/ und wav/ werden nie veraendert):
    data/wav/          WAV aus SoundFileConverter (Samplerate/Kanaele koennen variieren)
 -> data/clips/        mono, 16 kHz, Stille entfernt, Lautstaerke normalisiert,
                       in 3-Sekunden-Clips geschnitten:  {class}_{speaker}_c005.wav
 -> data/spectrograms/ Log-Mel-Spektrogramme als .npy:   {class}_{speaker}_c005.npy
                       + index.csv (file|class|speaker_id|clip_idx)

Aufruf immer aus dem Repo-Hauptordner:
    python -m src.preprocessing.preprocess all
    python -m src.preprocessing.preprocess clips --clip-sec 2
    python -m src.preprocessing.preprocess spectrograms
"""
import argparse
import csv
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16000
CLIP_SEC = 3.0
N_FFT = 1024
HOP = 256
N_MELS = 64
TOP_DB = 30  # Stille: mehr als 30 dB unter dem Peak
CLIP_RE = re.compile(r"^(?P<cls>[01])_(?P<speaker>.+)_c(?P<idx>\d+)$")


def remove_silence(y, top_db=TOP_DB):
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y
    return np.concatenate([y[s:e] for s, e in intervals])


def normalize(y, peak=0.95):
    m = np.max(np.abs(y))
    return y if m < 1e-8 else y / m * peak


# ---------- Stufe 1: wav -> clips ----------
def make_clips(wav_dir, clips_dir, clip_sec=CLIP_SEC, silence=True, min_frac=0.5):
    clips_dir = Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    n_len = int(SR * clip_sec)
    n_files = n_clips = 0
    for p in sorted(Path(wav_dir).glob("*.wav")):
        try:
            y, _ = librosa.load(p, sr=SR, mono=True)  # mono + Resample auf 16 kHz
        except Exception as e:
            print(f"[WARN] {p.name}: {e}")
            continue
        if silence:
            y = remove_silence(y)
        y = normalize(y)
        idx = 0
        for start in range(0, len(y), n_len):
            c = y[start:start + n_len]
            if len(c) < n_len * min_frac:  # zu kurzer Rest -> verwerfen
                continue
            if len(c) < n_len:
                c = np.pad(c, (0, n_len - len(c)))
            sf.write(clips_dir / f"{p.stem}_c{idx:03d}.wav", c, SR, subtype="PCM_16")
            idx += 1
        n_files += 1
        n_clips += idx
    print(f"clips: {n_clips} Clips ({clip_sec}s) aus {n_files} Dateien -> {clips_dir}")


# ---------- Stufe 2: clips -> spectrograms ----------
def mel_spectrogram(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)  # (n_mels, frames)


def make_spectrograms(clips_dir, spec_dir):
    spec_dir = Path(spec_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(Path(clips_dir).glob("*.wav")):
        m = CLIP_RE.match(p.stem)
        if not m:
            print(f"[WARN] Name passt nicht zu {{class}}_{{speaker}}_c000: {p.name}")
            continue
        y, _ = sf.read(p, dtype="float32")
        np.save(spec_dir / f"{p.stem}.npy", mel_spectrogram(y))
        rows.append([f"{p.stem}.npy", m["cls"], m["speaker"], int(m["idx"])])
    with open(spec_dir / "index.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="|")  # gleiches Format wie data/metadata.csv
        w.writerow(["file", "class", "speaker_id", "clip_idx"])
        w.writerows(rows)
    print(f"spectrograms: {len(rows)} .npy -> {spec_dir} (+ index.csv)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["clips", "spectrograms", "all"])
    ap.add_argument("--data", default="data", help="Basisordner (Default: data)")
    ap.add_argument("--clip-sec", type=float, default=CLIP_SEC)
    ap.add_argument("--keep-silence", action="store_true", help="Stille nicht entfernen")
    a = ap.parse_args()
    d = Path(a.data)
    if a.stage in ("clips", "all"):
        make_clips(d / "wav", d / "clips", a.clip_sec, silence=not a.keep_silence)
    if a.stage in ("spectrograms", "all"):
        make_spectrograms(d / "clips", d / "spectrograms")


if __name__ == "__main__":
    main()
