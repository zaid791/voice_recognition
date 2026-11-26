import csv
from pathlib import Path

WAV_DIR = Path("../../data/wav")
OUTPUT_FILE = Path("../../data/metadata.csv")


def main():
    if not WAV_DIR.exists():
        print(f"Error: WAV directory not found -> {WAV_DIR}")
        return

    wav_files = sorted(
        [f for f in WAV_DIR.iterdir() if f.is_file() and f.suffix == ".wav"],
        key=lambda f: (f.stem.split("_")[0], f.stem.split("_")[1], int(f.stem.split("_")[2]) if len(f.stem.split("_")) > 2 else 0),
    )

    if not wav_files:
        print(f"No .wav files found in: {WAV_DIR}")
        return

    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        writer.writerow(["file", "class", "speaker_id", "audio_number"])

        for wav_file in wav_files:
            # filename format: {class}_{speaker_id}_{audio_number}.wav  ->  e.g. 0_Anna_2.wav
            parts = wav_file.stem.split("_", maxsplit=2)
            file_class = parts[0]
            speaker_id = parts[1]
            audio_number = parts[2] if len(parts) > 2 else "1"

            # path relative to the repo root
            relative_path = Path("data/wav") / wav_file.name

            writer.writerow([relative_path, file_class, speaker_id, audio_number])

    print(f"metadata.csv created with {len(wav_files)} entries -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

