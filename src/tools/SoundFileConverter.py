from pathlib import Path
import miniaudio
import soundfile as sf


SOURCE_DIR = Path("../../data/raw")
OUTPUT_DIR = Path("../../data/wav")

SUPPORTED_EXTENSIONS = {".mp3", ".opus", ".ogg", ".flac", ".wav"}


def decode_audio(file_path: Path):
    """Decodes audio into raw samples without external system binaries."""
    extension = file_path.suffix.lower()
    if extension == ".mp3":
        decoded = miniaudio.decode_file(str(file_path))
        return decoded.samples, decoded.sample_rate, decoded.nchannels
    else:
        # soundfile natively supports .opus, .ogg, .flac, and .wav
        data, sample_rate = sf.read(str(file_path))
        channels = 1 if data.ndim == 1 else data.shape[1]
        return data, sample_rate, channels


def convert_file(input_file: Path, output_file: Path):
    """Processes a single audio file and exports it as standard 16-bit PCM WAV."""
    data, sample_rate, channels = decode_audio(input_file)

    if isinstance(data, (bytes, memoryview)):
        wav_bytes = miniaudio.wav_write(
            data,
            nchannels=channels,
            sample_rate=sample_rate,
            sample_width=2  # 16-bit PCM standard
        )
        with open(output_file, "wb") as f:
            f.write(wav_bytes)
    else:
        sf.write(str(output_file), data, sample_rate, subtype="PCM_16")


def main():
    if not SOURCE_DIR.exists() or not OUTPUT_DIR.exists():
        print(f"Error: Source directory or output directory not found -> {SOURCE_DIR} and {OUTPUT_DIR}")
        return


    files = [
        f for f in SOURCE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print(f"No supported audio files found in: {SOURCE_DIR}")
        return

    success_count = 0
    for input_file in files:
        output_file = OUTPUT_DIR / f"{input_file.stem}.wav"
        print(f"-> Converting: {input_file.name}...")
        try:
            convert_file(input_file, output_file)
            print(f"Saved: {output_file.name}")
            success_count += 1
        except Exception as error:
            print(f"Failed to convert {input_file.name}: {error}")

    print(f"\nProcessing complete: {success_count}/{len(files)} files converted successfully.")


if __name__ == "__main__":
    main()