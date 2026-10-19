# Data Format Specification

All team members must follow this specification so that the data from different sources is consistent and the pipeline is reproducible.

## 1. File format: WAV (16-bit PCM)

- All audio used for training must be stored as **uncompressed WAV, 16-bit PCM**.
- Compressed formats (MP3, M4A) remove parts of the signal and leave artifacts on spectrograms.
- **Why it matters:** if one class is recorded in M4A and the other in WAV, the model may learn to detect the compression instead of the actual class.
- Recording in any format (e.g. M4A on a phone) is fine. All files are converted to WAV by a script.

## 2. Channels: mono

- All audio is converted to **mono (1 channel)**.
- Stereo adds no useful information for this task.
- A mono spectrogram is a single-channel image, which keeps the CNN simpler and faster.
- The laptop microphone used in the final demo is effectively mono as well.

## 3. Sample rate: 16 kHz

- All audio is resampled to **16 kHz**.
- A 16 kHz sample rate preserves frequencies up to **8 kHz** (Nyquist), which covers almost all relevant speech information.
- 16 kHz is the standard for speech datasets and models (Common Voice, LibriSpeech, Speech Commands).
- Smaller files and spectrograms mean faster training.
- **Why it matters:** if some files are 48 kHz and others 16 kHz, their spectrograms differ in the high-frequency range, and the model can use this difference as a shortcut instead of learning the actual class.

## 4. Clip length: 3 seconds (default)

- Recordings are cut into **3-second clips**, which are the samples the CNN sees.
- 1 s is too short (one or two words). Longer clips give fewer samples from the same amount of audio and make the demo less convenient.
- **Raw recordings are always kept.** Cutting is done in code, so the clip length can be changed with one parameter.
- Clip length is one of the graded techniques ("choosing the length of a single excerpt"). We plan to compare 1 / 2 / 3 / 5 s in Milestone #2.

## 5. File naming

Pattern:

```
{class}_{speaker}_{recording}_{device}_{env}.wav
```

Examples:

```
1_anna_r03_laptop_quiet.wav
0_friend07_r01_phone_noisy.wav
```

After cutting into clips:

```
1_anna_r03_laptop_quiet_c005.wav    (c005 = clip number 5)
```

| Field | Description | Example values |
|---|---|---|
| `class` | class label | `0`, `1` |
| `speaker` | speaker ID | `anna`, `friend07` |
| `recording` | recording ID for this speaker | `r01`, `r02` |
| `device` | recording device | `laptop`, `phone` |
| `env` | environment | `quiet`, `noisy`, `echo` |

Rules:

- Use only Latin letters, digits and `_`
- No spaces and no Polish characters (ą, ł, ś, etc.)

**Why it matters:** the file name shows where each sample comes from. It also makes it easy to split data by speaker or recording, so that clips from the same recording never end up in both train and test (data leakage).

## 6. Metadata

All files must be listed in `data/metadata.csv`:

| file | class | speaker_id | recording_id | device | environment |
|---|---|---|---|---|---|

`metadata.csv` is the main source of truth. The file name is a backup and a convenience.

## 7. Folder structure

```
data/
  raw/           # original recordings, any format, never modified
  wav/           # converted: WAV, mono, 16 kHz
  clips/         # 3-second clips
  spectrograms/  # .npy files used for training
  metadata.csv
```

Each step can be recomputed from the previous one without overwriting the original data, which keeps the pipeline reproducible.
