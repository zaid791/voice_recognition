# Voice Recognition for an Intercom

Introduction to Machine Learning project, academic year 2026/2027.

## The idea

Access to our floor is restricted by a door with a voice-based intercom.
A person speaks into the microphone and the system decides whether to open the door.

This is a **binary classification** task based on the **speaker's voice**:

- the audio is converted to **spectrograms**;
- a **convolutional neural network** (trained from scratch) decides who is allowed in.

## Classes

We classify whether the speaker is one of the **accepted team members**:

| Class | Meaning | Who |
|---|---|---|
| **1** | allowed | Victor, Ulyana, Inigo, Mohammed, Damian |
| **0** | not allowed | everyone else |

The class depends only on **who is speaking**, not on **what** is said.

File names follow `{class}_{speaker}_{n}`, e.g. `1_Victor_2.wav` or `0_Anna_1.wav`
(see [data_format_spec.md](data_format_spec.md)).

## Repository structure

```
data/
  raw/            original recordings (any format), never modified
  wav/            converted to 16-bit PCM WAV
  metadata.csv    file | class | speaker_id
  splits/         train.txt / val.txt / test.txt (recording ids)
  clips/          3 s clips              (generated, not in git)
  spectrograms/   log-mel .npy + index   (generated, not in git)
src/
  tools/          SoundFileConverter.py (raw → wav), CSVMaker.py (metadata), NoiseAdder.py
  preprocessing/  preprocess.py: wav → clips → spectrograms
  model/          dataset, split, CNN, metrics, training
notebooks/
  01_eda.ipynb        exploratory data analysis
  02_training.ipynb   training, learning curves, metrics
models/           trained networks (generated, not in git)
```

## How to run

Run all commands from the repository root.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.preprocessing.preprocess all   # wav → clips → spectrograms
python -m src.model.train                    # split → training → metrics
```

After adding new recordings to `data/raw/`, first convert them and rebuild the metadata:

```bash
cd src/tools && python SoundFileConverter.py && python CSVMaker.py && cd ../..
```

The notebooks in `notebooks/` are saved with all outputs and can be read without running them.

## Pipeline

| Step | What happens |
|---|---|
| Conversion | any format → 16-bit PCM WAV |
| Clips | mono, resample to **16 kHz**, remove silence (> 30 dB below the peak), peak normalization, cut into **3 s** clips (remainders < 1.5 s are dropped, longer ones padded) |
| Spectrograms | log-mel, **64 mel bands**, window 1024 (64 ms), hop 256 (16 ms) → array **64 × 188** |
| Split | by **recording** (all clips of one recording in the same part), stratified by class, 70 / 15 / 15, seed 42 |
| Model | simple CNN from the PyTorch CIFAR10 tutorial with 1 input channel |
| Training | Adam (lr 0.001), batch 16, 30 epochs, class-weighted cross-entropy; the epoch with the best validation macro F1 is saved |

## Metrics

Measured separately on the train, validation and test sets:

- **FAR** (False Acceptance Ratio): class-0 samples accepted as class 1, divided by the number of class-0 samples
- **FRR** (False Rejection Ratio): class-1 samples rejected as class 0, divided by the number of class-1 samples
- **macro F1**

FAR and FRR are equally important and should stay close to each other.

## Results (Milestone 1)

77 recordings (51 men, 26 women) → 96 clips, seed 42:

| Set | macro F1 | FAR | FRR | clips |
|---|---|---|---|---|
| train | 1.000 | 0.000 | 0.000 | 67 |
| validation | 0.625 | 0.500 | 0.250 | 12 |
| test | 0.821 | 0.143 | 0.200 | 17 |

Goal of Milestone 1 (macro F1 ≥ 0.6 on the training set) is reached.
The validation and test sets are small, so their numbers are noisy.

## Known limitations

- Small dataset; 15 recordings are too short to produce a clip after silence removal.
- For 3 recordings (`1_unk32`–`1_unk34`) the speaker's gender is uncertain.
- The split is by recording, not yet by speaker.
- The network overfits (train F1 = 1.0); regularization and augmentation are planned for Milestone 2.
