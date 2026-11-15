# Voice Recognition for an Intercom

Introduction to Machine Learning project, academic year 2026/2027.

## The idea

Access to our floor is restricted by a door with a voice-based intercom.
A person speaks into the microphone and the system decides whether to open the door.

This is a **binary classification** task based on the **speaker's voice**:

- the audio is converted to **spectrograms**;
- a **convolutional neural network** (trained from scratch) decides who is allowed in.

## Classes

We classify the speaker's **gender**:

| Class | Meaning | Who |
|---|---|---|
| **1** | allowed | **men** |
| **0** | not allowed | **women** |

The class depends only on **who is speaking**, not on **what** is said.

File names follow `{class}_{speaker}_{n}`, e.g. `1_Victor_2.wav` or `0_Anna_1.wav`
(see [data_format_spec.md](data_format_spec.md)).

## Metrics

Measured separately on the train, validation and test sets:

- **FAR** (False Acceptance Ratio): class-0 samples accepted as class 1, divided by the number of class-0 samples
- **FRR** (False Rejection Ratio): class-1 samples rejected as class 0, divided by the number of class-1 samples
- **macro F1**

FAR and FRR are equally important and should stay close to each other.
