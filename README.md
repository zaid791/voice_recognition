# Voice Recognition for an Intercom

Introduction to Machine Learning project, academic year 2026/2027.

## The idea

Access to our floor is restricted by a door with a voice-based intercom.
A person speaks into the microphone and the system decides whether to open the door.

This is a **binary classification** task based on the **speaker's voice**:

- the audio is converted to **spectrograms**;
- a **convolutional neural network** (trained from scratch) decides who is allowed in.

## Classes

| Class | Meaning | Who |
|---|---|---|
| **1** | allowed | **only the five members of our team** |
| **0** | not allowed | **everyone else** |

The class depends only on **who is speaking**, not on **what** is said:

- one of us says anything → class **1**
- anyone else says anything → class **0**

## Metrics

Measured separately on the train, validation and test sets:

- **FAR** (False Acceptance Ratio): class-0 samples accepted as class 1, divided by the number of class-0 samples
- **FRR** (False Rejection Ratio): class-1 samples rejected as class 0, divided by the number of class-1 samples
- **macro F1**

FAR and FRR are equally important and should stay close to each other.
