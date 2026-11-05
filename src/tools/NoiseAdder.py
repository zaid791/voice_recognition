import numpy as np
from scipy import signal


def add_white_noise(audio, noise_level=0.02):
    """Adds random white (Gaussian) noise to an audio signal.
    """
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise


def add_simple_reverb(audio, delay_samples=1600, decay=0.3):
    """Simulates a simple room echo by convolving the signal with an impulse response.
    """
    # Build the impulse response: a spike at t=0 (direct sound) and
    # a smaller spike at t=delay_samples (the echo)
    impulse = np.zeros(delay_samples + 1)
    impulse[0] = 1.0
    impulse[-1] = decay

    # Convolve the audio with the impulse response to apply the echo effect
    return signal.convolve(audio, impulse, mode='same')
