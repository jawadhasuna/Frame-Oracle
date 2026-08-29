"""Controlled test signals for evaluating detectors.

Deliberately simple. A detector should be characterised against inputs whose
truth you know exactly -- a tone at a stated frequency and a stated SNR, in
noise of a stated power -- not against fully impaired modulated signals where
a disappointing result has a dozen possible causes.

Wave-Lathe's richer generator is the right tool for training classifiers.
This is the right tool for measuring a detector.
"""

import numpy as np


def noise(n: int, power: float = 1.0, rng=None) -> np.ndarray:
    """Complex white Gaussian noise of a given total power.

    Power splits evenly across I and Q, so each gets variance power/2 -- the
    same factor of two that matters everywhere else complex noise appears.
    """
    rng = rng or np.random.default_rng()
    sigma = np.sqrt(power / 2.0)
    return (rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)).astype(
        np.complex64
    )


def tone(n: int, f_norm: float, snr_db: float, noise_power: float = 1.0,
         rng=None):
    """A single complex tone in noise.

    Args:
        n: samples.
        f_norm: frequency in cycles per sample, -0.5 to 0.5.
        snr_db: tone power relative to noise power.
        noise_power: absolute noise power, so the noise FLOOR can be moved
            independently of SNR. That separation is what makes it possible
            to show energy detection failing while CFAR holds.
        rng: NumPy Generator.

    Returns:
        (signal, clean_noise) so the same noise realisation can be reused.
    """
    rng = rng or np.random.default_rng()
    nz = noise(n, noise_power, rng=rng)

    amp = np.sqrt(noise_power * 10.0 ** (snr_db / 10.0))
    phase = rng.uniform(0, 2 * np.pi)
    carrier = amp * np.exp(2j * np.pi * f_norm * np.arange(n) + 1j * phase)

    return (carrier + nz).astype(np.complex64), nz


def occupied_band(n: int, channels, snr_db_list, noise_power: float = 1.0,
                  rng=None):
    """Several tones at once -- a band with some channels in use.

    This is the situation an AI layer actually faces: a slice of spectrum
    where some channels are busy, some are free, and the noise floor is
    whatever it is today.

    Args:
        channels: normalised frequencies of the occupied channels.
        snr_db_list: SNR for each, same length as channels.

    Returns:
        (signal, occupied_freqs)
    """
    rng = rng or np.random.default_rng()
    sig = noise(n, noise_power, rng=rng).astype(np.complex128)

    for f, snr in zip(channels, snr_db_list):
        amp = np.sqrt(noise_power * 10.0 ** (snr / 10.0))
        phase = rng.uniform(0, 2 * np.pi)
        sig += amp * np.exp(2j * np.pi * f * np.arange(n) + 1j * phase)

    return sig.astype(np.complex64), np.asarray(channels)


def welch_psd(x: np.ndarray, nperseg: int = 256, overlap: float = 0.5):
    """Power spectral density, averaged over segments, linear scale.

    Averaging is not cosmetic here. A single periodogram has roughly 100%
    standard deviation per bin regardless of length, so a detector run on one
    would false-alarm constantly. Averaging N segments cuts that variance by
    N. The SCATTER PHY averages its PSD over multiple measurements before
    reporting to the AI layer for exactly this reason.

    Returns:
        (freqs, psd, n_looks) -- n_looks is how many segments were averaged,
        which detectors need in order to set a correct threshold.
    """
    step = max(1, int(nperseg * (1 - overlap)))
    n_frames = 1 + (len(x) - nperseg) // step
    if n_frames < 1:
        raise ValueError("signal shorter than one segment")

    w = np.hanning(nperseg)
    idx = np.arange(nperseg)[None, :] + step * np.arange(n_frames)[:, None]
    frames = x[idx] * w

    spec = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    psd = (np.abs(spec) ** 2).mean(axis=0)

    # Normalise by the window's power so PSD values track the input's power
    # rather than the window's shape.
    psd /= (w**2).sum()

    return np.fft.fftshift(np.fft.fftfreq(nperseg)), psd, n_frames
