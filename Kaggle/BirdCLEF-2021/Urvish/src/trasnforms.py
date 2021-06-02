import numpy as np
import random
import librosa
import colorednoise as cn
import albumentations
from albumentations.core.transforms_interface import DualTransform, BasicTransform
from albumentations.core.transforms_interface import ImageOnlyTransform


def calculate_rms(samples):
    """Given a numpy array of audio samples, return its Root Mean Square (RMS)."""
    return np.sqrt(np.mean(np.square(samples), axis=-1))


def calculate_desired_noise_rms(clean_rms, snr):
    """
    Given the Root Mean Square (RMS) of a clean sound and a desired signal-to-noise ratio (SNR),
    calculate the desired RMS of a noise sound to be mixed in.
    Based on https://github.com/Sato-Kunihiko/audio-SNR/blob/8d2c933b6c0afe6f1203251f4877e7a1068a6130/create_mixed_audio_file.py#L20
    :param clean_rms: Root Mean Square (RMS) - a value between 0.0 and 1.0
    :param snr: Signal-to-Noise (SNR) Ratio in dB - typically somewhere between -20 and 60
    :return:
    """
    a = float(snr) / 20
    noise_rms = clean_rms / (10 ** a)
    return noise_rms


class Normalize:
    def __call__(self, y: np.ndarray):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


class NewNormalize:
    def __call__(self, y: np.ndarray):
        y_mm = y - y.mean()
        return y_mm / y_mm.abs().max()


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class AudioTransform(BasicTransform):
    """Transform for Audio task"""

    @property
    def targets(self):
        return {"image": self.apply}

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params


class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5, sr=32000):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20, sr=32000):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20, sr=32000):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 /
                     a_pink * a_noise).astype(y.dtype)
        return augmented


class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_range=5, sr=32000):
        super().__init__(always_apply, p)
        self.max_range = max_range
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        n_steps = np.random.randint(-self.max_range, self.max_range)
        augmented = librosa.effects.pitch_shift(y, self.sr, n_steps)
        return augmented


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1, sr=32000):
        super().__init__(always_apply, p)
        self.max_rate = max_rate
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented


def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10**(db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    """
    Low level API for decreasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to decrease
    Returns
    -------
    applied: numpy.ndarray
        audio with decreased volume
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Low level API for increasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to increase
    Returns
    -------
    applied: numpy.ndarray
        audio with increased volume
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(y, db)
        else:
            return volume_down(y, db)


class CosineVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        return y * dbs


def drop_stripes(image: np.ndarray, dim: int, drop_width: int, stripes_num: int):
    total_width = image.shape[dim]
    lowest_value = image.min()
    for _ in range(stripes_num):
        distance = np.random.randint(low=0, high=drop_width, size=(1,))[0]
        begin = np.random.randint(
            low=0, high=total_width - distance, size=(1,))[0]

        if dim == 0:
            image[begin:begin + distance] = lowest_value
        elif dim == 1:
            image[:, begin + distance] = lowest_value
        elif dim == 2:
            image[:, :, begin + distance] = lowest_value
    return image


class TimeFreqMasking(ImageOnlyTransform):
    def __init__(self,
                 time_drop_width: int,
                 time_stripes_num: int,
                 freq_drop_width: int,
                 freq_stripes_num: int,
                 always_apply=False,
                 p=0.5):
        super().__init__(always_apply, p)
        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num
        self.freq_drop_width = freq_drop_width
        self.freq_stripes_num = freq_stripes_num

    def apply(self, img, **params):
        img_ = img.copy()
        if img.ndim == 2:
            img_ = drop_stripes(
                img_, dim=0, drop_width=self.freq_drop_width, stripes_num=self.freq_stripes_num)
            img_ = drop_stripes(
                img_, dim=1, drop_width=self.time_drop_width, stripes_num=self.time_stripes_num)
        return img_


class CutOut(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(CutOut, self).__init__(always_apply, p)

    def apply(self, data, **params):
        '''
        data : ndarray of audio timeseries
        '''
        sound = data
        start_ = np.random.randint(0, len(sound))
        end_ = np.random.randint(start_, len(sound))

        sound[start_:end_] = 0

        return sound


class AddCustomNoise(AudioTransform):
    """
    This Function allows you to add noise from any custom file you want.
    The idea is to load the sounds files first as a list of arrays using 
    librosa so that at inference time we do not have to wait for that.
    """

    def __init__(self, sound_array, always_apply=False, p=0.5):
        super(AddCustomNoise, self).__init__(always_apply, p)
        '''
        The array must be of [sound1, sound2, sound3, ....]
        '''

        self.sound_array = sound_array

    def apply(self, data, **params):
        '''
        data : ndarray of audio timeseries
        '''
        sound = data
        noise = self.sound_array[int(
            np.random.uniform(0, len(self.sound_array)))]

        if len(noise) > len(sound):
            start_ = len(noise)-len(sound)
            noise = noise[start_: start_+len(sound)]
        else:
            noise = np.pad(noise, (0, len(sound)-len(noise)), "constant")

        start_ = np.random.randint(0, len(sound))
        end_ = np.random.randint(start_, len(sound))

        sound[start_:end_] = noise[start_:end_]

        return sound


class AddBackgroundNoise(AudioTransform):
    """Adds background noise and does mix-up of two audio files."""

    def __init__(self, sound_array, min_snr_in_db=3, max_snr_in_db=30, always_apply=False, p=0.5):
        super(AddBackgroundNoise, self).__init__(always_apply, p)
        '''
        The array must be of [sound1, sound2, sound3, ....]
        '''

        self.sound_array = sound_array
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db

    def apply(self, data, **params):
        sound = data
        noise = self.sound_array[int(
            np.random.uniform(0, len(self.sound_array)))]

        snr_in_db = random.uniform(self.min_snr_in_db, self.max_snr_in_db)

        clean_rms = calculate_rms(sound)
        noise_rms = calculate_rms(noise)
        desired_noise_rms = calculate_desired_noise_rms(
            clean_rms, snr_in_db
        )

        # Repeat the sound if it shorter than the input sound
        num_samples = len(sound)
        while len(noise) < num_samples:
            noise = np.concatenate((noise, noise))

        if len(noise) > num_samples:
            noise = noise[0:num_samples]

        # Adjust the noise to match the desired noise RMS
        noise = noise * (desired_noise_rms / noise_rms)

        return sound + noise


class AddGaussianSNR(AudioTransform):
    """It simply add some random value into data by using numpy"""

    def __init__(self, min_SNR=0.001, max_SNR=1.0, always_apply=False, p=0.5):
        super(AddGaussianSNR, self).__init__(always_apply, p)
        self.min_SNR = min_SNR
        self.max_SNR = max_SNR

    def apply(self, data, **params):
        sound = data
        std = np.std(sound)
        noise_std = random.uniform(self.min_SNR * std, self.max_SNR * std)
        noise = np.random.normal(
            0.0, noise_std, size=len(sound)).astype(np.float32)
        return sound + noise
