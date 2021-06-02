from pathlib import Path
import albumentations
from src.trasnforms import NoiseInjection, RandomVolume, PinkNoise, AddGaussianSNR
from src.dataset import PANNsMultiLabelDataset, TARGET_COLUMNS

def train_transform():
    return albumentations.Compose([
        NoiseInjection(max_noise_level=0.04, p=0.3),
        RandomVolume(limit=4, p=0.4),
        PinkNoise(p=0.2),
        # AddBackgroundNoise(noise_sound_array, p=0.3),
        AddGaussianSNR(min_SNR=0.001, max_SNR=0.015, p=0.3)
    ], p=0.3)

class CFG:
    ######################
    # Globals #
    ######################
    seed = 1213
    epochs = 55
    train = True
    folds = [2]

    ######################
    # Data #
    ######################
    train_datadir = Path("../input/birdclef-2021/train_short_audio")
    train_csv = Path("../input/birdclef-2021/train_metadata.csv")
    train_soundscape = Path("../input/birdclef-2021/train_soundscape_labels.csv")

    ######################
    # Dataset #
    ######################
    transforms = train_transform()
    period = 30
    n_mels = 128
    fmin = 20
    fmax = 16000
    n_fft = 2048
    hop_length = 512
    sample_rate = 32000
    melspectrogram_parameters = {
        "n_mels": 224,
        "fmin": 20,
        "fmax": 16000
    }
    ######################
    # model #
    ######################
    model_config = {
        "sample_rate": 32000,
        "window_size": 1024,
        "hop_size": 320,
        "mel_bins": 128,
        "fmin": 20,
        "fmax": 16000,
        "classes_num": len(TARGET_COLUMNS),
        "apply_aug": True,
        "top_db": None
    }
    ######################
    # Loaders #
    ######################
    loader_params = {
        "train": {
            "batch_size": 32,
            "num_workers": 8,
            "shuffle": True
        },
        "valid": {
            "batch_size": 128,
            "num_workers": 8,
            "shuffle": False
        }
    }

    ######################
    # Split #
    ######################
    split = "StratifiedKFold"
    split_params = {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 1213
    }

    ######################
    # Optimizer #
    ######################
    optimizer_name = "Adam"
    base_optimizer = "Adam"
    optimizer_params = {
        "lr": 0.001
    }

    ######################
    # Scheduler #
    ######################
    scheduler_name = "CosineAnnealingLR"
    scheduler_params = {
        "T_max": 10
    }
