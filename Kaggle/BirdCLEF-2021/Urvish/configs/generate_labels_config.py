from pathlib import Path
import albumentations


from src.trasnforms import NoiseInjection, RandomVolume, PinkNoise, AddGaussianSNR
from src.dataset import TARGET_COLUMNS


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
    epochs = 100
    train = True
    folds = [0]

    ######################
    # Data #
    ######################
    train_datadir = Path("/home/wangxiang/dat01/WWX/Competition/kaggle/birdclef-2021/train_short_audio")
    train_csv = Path("/home/wangxiang/dat01/WWX/Competition/kaggle/birdclef-2021/train_metadata.csv")
    train_soundscape = Path("/home/wangxiang/dat01/WWX/Competition/kaggle/birdclef-2021/train_soundscape_labels.csv")

    ######################
    # Dataset #
    ######################
    transforms = train_transform()
    period = 30
    n_mels = 64
    fmin = 50
    fmax = 14000
    n_fft = 2048
    hop_length = 320
    sample_rate = 32000
    ######################
    # model #
    ######################
    model_config = {
        "sample_rate": 32000,
        "window_size": 1024,
        "hop_size": 320,
        "mel_bins": 64,
        "fmin": 50,
        "fmax": 14000,
        "classes_num": len(TARGET_COLUMNS),
        "apply_aug": True,
        "top_db": None
    }

    model_dir = Path("home/wangxiang/dat01/WWX/Competition/kaggle/birdclef-2021/models/")

    threshold = 0.9
    ######################
    # Loaders #
    ######################
    loader_params = {
        "valid": {
            "batch_size": 128,
            "num_workers": 16,
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
    ########################
    # # Save path and files
    ########################
    save_path = "./new_labels.csv"
    output_dir = "outputs_labels"

    # ######################
    # # Optimizer #
    # ######################
    # optimizer_name = "Adam"
    # base_optimizer = "Adam"
    # optimizer_params = {
    #     "lr": 0.001
    # }

    # ######################
    # # Scheduler #
    # ######################
    # scheduler_name = "CosineAnnealingLR"
    # scheduler_params = {
    #     "T_max": epochs
    # }