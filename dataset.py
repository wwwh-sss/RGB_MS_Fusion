import h5py
import torch
import random
from torch.utils.data import Dataset
# ====================== Training Dataset ======================
class CustomDataset(Dataset):
    """
    Read packed training HDF5 data using the original CustomDataset interface.
    Preprocessing includes demosaicing, short-edge resizing, center cropping,
    channel-first (CHW) layout, and sample filtering.
    Apply lightweight augmentation during training only.
    """
    def __init__(self, hyper, random_seed=42):
        self.hyper = hyper
        self.random_seed = random_seed  # Keep the original interface; packed data no longer requires a preprocessing seed.

        # Read the packed training data path from the configuration.
        self.h5_path = hyper["TRAIN"]["TRAIN_PACKED_H5"]

        # Read the total number of samples.
        with h5py.File(self.h5_path, "r") as f:
            self.total_num = f["rgb"].shape[0]
            print(f"训练集总数：{self.total_num}")

        # ========== Data augmentation settings ==========
        self.flip_prob = 0.5  # Geometric flips.

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        # Open the file for each read to support multiple DataLoader workers.
        with h5py.File(self.h5_path, "r") as f:
            rgb = torch.from_numpy(f["rgb"][idx]).float()
            ms = torch.from_numpy(f["msi"][idx]).float()
            gt_spd = torch.from_numpy(f["spd"][idx]).float()

        # ========== Data augmentation: training only ==========
        # 1. Apply matching geometric flips to both input images.
        if random.random() < self.flip_prob:
            rgb = torch.flip(rgb, dims=[-1])
            ms = torch.flip(ms, dims=[-1])

        if random.random() < self.flip_prob:
            rgb = torch.flip(rgb, dims=[-2])
            ms = torch.flip(ms, dims=[-2])
        return {
            "rgb_patch": rgb,
            "ms_patch": ms,
            "gt_spd": gt_spd
        }


# ====================== Validation Dataset ======================
class CustomValDataset(Dataset):
    """
    Read packed validation HDF5 data.
    Do not apply data augmentation during validation.
    """
    def __init__(self, hyper):
        self.hyper = hyper
        # Read the packed validation data path from the configuration.
        self.h5_path = hyper["TRAIN"]["VAL_PACKED_H5"]

        with h5py.File(self.h5_path, "r") as f:
            self.total_num = f["rgb"].shape[0]
            print(f"验证集总数：{self.total_num}")

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            rgb = torch.from_numpy(f["rgb"][idx]).float()
            ms = torch.from_numpy(f["msi"][idx]).float()
            gt_spd = torch.from_numpy(f["spd"][idx]).float()

        return {
            "rgb_patch": rgb,
            "ms_patch": ms,
            "gt_spd": gt_spd
        }


# Dataset inspection example.
if __name__ == "__main__":
    import yaml

    with open("hyper_parameters.yaml", "r", encoding="utf-8") as f:
        hyper = yaml.safe_load(f)

    ds_train = CustomDataset(hyper)
    ds_val = CustomValDataset(hyper)

    sample_train = ds_train[0]
    sample_val = ds_val[0]

    print("训练集样本形状：")
    print(f"RGB: {sample_train['rgb_patch'].shape}")
    print(f"MS: {sample_train['ms_patch'].shape}")
    print(f"SPD: {sample_train['gt_spd'].shape}")

    print("\n验证集样本形状：")
    print(f"RGB: {sample_val['rgb_patch'].shape}")
    print(f"MS: {sample_val['ms_patch'].shape}")
    print(f"SPD: {sample_val['gt_spd'].shape}")
