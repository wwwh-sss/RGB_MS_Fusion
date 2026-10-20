import os
import re
import h5py
import yaml
import numpy as np
from tqdm import tqdm
from tools import (
    check_rgb,
    process_image,
    read_spd_from_cri,
)

def pack_beyondrgb_dataset(dataset_root,txt_path,output_h5_path,rgb_camera,crop_size,compression=None):
    """
    Pack BeyondRGB data into one HDF5 file using the original Dataset preprocessing.

    Args:
        dataset_root: Dataset root, from hyper['TRAIN']['DATASET_PATH'].
        txt_path: Path to the training or validation sample list.
        output_h5_path: Output HDF5 file path.
        rgb_camera: RGB camera type, 'oppo' or 'samsung'.
        crop_size: Spatial crop size.
        compression: HDF5 compression; None favors speed, 'gzip' saves space.
    """
    # ========== 1. Read and filter samples using the original Dataset logic ==========
    with open(txt_path, "r", encoding="utf-8") as f:
        sample_list = [
            line.strip()
            for line in f
            if line.strip() and check_rgb(dataset_root, line.strip())
        ]

    total_num = len(sample_list)
    if total_num == 0:
        raise ValueError("有效样本数为0，请检查路径和过滤条件")
    print(f"有效样本总数: {total_num}")

    # ========== 2. Initialize the HDF5 file ==========
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

    # Process the first sample to determine the array shapes.
    first_scene = sample_list[0]
    scene_path_clean = re.sub(r"_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", first_scene)
    scene_full = os.path.join(dataset_root, scene_path_clean)
    ms_path = os.path.join(scene_full, "NT", "MIS.h5")
    rgb_path = os.path.join(scene_full, "NT", rgb_camera + ".h5")
    cri_path = os.path.join(scene_full, "CRI", "CRI.txt")

    ms_first, rgb_first = process_image(ms_path, rgb_path, rgb_camera, crop_size)
    spd_first = read_spd_from_cri(cri_path)

    # Create the datasets.
    with h5py.File(output_h5_path, "w") as f:
        # RGB dataset [N, 3, H, W].
        f.create_dataset(
            "rgb",
            shape=(total_num,) + rgb_first.shape,
            dtype=np.float32,
            chunks=(1,) + rgb_first.shape,
            compression=compression,
        )
        # Multispectral dataset [N, 16, H, W].
        f.create_dataset(
            "msi",
            shape=(total_num,) + ms_first.shape,
            dtype=np.float32,
            chunks=(1,) + ms_first.shape,
            compression=compression,
        )
        # Ground-truth SPD dataset [N, 36].
        f.create_dataset(
            "spd",
            shape=(total_num, len(spd_first)),
            dtype=np.float32,
            chunks=(1, len(spd_first)),
            compression=compression,
        )

        # Write the first sample.
        f["rgb"][0] = rgb_first.astype(np.float32)
        f["msi"][0] = ms_first.astype(np.float32)
        f["spd"][0] = spd_first.astype(np.float32)

        # ========== 3. Process and write the remaining samples ==========
        for idx in tqdm(range(1, total_num), desc="打包进度"):
            scene_path = sample_list[idx]
            scene_path_clean = re.sub(
                r"_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", scene_path
            )
            scene_full = os.path.join(dataset_root, scene_path_clean)

            ms_path = os.path.join(scene_full, "NT", "MIS.h5")
            rgb_path = os.path.join(scene_full, "NT", rgb_camera + ".h5")
            cri_path = os.path.join(scene_full, "CRI", "CRI.txt")

            try:
                ms_patch, rgb_patch = process_image(
                    ms_path, rgb_path, rgb_camera, crop_size
                )
                gt_spd = read_spd_from_cri(cri_path)

                f["rgb"][idx] = rgb_patch.astype(np.float32)
                f["msi"][idx] = ms_patch.astype(np.float32)
                f["spd"][idx] = gt_spd.astype(np.float32)

            except Exception as e:
                print(f"\n样本 {idx} 处理失败: {scene_path}, 错误: {e}")
                # Fill failed sample slots with zeros; adapt this to skip them if needed.
                f["rgb"][idx] = 0
                f["msi"][idx] = 0
                f["spd"][idx] = 0

    print(f"打包完成，输出文件: {output_h5_path}")
    print(f"RGB shape: {rgb_first.shape}, MS shape: {ms_first.shape}, SPD length: {len(spd_first)}")


if __name__ == "__main__":
    # Load the configuration.
    with open("hyper_parameters.yaml", "r", encoding="utf-8") as f:
        hyper = yaml.safe_load(f)

    dataset_root = hyper["TRAIN"]["DATASET_PATH"]
    rgb_camera = hyper["TRAIN"]["RGB_CAMERA"]
    crop_size = 256

    # ========== Pack the training set ==========
    print("===== 开始打包训练集 =====")
    pack_beyondrgb_dataset(
        dataset_root=dataset_root,
        txt_path="./BeyondRGB_Data/train_dataset_paper_git.txt",
        output_h5_path="./BeyondRGB_Data/train_packed_samsung.h5",
        rgb_camera=rgb_camera,
        crop_size=crop_size,
        compression=None,  # Use None for speed or 'gzip' to save space.
    )

    # ========== Pack the validation set ==========
    print("\n===== 开始打包验证集 =====")
    pack_beyondrgb_dataset(
        dataset_root=dataset_root,
        txt_path="./BeyondRGB_Data/val_dataset_paper_git.txt",
        output_h5_path="./BeyondRGB_Data/val_packed_samsung.h5",
        rgb_camera=rgb_camera,
        crop_size=crop_size,
        compression=None,
    )
