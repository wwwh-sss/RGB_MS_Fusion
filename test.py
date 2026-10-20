import os
import yaml
import numpy as np
import torch
import time
from tqdm import tqdm
import model
import tools
import re

class Tester:
    def __init__(self, log_path, config_file_name):
        self.log_path = log_path
        # Load the hyperparameter YAML saved during training.
        config_full_path = os.path.join(log_path, config_file_name)
        with open(config_full_path, 'r', encoding="utf-8") as f:
            self.hyper = yaml.safe_load(f)
        self.ms_size = self.hyper['TRAIN']['MS_SIZE']
        self.rgb_size = self.hyper['TRAIN']['RGB_SIZE']
        self.rgb_camera = self.hyper['TRAIN']['RGB_CAMERA']  # Store the RGB camera type.

        # Compute device.
        self.device = torch.device(self.hyper['TRAIN']['DEVICE'])

        # Initialize the model and load its weights.
        self.model = getattr(model, self.hyper["MODEL"]["NAME"])()
        self.load() # Load model weights.
        self.model = self.model.to(self.device)
        self.model.eval()

    def load(self):
        ckpt_path = os.path.join(self.log_path, "model_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"模型权重不存在：{ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=True)
        print(f"成功加载最优模型：{ckpt_path}")
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)

    def test(self):
        # Some scenes have missing files.
        # ../ beyondRGB / outdoor / 10
        # ../ beyondRGB / outdoor / 264
        with open('BeyondRGB_Data/test_dataset_paper_git.txt', "r", encoding="utf-8") as f:
            self.test_list = [line.strip() for line in f if line.strip() \
                              and tools.check_rgb(self.hyper['TRAIN']['DATASET_PATH'], line.strip())]

        # Split the test samples into Lab and Field subsets.
        self.lab_test_list = self.test_list[:121]
        self.field_test_list = self.test_list[121:]

        print("============== 开始测试 Lab 数据集 ==============")
        lab_records, lab_fps = self._test_single_subset(self.lab_test_list, "Lab")

        print("\n============== 开始测试 Field 数据集 ==============")
        field_records, field_fps = self._test_single_subset(self.field_test_list, "Field")

        print(f"\n============== 全部测试完成 ==============")
        print(f"Lab 有效样本数: {len(lab_records)}, FPS: {lab_fps:.2f}")
        print(f"Field 有效样本数: {len(field_records)}, FPS: {field_fps:.2f}")

    def _test_single_subset(self, data_list, subset_name):
        records = []
        time_list = []

        for scene_path in tqdm(data_list, desc=f"Testing {subset_name}"):
            scene_path = re.sub(r"_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", scene_path) # Remove the timestamp from the scene path using a regular expression.
            scene_path = os.path.join(self.hyper['TRAIN']['DATASET_PATH'], scene_path)

            ms_path = os.path.join(scene_path, "NT", "MIS.h5")
            rgb_path = os.path.join(scene_path, "NT", self.rgb_camera + '.h5')
            cri_path = os.path.join(scene_path, "CRI", "CRI.txt")

            # Read the multispectral and RGB images.
            ms_patch, rgb_patch = tools.process_image(ms_path, rgb_path, self.rgb_camera, self.ms_size)
            ms_patch = ms_patch[np.newaxis, :]
            rgb_patch = rgb_patch[np.newaxis, :]
            # Read the ground-truth SPD.
            gt_spd_np = tools.read_spd_from_cri(cri_path)

            # ===================== Model inference =====================
            t0 = time.time()
            with torch.no_grad():
                spd_pred = self.model(torch.from_numpy(rgb_patch).to(self.device),
                                      torch.from_numpy(ms_patch).to(self.device))
            cost = time.time() - t0

            # Convert to NumPy arrays to compute metrics.
            pred_np = spd_pred.detach().cpu().numpy().squeeze(0)
            angle = tools.getMAE_numpy_2D(pred_np, gt_spd_np)
            records.append({'angle':angle}) # Record the angular error.
            time_list.append(cost) # Record inference time.

        # Compute FPS.
        total_time = sum(time_list)
        fps = len(time_list) / total_time if total_time > 0 else 0.0
        print(f"\n【{subset_name} 数据集】有效样本：{len(records)} 个，FPS: {fps:.2f}")

        if len(records) > 0:
            self.statistic(records, subset_name)
        else:
            print(f"{subset_name} 无有效测试样本！")
        return records, fps

    def statistic(self, records, subset_name):
        records_sorted = sorted(records, key=lambda x: x["angle"])
        angle_list = [item["angle"] for item in records_sorted]
        n = len(angle_list)

        # Compute evaluation statistics.
        mean_angle = np.mean(angle_list)
        std_angle = np.std(angle_list)
        median_angle = np.median(angle_list)
        q1, q3 = np.percentile(angle_list, [25, 75])
        trimean = (2 * median_angle + q1 + q3) / 4
        best_25 = np.mean(angle_list[:max(1, int(n * 0.25))])
        tail_idx = int(n * 0.75)
        worst_25 = np.mean(angle_list[tail_idx:]) if tail_idx < n else angle_list[-1]
        per95 = np.percentile(angle_list, 95)

        # Print evaluation metrics.
        print(f"\n【{subset_name} 数据集统计结果】")
        print(f"Mean: {mean_angle:.4f}°")
        print(f"Std: {std_angle:.4f}°")
        print(f"Median: {median_angle:.4f}°")
        print(f"Trimean: {trimean:.4f}°")
        print(f"Best25: {best_25:.4f}°")
        print(f"Worst25: {worst_25:.4f} °")
        print(f"95%: {per95:.4f}°")


if __name__ == '__main__':
    # Set this to the directory containing your training outputs.
    MODEL_PATH = "./BeyondRGB_experiment/09102149"
    tester = Tester(log_path = MODEL_PATH, config_file_name="hyper_parameters.yaml")
    tester.test()