import re
import os

def check_path(path):
    print(path)
    # Read the training sample list.
    with open(path, 'r', encoding='utf-8') as f:
        train_list = [line.strip() for line in f if line.strip()]
        for scene_path in train_list:
            scene_path = re.sub(r"_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", scene_path) # Remove the timestamp from the scene path using a regular expression.
            scene_path = os.path.join('/home/wh/beyondRGB', scene_path)

            ms_h5_path = os.path.join(scene_path, "NT", "MIS.h5")
            oppo_path = os.path.join(scene_path, "NT", "oppo.h5")
            samsung_path = os.path.join(scene_path, "NT", "samsung.h5")
            cri_path = os.path.join(scene_path, "CRI","CRI.txt")

            if not os.path.exists(ms_h5_path):
                print(f"文件不存在: {ms_h5_path}")

            if not os.path.exists(oppo_path):
                print(f"文件不存在: {oppo_path}")

            if not os.path.exists(samsung_path):
                print(f"文件不存在: {samsung_path}")

            if not os.path.exists(cri_path):
                print(f"文件不存在: {cri_path}")

train_path = './BeyondRGB_Data/train_dataset_paper_git.txt'
val_path = './BeyondRGB_Data/val_dataset_paper_git.txt'
test_path = './BeyondRGB_Data/test_dataset_paper_git.txt'

check_path(train_path)
check_path(val_path)
check_path(test_path)