import torch
import numpy as np
import random
import os
import cv2
import math
from torch.optim.lr_scheduler import _LRScheduler
from sympy import true
from torchvision.utils import make_grid, save_image
import h5py
import json
import torch.nn.functional as F
import re

# Small constant for numerical stability.
EPS = 1e-8
# Multispectral channel triplets specified in the paper.
TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6),
    (5, 6, 7), (6, 7, 8), (7, 8, 9), (8, 9, 10), (9, 10, 11),
    (10, 11, 12), (11, 12, 13), (12, 13, 14), (13, 14, 15),
    (0, 5, 9), (1, 6, 10), (2, 7, 11), (3, 8, 12), (4, 9, 13),
    (5, 10, 14), (3, 6, 15)
]

# Set random seeds for the program.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    os.environ['PYTHONHASHSEED'] = str(seed)

# Create a directory.
def make_dir(path):
    os.makedirs(path,exist_ok=True)

# Display a 16-bit PNG image.
def show_16png(img_path):
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(np.float32)
    img_nor = img/np.max(img)
    img_nor_8bit = (img_nor*255).astype(np.uint8)
    cv2.imshow("example image", img_nor_8bit)
    cv2.waitKey(0) # Wait for a key press.

# Check that the multispectral image and both RGB images exist.
# Return a truthy value only when all required images are present.
def check_rgb(dataset_path, scene_path):
    scene_path = re.sub(r"_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", scene_path)  # Remove the timestamp from the scene path using a regular expression.
    flag = (scene_path != 'outdoor/72') # Exclude a scene with missing image data.
    scene_path = os.path.join(dataset_path, scene_path) # Dataset path.
    ms_h5_path = os.path.join(scene_path, "NT", "MIS.h5")
    oppo_h5_path = os.path.join(scene_path, "NT", "oppo.h5")
    samsung_h5_path = os.path.join(scene_path, "NT", "samsung.h5")

    if os.path.exists(ms_h5_path) and os.path.exists(oppo_h5_path) \
        and os.path.exists(samsung_h5_path) and flag:
        return true
    else:
        print(scene_path)
    # return os.path.exists(ms_h5_path) and os.path.exists(oppo_h5_path) \
    #     and os.path.exists(samsung_h5_path) and flag

# Save an array as a PNG image.
def show_array(img, path):
    img = img / img.max()
    img_8bit = (img*255).astype(np.uint8)
    cv2.imwrite(path, img_8bit)

# Mean squared error (MSE).
def MSE(pred, gt):
    return np.mean((pred - gt) ** 2, axis = 1)

def getMAE_numpy_2D(pred, gt):
    pred = pred / (np.linalg.norm(pred, axis=-1, ord=2, keepdims=True) + EPS)
    gt = gt / (np.linalg.norm(gt, axis=-1, ord=2, keepdims=True) + EPS)
    dot_product = (pred * gt).sum(axis=-1)
    clipped_result = np.clip(dot_product, -1 + EPS, 1 - EPS)
    radian = np.arccos(clipped_result)
    angle = (radian / np.pi) * 180
    return angle

def getMAE_numpy_3D(pred, gt):
    product = (pred * gt).sum(axis=2)  # (256,256,3)
    pred_length = np.linalg.norm(pred, axis=2, ord=2)
    gt_length = np.linalg.norm(gt, axis=2, ord=2)
    result = product / ((pred_length * gt_length) + EPS)
    clipped_result = np.clip(result, -1, 1)
    radian = np.arccos(clipped_result)
    angle = (radian / math.pi) * 180
    return angle

# Compute summary statistics.
def errors_statistics(errors):
    mean = np.mean(errors, axis=0) # mean
    q1 = np.percentile(errors, 25, axis=0) # q1
    median = np.percentile(errors, 50, axis=0) # q2(median)
    q3 = np.percentile(errors, 75, axis=0) # q3
    trimean = (q1 + 2 * median + q3) / 4
    return [mean, q1, median, q3, trimean]

# Combine a batch of images into a grid and save it.
def save_batch(batch, idx):
    images = make_grid(batch, 4, 2)
    save_image(images, f'batch_{idx}_show.png')

# Process multispectral images for the training set.
def process_image(ms_path, rgb_path, camera, crop_size):

    CFA_MAP = [
        [9, 11, 13, 15],
        [8, 10, 12, 14],
        [1, 3, 5, 7],
        [0, 2, 4, 6]
    ]

    with h5py.File(ms_path, "r", driver='core') as f:
        ms_raw = f['MIS'][:]

    # ../beyondRGB/outdoor/277/NT/oppo.h5 uses the MIS key.
    # ../beyondRGB/outdoor/68/NT/oppo.h5 uses the MIS key.
    # ../beyondRGB/outdoor/72/NT/oppo.h5 contains incomplete data.
    try:
        with h5py.File(rgb_path, "r") as f:
            if camera in f:
                rgb_raw = f[camera][:]
            else:
                rgb_raw = f['MIS'][:]
    except Exception as e:
        breakpoint()
        tmp = 0
    # Demosaic the multispectral image.
    h_phy, w_phy = ms_raw.shape
    h_eff = h_phy // 4
    w_eff = w_phy // 4
    ms_16ch = np.empty((h_eff, w_eff, 16), dtype=np.float32) # Store the demosaiced image.
    # Extract CFA channels using array slicing.
    for hw_row in range(4):
        for hw_col in range(4):
            ch = CFA_MAP[hw_row][hw_col]
            ms_16ch[:, :, ch] = ms_raw[hw_row::4, hw_col::4]
    # Resize the short edge and center-crop both multispectral and RGB images.
    ms_resize = resize_short_edge(ms_16ch, crop_size)
    ms_patch = center_crop(ms_resize, crop_size)

    rgb_resize = resize_short_edge(rgb_raw, crop_size)
    rgb_patch = center_crop(rgb_resize, crop_size)

    return ms_patch.transpose(2, 0, 1), rgb_patch.transpose(2, 0, 1)


# Process multispectral images for validation and testing.
def process_ms_image_2(ms_h5_path, ms_json, crop_size, n_bins, hist_range):

    CFA_MAP = [
        [9, 11, 13, 15],
        [8, 10, 12, 14],
        [1, 3, 5, 7],
        [0, 2, 4, 6]
    ]

    with h5py.File(ms_h5_path, "r", driver='core') as f:
        ms_raw = f['MIS'][:]

    # Demosaic the multispectral image.
    h_phy, w_phy = ms_raw.shape
    h_eff = h_phy // 4
    w_eff = w_phy // 4
    ms_16ch = np.empty((h_eff, w_eff, 16), dtype=np.float32) # Store the demosaiced image.
    # Extract CFA channels using array slicing.
    for hw_row in range(4):
        for hw_col in range(4):
            ch = CFA_MAP[hw_row][hw_col]
            ms_16ch[:, :, ch] = ms_raw[hw_row::4, hw_col::4]

    # Get the multispectral white point.
    ms_wp = extract_ms_white_point(ms_16ch, ms_json)

    # Remove the color chart from the image.
    ms_wo_cc = remove_cc(ms_16ch, ms_json)

    # Resize the short edge while preserving the aspect ratio, then center-crop.
    ms_resize = resize_short_edge(ms_wo_cc, crop_size)
    ms_center = center_crop(ms_resize, crop_size) # Apply a center crop.

    # Build patch triplets and their log-chromaticity histograms.
    ms_crop_hist = generate_21_histograms(ms_center, n_bins, hist_range)

    # Compute the UV cost map.
    uv_cost_map = generate_uv_cost_maps(ms_wp, n_bins, hist_range)

    return ms_crop_hist, uv_cost_map

# Random cropping helper.
def random_crops(ms_np, crop_size, crop_number):
    # Check whether the requested size is a scalar or a sequence.
    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size # Use equal crop height and width for a scalar size.
    else:
        crop_h, crop_w = crop_size

    H, W, _ = ms_np.shape # Original image dimensions.

    crops = []
    
    for _ in range(crop_number):
        h_start = np.random.randint(0, H - crop_size + 1) 
        w_start = np.random.randint(0, W - crop_size + 1)
        crops.append(ms_np[h_start:h_start + crop_h, w_start:w_start + crop_w, :])
    return crops

# Resize the short edge while preserving the aspect ratio.
def resize_short_edge(img, target_size):
    """
    Resize an image array while preserving its aspect ratio.

    Args:
        img: NumPy array with shape (H, W, C).
        target_size: Integer target length for the shorter edge.

    Returns:
        The resized image array.
    """
    target_h = target_w = target_size

    h, w = img.shape[:2]

    # Compute the scale needed to resize the short edge to the target size.
    scale = max(target_h / h, target_w / w)

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Resize each channel separately.
    resized_channels = []
    for i in range(img.shape[2]):
        resized_channels.append(
            cv2.resize(img[..., i], (new_w, new_h), interpolation=cv2.INTER_AREA)
        )
    resized = np.stack(resized_channels, axis=-1)
    return resized

# Center crop.
def center_crop(img, target_size):
    """
    Extract a center crop from an image.

    Args:
        img: NumPy image array.
        target_size: Integer for a square crop, or a (height, width) tuple.

    Returns:
        The cropped image array.
    """
    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size

    h, w = img.shape[:2]

    start_h = max(0, (h - target_h) // 2)
    start_w = max(0, (w - target_w) // 2)

    end_h = start_h + target_h
    end_w = start_w + target_w

    return img[start_h:end_h, start_w:end_w, :]

# Ground-truth RGB white point.
def clip_bbox(bbox, img_shape):
    x1, y1, x2, y2 = bbox
    h, w = img_shape
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return x1, y1, x2, y2

def single_gt_light_rgb(rgb_h5_path, gt_json_path, target_patch="patch_22"):
    if not os.path.exists(gt_json_path) or not os.path.exists(rgb_h5_path):
        return np.ones(3, dtype=np.float32)

    try:
        with open(gt_json_path, "r", encoding="utf-8") as f:
            cc_data = json.load(f)
        corners = cc_data[target_patch]["corners"]

        x_coords = [c[0] for c in corners]
        y_coords = [c[1] for c in corners]
        x1, x2 = int(min(x_coords)), int(max(x_coords))
        y1, y2 = int(min(y_coords)), int(max(y_coords))

        with h5py.File(rgb_h5_path, "r", driver='core') as f:
            img = f["oppo"][:]

        x1, y1, x2, y2 = clip_bbox((x1, y1, x2, y2), img.shape[:2])
        patch_rgb = img[y1:y2, x1:x2].mean(axis=(0, 1))
        return patch_rgb.astype(np.float32)
    except:
        return np.ones(3, dtype=np.float32)

# Multispectral white-point location.
def get_valid_bbox(corners, img_h, img_w):
    xs = [int(c[0]) for c in corners]
    ys = [int(c[1]) for c in corners]
    # Find the top-left and bottom-right coordinates.
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    # Clamp the coordinates to the image bounds.
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w - 1))
    y2 = max(y1 + 1, min(y2, img_h - 1))
    return x1, y1, x2, y2

# Extract the white point from the image.
def extract_ms_white_point(ms_np, gt_json_path, target_patch="patch_20"):
    h, w = ms_np.shape[:2]
    with open(gt_json_path, "r", encoding="utf-8") as f:
        corner_20 = json.load(f)[target_patch]["corners"]
    
    corners_20_xy = [[round(x/4), round(y/4)] for x, y in corner_20] # Divide the coordinates by four.
    x1, y1, x2, y2 = get_valid_bbox(corners_20_xy, h, w) # Get a valid bounding box.

    return np.median(ms_np[y1:y2, x1:x2], axis=(0, 1)).astype(np.float32)

# Remove the color chart from the multispectral image.
def remove_cc(ms_np, gt_json_path):
    h, w = ms_np.shape[:2]
    all_corners = [] # Collect all coordinates.
    with open(gt_json_path, "r", encoding="utf-8") as f:
        json_f = json.load(f)
        # Collect patch coordinates to locate the color chart region.
        for i in range(1, 25):
            for corordinate in json_f['patch_' + str(i)]['corners']:
                all_corners.append(corordinate)

    corners_all_xy = [[round(x/4), round(y/4)] for x, y in all_corners] # Divide the coordinates by four.
    x1_all, y1_all, x2_all, y2_all = get_valid_bbox(corners_all_xy, h, w) # Color chart coordinates.

    ms_np[y1_all:y2_all, x1_all:x2_all, :] = 0
    return ms_np

# Read the ground-truth SPD.
def read_spd_from_cri(cri_txt_path):
    with open(cri_txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    spd_line = ""
    for idx, line in enumerate(lines):
        if "Illumination condition:" in line and "Emissive" in line:
            spd_line = lines[idx + 1]
            break

    str_vals = spd_line.strip(',').split(',')
    spd_values = np.array([float(v.strip()) for v in str_vals], dtype=np.float32)
    return spd_values

# Compute a log-chromaticity histogram.
def compute_log_chroma_histogram(triplet, nbins, hist_range):
    c1, c2, c3 = triplet[:, :, 0], triplet[:, :, 1], triplet[:, :, 2]

    # Paper formulation: u = log(c2/c1), v = log(c2/c3).
    c1 = np.maximum(c1, EPS) # Avoid division by zero.
    c3 = np.maximum(c3, EPS) # Avoid division by zero.
    u = np.log((c2) / (c1)).flatten() # [H*W]
    v = np.log((c2) / (c3)).flatten()  # [H*W]
    luminance = np.linalg.norm(triplet, axis=2).flatten()  # Intensity.

    min_val, max_val = hist_range # 
    bin_width = (max_val - min_val) / nbins # 

    # Compute the two-dimensional histogram.
    hist, _, _ = np.histogram2d(u, v, bins = nbins,\
                        range = ((min_val - bin_width / 2, max_val + bin_width / 2),) * 2, \
                        weights = luminance)
    
    # Apply square-root normalization to match the original CCC implementation.
    norm_factor = hist.sum() + EPS
    hist = np.sqrt(hist / norm_factor)
    return hist

# Build triplets and corresponding log-chromaticity histograms for each patch.
def generate_21_histograms(ms_crops, nbins=128, hist_range = [-3.2, 3.2]):
    all_hist = [] # Collect all histograms.
    for i in range(len(ms_crops)):
        crop_hist = []
        for (c1, c2, c3) in TRIPLETS:
            triplet = ms_crops[i][:, :, [c1, c2, c3]]
            hist = compute_log_chroma_histogram(triplet, nbins, hist_range)
            crop_hist.append(hist)
        all_hist.append(crop_hist)
    return np.array(all_hist, dtype=np.float32)

# Build UV cost maps.
def generate_uv_cost_maps(ms_wp, n_bins, hist_range):
    ms_wp_triplet = get_21_uv_gt(ms_wp) # Compute multispectral white-point triplets and their UV chromaticities.
    H, _ = ms_wp_triplet.shape # Number of triplets.
    # Create the grid.
    u_cor = np.linspace(hist_range[0], hist_range[1], n_bins)
    v_cor = np.linspace(hist_range[0], hist_range[1], n_bins)
    # Compute grid coordinates for the centers of the chromaticity histogram bins.
    u_map, v_map = np.meshgrid(u_cor, v_cor)
    # Corresponds to Equation 5 in the paper.
    u_map = np.exp(-np.expand_dims(u_map, axis = 2))
    v_map = np.exp(-np.expand_dims(v_map, axis = 2))
    one_map = np.ones_like(u_map) # Matrix of ones.
    exp_uv_map = np.concatenate((u_map, one_map, v_map), axis = 2)

    # Compute all cost maps.
    all_cost_map = np.zeros((H, n_bins, n_bins), dtype=np.float32)

    # Compute the cost map.
    for i in range(H):
        gt_uv = ms_wp_triplet[i, :] # Ground-truth UV values.
        exp_gt_uv = np.exp(-gt_uv)
        exp_gt_uv = np.insert(exp_gt_uv, 1, 1) # Insert a constant one.
        exp_gt_uv = np.expand_dims(exp_gt_uv, (0, 1)) # Expand dimensions for broadcasting.

        # Compute angular differences.
        # Normalize vectors.
        exp_uv_map_norm = np.linalg.norm(exp_uv_map, axis=-1, keepdims = True) + EPS
        exp_gt_uv_norm = np.linalg.norm(exp_gt_uv, axis=-1, keepdims = True) + EPS

        exp_uv_map_unit = exp_uv_map / exp_uv_map_norm
        exp_gt_uv_unit = exp_gt_uv / exp_gt_uv_norm

        # Compute pointwise dot products.
        dot = np.sum(exp_uv_map_unit * exp_gt_uv_unit, axis=-1)

        # Clip to [-1, 1] to keep arccos inputs in range.
        dot = np.clip(dot, -1.0, 1.0)

        # Compute angles in radians.
        angle_rad = np.arccos(dot)

        # Convert to degrees.
        all_cost_map[i, :, :] = angle_rad * 180.0 / np.pi

    return all_cost_map

# ===================== 3. Spectral angular error =====================
def angular_error(vec1, vec2):
    vec1 = F.normalize(vec1, p=2, dim=-1)
    vec2 = F.normalize(vec2, p=2, dim=-1)
    dot = (vec1 * vec2).sum(dim=-1).clamp(-1.0 + EPS, 1.0 - EPS)
    return torch.acos(dot).mean()

# ===================== 4. Ground-truth UV for 21 triplets =====================
def get_21_uv_gt(gt_ms):
    gt_uv = []
    for (c1, c2, c3) in TRIPLETS:
        g1 = gt_ms[c1] + EPS
        g2 = gt_ms[c2]
        g3 = gt_ms[c3] + EPS

        u = np.log(g2 / g1)
        v = np.log(g2 / g3)
        gt_uv.append(np.array([u, v]))
    return np.array(gt_uv)

# # ===================== 6. Loss for a single CCC pyramid =====================
# # f_i is the softmax score map; C_Ti is the cost map.
# def loss_CCC_single(f_i, C_Ti):
#     product = f_i.squeeze(1) * C_Ti
#     loss = product.sum(dim=[1, 2]).mean()
#     return loss

# # 7. Total loss: 21 CCC losses plus the CNN spectral angular loss.
# def ise_total_loss(
#     f_list,        # Score maps from 21 CCC blocks: list of [B, 1, 128, 128].
#     uv_gt_list,    # Ground-truth UV for 21 triplets: list of [B, 2].
#     spd_pred,      # Spectrum predicted by the CNN [B, 36].
#     spd_gt,        # Ground-truth spectrum [B, 36].
#     w1=1.0,
#     w2=30.0,
# ):
#     # 1. Sum the 21 CCC losses, propagating gradients through the CCC modules.
#     loss_CCC = 0.0
#     for i in range(21):
#         # CCC outputs have already passed through softmax.
#         C = build_C_Ti(uv_gt_list[i])
#         loss_CCC += loss_CCC_single(f_list[i], C)

#     # 2. CNN spectral angular loss; gradients reach only the CNN backbone, depending on forward-pass detach.
#     loss_CNN = angular_error(spd_pred, spd_gt)

#     # 3. Total loss; weight decay is applied by the optimizer.
#     total_loss = w1 * loss_CCC + w2 * loss_CNN

#     return total_loss, loss_CCC.item(), loss_CNN.item()
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
if __name__ == '__main__':
    ms_path = '/home/wh/beyondRGB/clb/BLUE_blue/1_light1/NT/MIS.h5'
    oppo_path = '/home/wh/beyondRGB/clb/BLUE_blue/1_light1/NT/oppo.h5'
    samsung_path = '/home/wh/beyondRGB/clb/BLUE_blue/1_light1/NT/samsung.h5'

    # ms_crop_hist, ms_cost_map = process_ms_image_2(img_path, json_path, 256, 128, [-3.2, 3.2])
    # process_image(ms_path, oppo_path, 'oppo')
    process_image(ms_path, samsung_path, 'samsung')
    # tmp = 0
    # show_array(ms_crop_hist[0, 0, :, :], 'image_resize.png')
    # show_array(ms_cost_map[0, :, :], 'image_center.png')
    # show_array_as_gray(ms_cc[:,:,:3], 'image_cc.png')
    # show_array_as_gray(ms_wp[:,:,:3], 'image_wp.png')