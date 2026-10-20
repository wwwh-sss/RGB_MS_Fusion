import torch
import torch.nn as nn
import torch.nn.functional as F
import math
EPS = 1e-8
class WHLoss(nn.Module):
    def __init__(self, hyper):
        super(WHLoss, self).__init__()
        self.hyper = hyper

    def sam_loss(self , pred, gt):
        # Multispectral angular error loss.
        vec1 = F.normalize(pred, p=2, dim=-1)
        vec2 = F.normalize(gt, p=2, dim=-1)
        dot = (vec1 * vec2).sum(dim=-1).clamp(-1.0 + EPS, 1.0 - EPS)
        return torch.acos(dot).mean()

    def mse_loss(self, pred, gt):
        # Mean squared error (MSE).
        return torch.mean((pred - gt) ** 2)

    def forward(self, pred, gt):
        return F.mse_loss(pred, gt)

    def angular_error(self, pred, gt):
        eps = 1e-6

        # Ensure the input shapes match.
        assert pred.shape == gt.shape, "Pred and GT must have same shape"

        # Compute dot products and vector norms.
        dot_product = torch.sum(pred * gt, dim=1)
        pred_length = torch.norm(pred, p=2, dim=1) + eps  # Avoid division by zero.
        gt_length = torch.norm(gt, p=2, dim=1) + eps

        # Compute cosine similarity and clip it to a safe range.
        cosine_sim = dot_product / (pred_length * gt_length)

        # Keep the cosine values within [-1, 1].
        # Use a slightly narrower interval for numerical stability.
        cosine_sim = torch.clamp(cosine_sim, -1 + eps, 1 - eps)

        # Compute angles in radians.
        angle_rad = torch.acos(cosine_sim)

        # Convert to degrees and compute the mean.
        angle_deg = angle_rad * (180.0 / math.pi)
        angle_mean = torch.mean(angle_deg)
        angle_std = torch.std(angle_deg)
        return angle_mean, angle_std


class MyLoss(nn.Module):
    def __init__(self, hyper):
        super(MyLoss, self).__init__()
        self.hyper = hyper

    def angular_error(self, pred, gt):
        eps = 1e-6

        # Ensure the input shapes match.
        assert pred.shape == gt.shape, "Pred and GT must have same shape"

        # Compute dot products and vector norms.
        dot_product = torch.sum(pred * gt, dim=1)
        pred_length = torch.norm(pred, p=2, dim=1) + eps  # Avoid division by zero.
        gt_length = torch.norm(gt, p=2, dim=1) + eps

        # Compute cosine similarity and clip it to a safe range.
        cosine_sim = dot_product / (pred_length * gt_length)

        # Keep the cosine values within [-1, 1].
        # Use a slightly narrower interval for numerical stability.
        cosine_sim = torch.clamp(cosine_sim, -1 + eps, 1 - eps)

        # Compute angles in radians.
        angle_rad = torch.acos(cosine_sim)

        # Convert each sample angle to degrees.
        angle_deg = angle_rad * (180.0 / math.pi)
        return angle_deg

    def forward(self, pred_spd, gt_spd):
        Lcnn = self.angular_error(pred_spd, gt_spd) # Compute angular errors.
        return torch.sum(Lcnn).item(), torch.mean(Lcnn) # Return the sum and mean of the per-sample losses.

class VanillaGANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(VanillaGANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()  # Numerically stable combined sigmoid and BCE loss.

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss