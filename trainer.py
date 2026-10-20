import torch
from datetime import datetime
import os
import model
from dataset import CustomDataset, CustomValDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import tools
import shutil
import yaml
import argparse
import sys
from loss import MyLoss
from tools import CosineAnnealingWarmupRestarts

rand_seed = 100
tools.set_seed(rand_seed)

class Trainer():
    def __init__(self, hyper):
        self.hyper = hyper
        self.count = 0
        self.model = getattr(model, hyper['MODEL']['NAME'])().to(self.hyper['TRAIN']['DEVICE'])
        self.optimizer = self.define_optimizer()
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=49500,  # Use one cycle spanning training for smooth learning-rate decay.
            cycle_mult=1.0,
            max_lr=self.hyper['TRAIN']['LR'],  #
            min_lr=5e-6,
            warmup_steps=2000,  # Warm up for approximately 12 epochs, or 4% of the total steps.
            gamma=1.0,  # Use a single cycle without restarts.
        )
        self.train_ld = self.define_train_loader()
        self.total_patches = self.dataset_len # Number of training image patches.
        self.val_ld = self.define_validation_loader()
        self.loss = MyLoss(hyper) # Loss function.

        self.log_dir = os.path.join('Result', hyper['DESCRIBE'], datetime.now().strftime("%m%d%H%M"))
        tools.make_dir(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)

        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = 1e10  #

    def train(self):
        device = self.hyper['TRAIN']['DEVICE']
        for epoch in range(self.start_epoch, self.hyper['TRAIN']['EPOCH']):
            epoch_total_loss = 0
            batch_num = len(self.train_ld)

            with tqdm(total=batch_num, leave=False, desc=f'Epoch {epoch + 1}/{self.hyper["TRAIN"]["EPOCH"]}') as pbar:
                for _, batch in enumerate(self.train_ld):
                    ms_patch = batch['ms_patch'].to(device)
                    rgb_patch = batch['rgb_patch'].to(device)
                    gt_spd = batch['gt_spd'].to(device)

                    # Run the forward pass and compute the loss for each image patch.
                    pred_spd = self.model(rgb_patch, ms_patch)
                    loss_sum, loss_mean = self.loss(pred_spd, gt_spd)
                    epoch_total_loss += loss_sum # Accumulate batch loss sums to compute the epoch average.

                    self.optimizer.zero_grad()
                    loss_mean.backward()
                    self.optimizer.step()

                    # ========== Update the learning rate after each batch ==========
                    self.scheduler.step()
                    self.global_step += 1
                    # Log the learning rate.
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('train/lr', current_lr, self.global_step)
                    # Display per-batch metrics.
                    pbar.set_postfix({
                        'total': f'{loss_mean.item():.4f}',
                    })
                    pbar.update(1)
            # Log epoch-level metrics.
            avg_total = epoch_total_loss / self.total_patches # Average error over the epoch.
            self.writer.add_scalar('train/total_loss', avg_total, epoch + 1)

            if (epoch + 1) % self.hyper['TRAIN']['VALIDATION_INTERVAL'] == 0:
                val_total = self.validate()
                self.writer.add_scalar('val/average total loss', val_total, epoch + 1)
                print(f'Val | Average Total Loss: {val_total:.4f} ')
                if val_total < self.best_loss:
                    self.best_loss = val_total
                    self.save(epoch, best=True)

            if (epoch + 1) % self.hyper['TRAIN']['SAVE_INTERVAL'] == 0:
                self.save(epoch)

    def validate(self):
        self.model.eval()
        device = self.hyper['TRAIN']['DEVICE']
        total_loss_sum = 0
        n_val = len(self.val_ld)

        with torch.no_grad():
            with tqdm(total=n_val, desc='validation') as pbar:
                for batch in self.val_ld:
                    ms_patch = batch['ms_patch'].to(device)
                    rgb_patch = batch['rgb_patch'].to(device)
                    gt_spd = batch['gt_spd'].to(device)

                    pred_spd = self.model(rgb_patch, ms_patch)
                    loss_sum, _ = self.loss(pred_spd, gt_spd)
                    total_loss_sum += loss_sum # Accumulate batch loss sums to compute the overall average.
                    pbar.update(1)

        self.model.train()
        avg_total = total_loss_sum / self.vali_len
        return avg_total

    def define_optimizer(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.hyper['TRAIN']['LR'],
            betas=(0.9, 0.999),
            weight_decay= self.hyper['TRAIN']['LOSS']['W3']
        )
        return optimizer

    def define_train_loader(self):
        dataset = CustomDataset(self.hyper)
        self.dataset_len = len(dataset) # Store the number of training samples.
        return DataLoader(dataset, batch_size=self.hyper['TRAIN']['BATCH_SIZE'], shuffle=True,
                          num_workers=self.hyper['TRAIN']['NUM_WORKERS'], pin_memory=True)

    def define_validation_loader(self):
        dataset = CustomValDataset(self.hyper)
        self.vali_len = len(dataset) # Store the number of validation samples.
        return DataLoader(dataset, batch_size=self.hyper['TRAIN']['BATCH_SIZE'], shuffle=False,
                          num_workers=self.hyper['TRAIN']['NUM_WORKERS'], pin_memory=True)

    def save(self, epoch, best=False):
        base = os.path.join(self.log_dir, 'model')
        if best:
            base += '_best'
        else:
            base += '_latest'

        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': epoch,
            'best_error': self.best_loss
        }, base + '.pt')

    def save_hyper(self, hyper_file_path):
        shutil.copy(hyper_file_path, self.log_dir)
        code_path = os.path.join(self.log_dir, 'code')
        os.makedirs(code_path, exist_ok=True)
        files = ['dataset.py', 'model.py', 'trainer.py', 'tools.py']
        for f in files:
            if os.path.exists(f):
                shutil.copy(f, code_path)

if __name__ == '__main__':
    sys.argv = ['trainer.py', '--hyper_parameter_path', 'hyper_parameters.yaml']
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyper_parameter_path', type=str, required=True)
    config = parser.parse_args()

    with open(config.hyper_parameter_path, 'r') as f:
        hyper = yaml.safe_load(f)

    tools.set_seed(hyper['TRAIN']['SEED'])
    trainer = Trainer(hyper)
    trainer.save_hyper('hyper_parameters.yaml')
    trainer.train()