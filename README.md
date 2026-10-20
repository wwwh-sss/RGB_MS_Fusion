# RGB_MS_Fusion

A PyTorch project for estimating illuminant spectral power distributions (SPDs) by fusing RGB and multispectral images.

The core model, `FusionSPDNet`, takes an RGB image and a 16-channel multispectral image and predicts a 36-dimensional SPD vector for each sample.

This repository includes model implementations, data processing, training and evaluation scripts, and dataset split lists. Raw data, packed HDF5 datasets, and pretrained weights must be prepared separately.

Default input and output shapes:

| Data | Tensor shape |
| --- | --- |
| RGB image | `[B, 3, 256, 256]` |
| Multispectral image | `[B, 16, 256, 256]` |
| Predicted SPD | `[B, 36]` |

`B` is the batch size.

## Repository Structure

```text
RGB_MS_Fusion/
├── model.py                 # FusionSPDNet, feature encoders, and fusion modules
├── dataset.py               # HDF5 loading and training flip augmentation
├── packageData.py           # Preprocess raw data and pack it into HDF5 files
├── trainer.py               # Training, validation, checkpoints, and TensorBoard logs
├── test.py                  # Load the best checkpoint and evaluate Lab / Field subsets
├── loss.py                  # Angular error and other loss functions
├── tools.py                 # Preprocessing, SPD loading, metrics, and LR scheduling
├── H5Image.py               # HDF5 image utilities
├── check.py                 # Dataset path inspection helper
├── hyper_parameters.yaml    # Data, training, model, and test configuration
└── BeyondRGB_Data/
    ├── train_dataset_paper_git.txt
    ├── val_dataset_paper_git.txt
    ├── test_dataset_paper_git.txt
    ├── train.txt
    ├── val.txt
    └── test.txt
```

The main workflow uses `*_dataset_paper_git.txt`. The other three lists retain absolute paths from the original experiment environment and need adjustment before use on another machine.

## Environment Setup

Run project commands from the repository root. On Linux, you can create a dedicated Python environment as follows:

```bash
git clone https://github.com/wwwh-sss/RGB_MS_Fusion.git
cd RGB_MS_Fusion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy h5py PyYAML tqdm opencv-python sympy tensorboard colour-science
```

This dependency list is based on the source imports. Package versions are not pinned, and the full training workflow has not been verified in a fresh environment. For GPU training, install PyTorch and torchvision builds compatible with your local drivers. The current configuration selects `cuda:0`.

## Data Preparation

### 1. Arrange the BeyondRGB data

Set `TRAIN.DATASET_PATH` in `hyper_parameters.yaml` to your dataset root. The default is `../beyondRGB`. Each scene should contain the following files:

```text
beyondRGB/
└── <relative-scene-path>/
    ├── NT/
    │   ├── MIS.h5
    │   ├── oppo.h5
    │   └── samsung.h5
    └── CRI/
        └── CRI.txt
```

Relative scene paths come from the split lists. The current implementation removes timestamps such as `_2023-02-16-12-36-37` from those paths, so the directories on disk must match the processed paths.

The `check_rgb()` helper checks for all three files, `MIS.h5`, `oppo.h5`, and `samsung.h5`, and excludes `outdoor/72`. This filter applies even when only one RGB camera is selected. Ground-truth spectra are read from `CRI/CRI.txt`.

### 2. Configure and pack the data

Review these settings before packing:

| Setting | Purpose / current value |
| --- | --- |
| `TRAIN.DATASET_PATH` | Root directory of the raw dataset |
| `TRAIN.RGB_CAMERA` | `oppo` or `samsung`; currently `oppo` |
| `TRAIN.TRAIN_PACKED_H5` | Path to the packed training HDF5 file |
| `TRAIN.VAL_PACKED_H5` | Path to the packed validation HDF5 file |
| `TRAIN.DEVICE` | Device used by training and the current test script; defaults to `cuda:0` |
| `TRAIN.BATCH_SIZE` | Defaults to `4`; adjust for available GPU memory |
| `TRAIN.NUM_WORKERS` | Defaults to `4` |

Run:

```bash
python packageData.py
```

The script creates:

```text
BeyondRGB_Data/train_packed_samsung.h5
BeyondRGB_Data/val_packed_samsung.h5
```

These output filenames are currently fixed in `packageData.py`. The actual camera is selected by `TRAIN.RGB_CAMERA`, regardless of the filename suffix. If you change the filenames, update the HDF5 paths in the configuration as well. Running the script again overwrites files with the same names.

Each HDF5 file contains three datasets: `rgb`, `msi`, and `spd`. If processing fails for a sample after the first one, the script prints an error and fills that sample's slot with zeros. Review the packing logs and resolve these failures before training.

## Training

Once the packed data is ready, edit `hyper_parameters.yaml` and run:

```bash
python trainer.py
```

The default configuration trains for 300 epochs using Adam and a cosine learning-rate schedule with warmup. Training samples receive synchronized random horizontal and vertical flips; validation samples do not use these augmentations.

Training outputs are saved under:

```text
Result/<DESCRIBE>/<timestamp>/
├── model_best.pt          # Checkpoint with the lowest validation loss
├── model_latest.pt        # Most recently saved checkpoint
├── hyper_parameters.yaml  # Experiment configuration
├── code/                  # Backup of selected training source files
└── events.out.tfevents.*   # TensorBoard logs
```

View the logs with:

```bash
tensorboard --logdir Result
```

The current entry point overwrites `sys.argv` and always reads `hyper_parameters.yaml` from the repository root. To select another configuration through command-line arguments, first update the entry-point logic and configuration backup path in `trainer.py`.

Both training and testing instantiate the model with its default constructor arguments. Changing only `MODEL.MS_CHANNELS` or `MODEL.OUT_CHANNELS` in the YAML file will not change the model dimensions. `TRAIN.LOSS.W3` supplies Adam's weight decay; `W1` and `W2` are not currently used by `MyLoss`.

## Testing and Evaluation

1. Set `MODEL_PATH` at the bottom of `test.py` to an actual training output directory, such as `./BeyondRGB_experiment/<timestamp>`.
2. Ensure that the directory contains `model_best.pt` and `hyper_parameters.yaml`.
3. Check `TRAIN.DATASET_PATH`, `TRAIN.RGB_CAMERA`, `TRAIN.DEVICE`, and `TRAIN.MS_SIZE` in that saved configuration.
4. Run:

```bash
python test.py
```

The test script loads raw scenes listed in `BeyondRGB_Data/test_dataset_paper_git.txt`. It reports angular-error statistics including Mean, Std, Median, Trimean, Best25, Worst25, and the 95th percentile, as well as FPS measured by the script.

## Files and Reproducibility

- `.gitignore` excludes Python caches, editor settings, environment credentials, HDF5 data, model weights, and training output directories.
- Some standalone debugging entry points in `check.py` and `tools.py` contain absolute paths from the original experiment environment. Update them before using those entry points on another machine.
