# Biohub Cell Tracking

Linked to the Kaggle competition [Biohub: Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/): Detecting and tracking cells through 3D light-sheet microscopy videos over time.

## Pipeline

**1. Preprocessing (GPU).** Each 3D frame is cleaned with a median filter, rolling-ball background subtraction, CLAHE contrast equalization, and gamma correction, implemented with cuCIM/CuPy for speed (`detection/pre_proc.ipynb`).

**2. Training labels.** Ground truth in the competition is sparse, so candidate cell centers were bootstrapped with Difference-of-Gaussians blob detection and then corrected by hand in napari (`detection/annotate.ipynb`). Training patches of 16 × 64 × 64 voxels were cropped around labeled points (`detection/patch_crop.ipynb`).

**3. Detection with a 3D U-Net.** A compact 3D U-Net (1.4M parameters, `detection/unet3d.py`) predicts a heatmap with a Gaussian peak at each cell center. Pooling along z is delayed because patches are only 16 slices deep. Training uses a weighted MSE loss that upweights cell voxels. The model went through four versions (v0 to v3); from v2 onward, each cell's Gaussian is sized from a seeded watershed instead of a fixed width (`detection/unet_train_*.ipynb`). Checkpoints and peak-detection thresholds were selected by sweeping against ground truth plus Cellpose-derived reference centroids (`detection/validate.ipynb`, `detection/test.ipynb`).

**4. Tracking as an optimization problem.** Detected peaks become nodes in a candidate graph, with edges to the nearest candidates in up to four later frames so tracks can survive missed detections. Linking is solved as an integer linear program with [`motile`](https://github.com/funkelab/motile), in overlapping time windows to keep each solve tractable (`tracking_baseline.ipynb`).

## Results

Pooled over four evaluation videos, the final pipeline found **96.8% of ground-truth cells** and **84.2% of ground-truth links**, with an edge Jaccard of **0.655**. Per-video edge Jaccard ranged from 0.57 to 0.81.

Preprocessing mattered most on hard videos: on the noisiest one, detection recall rose from 0.22 on raw volumes to 0.92 after preprocessing.

## What didn't work

- **Precision on dense videos.** Detections far outnumber annotated cells on crowded volumes, and agreement with reference centroids dropped sharply on the densest video.
- **Off-the-shelf segmentation.** Cellpose and StarDist were tried as detectors and set aside; Cellpose masks were kept only as a secondary reference for evaluation.
- **Leaderboard result.** The entry did not achieve a competitive score in the competition.

## Repository layout

```
tracking_baseline.ipynb     inference, candidate graph, ILP tracking, evaluation
detection/
  pre_proc.ipynb            GPU preprocessing
  annotate.ipynb            DoG label bootstrapping and napari correction
  patch_crop.ipynb          training patch extraction
  unet3d.py                 3D U-Net model
  unet_train_1/2/3.ipynb    training iterations v1 to v3
  validate.ipynb            checkpoint and threshold sweep
  test.ipynb                detection evaluation on evaluation videos
  temp.ipynb                cell-size statistics from Cellpose masks
  visualize copy.ipynb      napari visualization of labels and watersheds
references.txt              dataset link and papers consulted
```

## Running it

Download the competition data from Kaggle into `train/` and `test/` next to this folder. Data, trained models, and intermediate outputs are not included (see `.gitignore`).

Main dependencies: `torch`, `numpy`, `scipy`, `scikit-image`, `zarr`, `dask`, `networkx`, `pandas`, `napari`, and `motile`. Preprocessing requires an NVIDIA GPU with `cupy` and `cucim` (Linux).
