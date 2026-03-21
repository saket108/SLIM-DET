# SLIM-Det

Structured Language-Image Multimodal Detector for aircraft damage detection.

SLIM-Det is a query-based detector that combines image features with structured inspection context: text prompts, zone priors, and numeric damage metrics. The repository also includes an `image_only` baseline so the multimodal path can be compared against the same architecture without metadata.

## Highlights

- Multimodal detector with dedicated text, zone, and metrics encoders
- `image_only` baseline mode for fair ablations
- Default vision backbone: pretrained `ConvNeXtV2-Tiny`
- Query-based decoder with auxiliary losses and severity prediction
- Evaluation reports `AP50`, `AP50-95`, `precision`, `recall`, and `F1`
- Mid-epoch checkpointing and epoch-wise CSV logging

## Model Overview

The full model path is:

```text
image
  -> PretrainedBackbone (ConvNeXtV2-Tiny by default)
  -> CAFPN
  -> visual memory

text prompt -> TextEncoder
zone label  -> ZoneEncoder
4 metrics   -> MetricsNormalizer -> MetricsEncoder
  -> SMFE -> context tokens / context embedding
  -> SGQI -> initial queries

queries + visual memory + context
  -> RWDA Decoder
  -> DetectorHead   -> class logits + boxes
  -> SeverityHead   -> severity score
  -> SCF            -> calibrated final scores
```

`image_only` mode bypasses the text, zone, and metrics branches and replaces them with learned visual-only queries and context.

Core modules:

- [model/slim_det.py](model/slim_det.py)
- [model/smfe.py](model/smfe.py)
- [model/sgqi.py](model/sgqi.py)
- [model/rwda_decoder.py](model/rwda_decoder.py)
- [model/scf.py](model/scf.py)

## Repository Layout

```text
SLIM-DET/
├── configs/
│   └── slim_det.yaml
├── data/
│   ├── loader.py
│   └── prompt_builder.py
├── model/
│   ├── slim_det.py
│   ├── text_encoder.py
│   ├── zone_encoder.py
│   ├── metrics_encoder.py
│   ├── smfe.py
│   ├── cafpn.py
│   ├── sgqi.py
│   ├── rwda_decoder.py
│   ├── detector_head.py
│   ├── severity_head.py
│   ├── scf.py
│   └── ghost_csp_backbone.py
├── training/
│   ├── task_aligned_assigner.py
│   ├── total_loss.py
│   ├── augmentation.py
│   └── trainer.py
├── utils/
│   ├── metrics.py
│   ├── runtime.py
│   └── visualize.py
├── train.py
├── evaluate.py
└── requirements.txt
```

## Dataset Format

This repo expects the aircraft dataset in JSON split files, not YOLO `labels/*.txt` files.

Expected layout:

```text
Aircraft_dataset/
├── Aircraft_train.json
├── Aircraft_val.json
├── Aircraft_test.json
└── images/
    ├── train/
    ├── val/
    └── test/
```

Important:

- The JSON files should contain image entries and annotations.
- Image filenames inside the JSON should match files inside the corresponding split folder.
- `SLIM-Det` currently trains from the JSON annotations even in `image_only` mode.

## Installation

```bash
git clone https://github.com/saket108/SLIM-DET.git
cd SLIM-DET
pip install -r requirements.txt
```

Current minimal dependencies in [requirements.txt](requirements.txt):

- `torch`
- `torchvision`
- `timm`
- `transformers`
- `numpy`
- `opencv-python`
- `Pillow`
- `PyYAML`

## Configuration

Default training configuration lives in [configs/slim_det.yaml](configs/slim_det.yaml).

Current defaults:

- classes: `6`
- hidden dim: `256`
- queries: `90`
- decoder layers: `4`
- image size: `640`
- backbone: `convnextv2_tiny.fcmae_ft_in22k_in1k`
- prompt mode: `full`

CLI arguments override config values.

## Training

### 1. Quick Local Smoke Run

```bash
python train.py \
  --dataset_root /path/to/Aircraft_dataset \
  --epochs 1 \
  --batch 1 \
  --imgsz 128 \
  --workers 0 \
  --no_pretrained_backbone
```

### 2. Image-Only Baseline

Use this first if you want a fair internal baseline before the full multimodal run.

```bash
python train.py \
  --dataset_root /path/to/Aircraft_dataset \
  --train_images /path/to/Aircraft_dataset/images/train \
  --val_images /path/to/Aircraft_dataset/images/val \
  --epochs 300 \
  --batch 8 \
  --imgsz 640 \
  --workers 2 \
  --image_only \
  --no_scf \
  --save_dir runs/slim_det_image_only
```

### 3. Full Multimodal Model

```bash
python train.py \
  --dataset_root /path/to/Aircraft_dataset \
  --train_images /path/to/Aircraft_dataset/images/train \
  --val_images /path/to/Aircraft_dataset/images/val \
  --epochs 300 \
  --batch 4 \
  --imgsz 640 \
  --workers 2 \
  --save_dir runs/slim_det_full
```

### 4. Useful Training Flags

- `--image_only`: disable text/zone/metrics conditioning
- `--multimodal`: explicitly force full model mode
- `--no_scf`: disable severity-conditioned score filtering
- `--save_every_batches N`: overwrite `slim_det_last.pt` mid-epoch
- `--eval_every_epochs N`: run evaluation every `N` epochs during training
- `--eval_max_batches N`: cap validation/evaluation batches for faster debug runs
- `--no_pretrained_backbone`: disable pretrained timm weights

Training outputs:

- `slim_det_last.pt`
- `slim_det_best.pt`
- periodic `slim_det_ep{N}.pt`
- `train_history.csv`

## Evaluation

Evaluate a specific checkpoint:

```bash
python evaluate.py \
  --dataset_root /path/to/Aircraft_dataset \
  --checkpoint runs/slim_det_full/slim_det_best.pt \
  --batch 4 \
  --workers 2
```

Checkpoint aliases are also supported:

```bash
python evaluate.py --checkpoint best
python evaluate.py --checkpoint last
```

Reported metrics:

- `AP50`
- `AP50-95`
- `precision`
- `recall`
- `F1`

## Recommended Experiment Order

If your goal is to compare against a YOLO baseline, use this order:

1. Train and record the external baseline, for example `YOLOv8s`.
2. Train `SLIM-Det image_only`.
3. Train `SLIM-Det full`.
4. Compare all three on the same split and image size.

This gives you:

- an external baseline: `YOLOv8s`
- an internal baseline: `SLIM-Det image_only`
- the full model: `SLIM-Det multimodal`

## Colab Notes

If your dataset is a zip archive in Colab:

```bash
unzip -q /content/Aircraft_dataset.zip -d /content/dataset
find /content/dataset -name "Aircraft_train.json"
```

Then point `--dataset_root` to the folder containing:

- `Aircraft_train.json`
- `Aircraft_val.json`
- `Aircraft_test.json`
- `images/train`
- `images/val`
- `images/test`

Example Colab run:

```python
DATASET_ROOT = "/content/dataset/content/Aircraft_dataset"

!cd /content/SLIM-DET && PYTHONUNBUFFERED=1 python -u train.py \
  --dataset_root "$DATASET_ROOT" \
  --train_json "$DATASET_ROOT/Aircraft_train.json" \
  --val_json "$DATASET_ROOT/Aircraft_val.json" \
  --train_images "$DATASET_ROOT/images/train" \
  --val_images "$DATASET_ROOT/images/val" \
  --epochs 300 \
  --batch 4 \
  --imgsz 640 \
  --workers 0 \
  --image_only \
  --no_scf \
  --save_every_batches 200 \
  --save_dir /content/drive/MyDrive/slim_det_runs/image_only
```

## Notes and Current Scope

- The repo is set up for the aircraft JSON dataset format.
- The evaluator has been corrected to report proper IoU-based detection metrics.
- `image_only` is the right first run if you are comparing against a strong image-only baseline.
- Full multimodal performance should be judged against both `YOLOv8s` and `SLIM-Det image_only`.

## License

Add your project license here if you intend to distribute the repository publicly.
