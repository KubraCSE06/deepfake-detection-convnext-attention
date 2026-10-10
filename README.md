# Deepfake Detection with ConvNeXt-Tiny: An Attention Ablation Study

This project investigates whether adding channel attention, spatial attention, or a combined Feature Fusion Attention Module (FFAM) changes visual deepfake-detection performance relative to a ConvNeXt-Tiny baseline.

## Research question

Do attention mechanisms improve frame-based deepfake detection when all model variants use the same frozen split and video-level evaluation protocol?

## Models compared

1. **ConvNeXt-Tiny baseline** — pretrained backbone without the added attention modules.
2. **Channel Attention** — channel-wise feature recalibration.
3. **Spatial Attention** — spatial feature weighting.
4. **ConvNeXt-Tiny + FFAM** — sequential channel and spatial attention with a residual connection.

## Dataset and evaluation protocol

- Dataset: FaceForensics++ (the dataset itself is not included in this repository).
- Experiment subset: 400 videos (200 real and 200 fake).
- Sampling: 10 frames per video, for 4,000 sampled frame instances.
- Split: 280 training videos, 60 validation videos, and 60 test videos.
- Fake videos belonging to the same manipulation pair are kept in one partition.
- The split is **manipulation-pair-aware, not strictly identity-independent**. An identity may occur across partitions through different manipulation pairs.
- Final metrics are calculated at the **video level** by averaging each video's sampled frame-level fake probabilities.
- Bootstrap confidence intervals use 2,000 video-level resamples.

## Reported test results

The notebook's final comparison reports the following values:

| Model | Accuracy | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny baseline | 83.33% | 82.76% | 0.8178 | 0.8311 |
| Channel Attention | 75.00% | 71.70% | 0.8144 | 0.8080 |
| Spatial Attention | 81.67% | 81.36% | **0.8333** | 0.8336 |
| ConvNeXt-Tiny + FFAM | **83.33%** | **83.33%** | 0.8300 | **0.8366** |

These results do not show that attention improves every metric. FFAM has the highest reported F1-score and PR-AUC, the baseline ties FFAM for accuracy, and Spatial Attention has the highest ROC-AUC. Channel Attention is below the baseline on the listed metrics. Since the test set contains 60 videos, the differences should be interpreted cautiously alongside the bootstrap confidence intervals.

## Repository structure

```text
.
├── notebooks/
│   └── deepfake_detection_convnext_attention.ipynb
├── results/       # Add exported CSV/JSON tables here
├── figures/       # Add exported PNG/PDF figures here
├── README.md
├── requirements.txt
└── .gitignore
```

## Environment and running the notebook

The notebook was developed for a Kaggle environment with GPU support. It expects the FaceForensics++ files to be available from a Kaggle input dataset and includes Kaggle-specific paths. Adjust `DATASET_PATH` and the output paths if running elsewhere.

Install the core Python dependencies with:

```bash
pip install -r requirements.txt
```

Open `notebooks/deepfake_detection_convnext_attention.ipynb` in Jupyter or Kaggle. The full experiment involves frame extraction, face detection, training four models, and evaluation, so it can require substantial compute time. Do not use “Run All” merely to view the saved notebook outputs.

## Artifacts

Exported tables and figures should be placed in `results/` and `figures/`. Model checkpoints can be made available separately if needed. FaceForensics++ source videos, extracted frames, and face crops are intentionally excluded from this repository; follow the dataset's access and usage terms.

## Limitations

- The test set contains 60 videos, limiting precision and generalizability of performance estimates.
- The split does not establish strict identity-independent generalization.
- Results are from a subset of FaceForensics++ and do not establish generalization to other datasets or newer manipulation methods.
- The models use sampled face frames and video-level probability averaging rather than explicit long-range temporal modelling.
- The experiment compares four ConvNeXt-Tiny configurations rather than a comprehensive set of established deepfake-detection architectures.

## Responsible interpretation

This is a research experiment, not a production-ready detector. Report the test protocol and limitations when presenting the scores; do not interpret small differences as definitive proof of model superiority.
