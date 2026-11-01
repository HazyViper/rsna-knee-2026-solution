# Inference: late submission L34

`late-l34-tau-l14.ipynb` is a byte copy of the public Kaggle notebook
[ttsuyaogawa/late-l34-tau-l14](https://www.kaggle.com/code/ttsuyaogawa/late-l34-tau-l14) (version 1).
It was submitted after the competition closed and scored **public 0.963 / private 0.955**
(a Late Submission, so it is not on the official leaderboard).

## Running it

On Kaggle, open the notebook and use *Copy & Edit*, keep the competition and the ten Datasets below attached,
and set the accelerator to **GPU T4 x2** with internet off. It writes `/kaggle/working/submission.csv`.
With one GPU it starts one worker instead of two, so it takes longer.

If you push it with the Kaggle CLI, write `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json`.
Without it, the default GPU was a P100 (sm_60), on which the image's PyTorch build fails at the first CUDA kernel
even though `torch.cuda.is_available()` returns True.

## The ten Datasets

L34 reads its 55 weight files (11 members x 5 folds) from five of them. The other five hold the in-competition
versions of the whole-knee members (trained on our own labels; submission 48 used them) and the earlier member G80.
The notebook still mounts them, but reads no file from them.

| Dataset | Members | Training labels | Read by L34 |
|---|---|---|---|
| [rsna-knee-p83-ext21x4](https://www.kaggle.com/datasets/ttsuyaogawa/rsna-knee-p83-ext21x4) | M, M_cor, M_ax, M_ax_nx | drizlo's public labels (21st place) | yes |
| [rsna-knee-p83-ext21](https://www.kaggle.com/datasets/ttsuyaogawa/rsna-knee-p83-ext21) | M_cor_nx | drizlo's public labels (21st place) | yes |
| [rsna-knee-sub46-x](https://www.kaggle.com/datasets/ttsuyaogawa/rsna-knee-sub46-x) | X_lat, X_notch_t | our report-derived labels | yes |
| [rsna-knee-sub47-x](https://www.kaggle.com/datasets/ttsuyaogawa/rsna-knee-sub47-x) | X_med | our report-derived labels | yes |
| [rsna-knee-sub48-x](https://www.kaggle.com/datasets/ttsuyaogawa/rsna-knee-sub48-x) | X_med_t, X_lat_t, X_pf_t | our report-derived labels | yes |
| [rsna-knee-sub31-m](https://www.kaggle.com/datasets/ttsuyaogawa/rsna-knee-sub31-m) | M | our report-derived labels | no |
| [rsna-knee-sub37-mcor](https://www.kaggle.com/datasets/ttsuyaogawa/rsna-knee-sub37-mcor) | M_cor | our report-derived labels | no |
| [rsna-knee-sub42-max](https://www.kaggle.com/datasets/ttsuyaogawa/rsna-knee-sub42-max) | M_ax | our report-derived labels | no |
| [rsna-knee-sub43-nx](https://www.kaggle.com/datasets/ttsuyaogawa/rsna-knee-sub43-nx) | M_cor_nx, M_ax_nx | our report-derived labels | no |
| [rsna-knee-sub21-g80](https://www.kaggle.com/datasets/ttsuyaogawa/rsna-knee-sub21-g80) | G80 | our report-derived labels (v1) | no |

## What runs: `decoded/`

The notebook is one code cell. The cell base64-decodes an orchestrator and a worker and runs them, and the worker
decodes six more embedded texts. Each embedded text is checked against a sha256 written next to it.
`python inference/decode_notebook.py` writes all of them to `decoded/` and repeats those checks; `--check` compares
`decoded/` with the notebook without writing anything.

| File | What it is |
|---|---|
| `cell_elided.py` | the notebook's cell, with the two base64 literals elided (for reading) |
| `p28s_orch.py` | the orchestrator: one worker process per GPU, heartbeats, re-running a dead worker's chunks, composing `submission.csv` |
| `p28s_worker.py` | the worker, exactly as the cell writes it to disk |
| `p28s_worker_elided.py` | the same, with its six base64 literals elided (for reading) |
| `p26_decode.py` | DICOM to image stacks, ported from dreaddevelopment's public notebooks (see `NOTICE`) |
| `p41_decode.py` | the same decoder with right knees mirrored in plane (coronal and axial stacks) |
| `reweight_g1_drz5.json`, `reweight_f19_drz5.json`, `reweight_f20_drz5.json` | the three per-label stackers (whole knee; + crops; + tiles) |
| `crop_functions.json` | helpers that place the crop and tile boxes |

The comments in these files refer to the private research repository (phase numbers, ADRs, file paths).
They were written for that repository and are left as they are.

## What happens to one study

1. **Series and slices** (`p26_decode.py`): for each plane, pick a series (plane, then the fluid-sensitive one), order its
   slices by the projection of ImagePositionPatient onto the plane normal, take the slices at even spacing over the span,
   apply the modality LUT (and invert MONOCHROME1), window at the 2nd / 98th percentile pooled over the chosen slices,
   and cut a 140 mm box at the image centre, resized to 336 px.
2. **Laterality**: one decision per study. The first plane (sagittal, coronal, axial) whose series can be ordered gives
   its Laterality / ImageLaterality tag; without a tag, the median IPP x decides (> 0 means a left knee). For right knees
   the coronal and axial stacks are mirrored in plane (`p41_decode.py`), so the medial side is always on the same side
   of the image. M's sagittal stack is not canonicalised; the crop and tile members use the same decision on the
   sagittal series.
3. **Members** (five folds each):
   - whole knee: **M** (CoAtNet, sagittal 80-slice stack), **M_cor** and **M_ax** (CoAtNet, coronal and axial 40-slice
     stacks), **M_cor_nx** and **M_ax_nx** (ConvNeXt-Small on the same two stacks). Each takes 21 windows of three
     adjacent slices at 384 x 384 px and pools them with attention.
   - region specialists, each scoring only its own labels, cut from the sagittal series: **X_lat** (CoAtNet, lateral crop:
     Lateral Meniscus, Lateral OA), **X_med** (CoAtNet, medial crop: Medial Meniscus, Medial OA), **X_notch_t**
     (ConvNeXt-Small, 64 mm tile of the notch: ACL), **X_med_t**, **X_lat_t**, **X_pf_t** (ConvNeXt-Small tiles: the
     medial labels, the lateral labels, PF OA).
4. **L34's only change over L14**: in the five whole-knee members, the attention scores over the windows are multiplied
   by tau = 12 and by tau = 32 before the softmax, and the two probabilities are averaged. Nothing was retrained.
5. **Combination**: in each fold, the members' logits go through a per-label linear stacker, `((x - mu) / sd) . W + b`,
   in which each label sees only the members declared for it. The five folds' sigmoids are then averaged. If a crop or a
   tile is missing, its labels come from the stacker without it. If a whole-knee member is missing (a study with no
   axial series, say), the study gets the equal-weight mean of the whole-knee members present. A non-finite cell gets
   the members' mean, then the label's prior.
