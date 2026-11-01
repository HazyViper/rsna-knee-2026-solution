# RSNA Knee Abnormality Detection 2026: a solo entry's code and measurements

This repository holds the code, the inference notebook and the measurements from a solo entry to the Kaggle
competition [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) (2026).
The task was to score twelve findings for each knee MRI study (ACL, MCL, medial and lateral meniscus tears, medial,
lateral and patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture), and the metric was the
macro-averaged ROC AUC.

[日本語の要約](README.ja.md) · [Kaggle discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/754735)

## Results

Final rank: **623 / 6,300 (bronze)**, after Kaggle re-ranked the private leaderboard on 2026-10-27 and removed 125
teams.

**Read this before the rank.** Both submissions we selected were built on other people's public notebooks. Our own
models alone reached private 0.939 before the deadline (submission 48). L14 and L34 are Late Submissions, scored after
the deadline. They use labels that another team published after the deadline.

| Submission | What it is | Public | Private |
|---|---|---|---|
| 48 | our own bundle of eleven members, the best own-only submission before the deadline | 0.950 | 0.939 |
| 55 (selected) | the public notebook [goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958), run as is (only paths edited) | 0.958 | 0.942 |
| 57 (selected) | a rank blend of a public notebook built on heliosli's `rsna-knee-compact-r7cnxfsss` (weight 0.7) and our bundle (weight 0.3) | 0.957 | 0.942 |
| L14 (late) | submission 48 with its five whole-knee members retrained on the 21st place team's public labels | 0.962 | 0.954 |
| L34 (late) | L14 with the attention over slice windows sharpened at inference | 0.963 | 0.955 |

The L34 notebook is public: [ttsuyaogawa/late-l34-tau-l14](https://www.kaggle.com/code/ttsuyaogawa/late-l34-tau-l14).
Its weights are in ten public Kaggle Datasets, listed in [inference/README.md](inference/README.md).

## What is in this repository

| Path | What |
|---|---|
| [inference/](inference/) | the L34 notebook (a byte copy of the public one), the sources it embeds (decoded), and what it does to a study |
| [src/eval/metrics.py](src/eval/metrics.py), [tests/](tests/) | the validation metric (within-site concordance, paired bootstrap) and its tests: `python tests/test_metrics.py` (numpy, pandas) |
| [labels/](labels/) | the prompt that turned the radiology reports into soft labels, with one block withheld |

What is not here, and why: [below](#what-is-not-here).

## The approach

**Data.** There are 4,407 training studies, each with sagittal, coronal and axial series in DICOM. Only 58 of them carry
the organisers' labels. Every training study also has a free-text radiology report, in one of nine languages. The
hidden test set has about 1,300 studies and no reports. The public leaderboard scores about 30% of it, and the private
leaderboard scores the other 70%.

**Labels.** An LLM (Claude, `claude-opus-5`) read each report. For every finding it gave a continuous score in [0, 1]
and quoted the text that drove the score ([labels/](labels/)). The members were trained on these soft labels. The 58
organiser-labelled studies were never trained on; we kept them as a check.

**Members.** The members are 2.5D image models, fine-tuned end to end. Three adjacent slices make one RGB window, a stack gives
21 windows, and the model pools the windows with attention.
- Whole-knee members: `coatnet_rmlp_2_rw_384` on a sagittal 80-slice stack and on coronal and axial 40-slice stacks,
  and ConvNeXt-Small on the coronal and axial stacks.
- Region specialists: models on fixed crops and tiles of the sagittal series, each scoring only its own labels
  (meniscus and OA by compartment, ACL from a tile of the notch, PF OA).
- Right knees: their coronal and axial stacks are mirrored, so the medial side always lies on the same side of the
  image.

The decoder and the CoAtNet recipe started from dreaddevelopment's public notebooks ([NOTICE](NOTICE)).

**Combination.** Every member has five fold checkpoints. In each fold, a per-label linear stacker combines the members'
logits; the stacker was fitted on out-of-fold predictions. The probabilities of the five folds are averaged.

**Validation.** We used five fixed folds and scored the out-of-fold predictions (about 4,400 studies) with the
*within-site concordance c* ([src/eval/metrics.py](src/eval/metrics.py)):
- c is the concordance between a model's scores and the report-derived scores.
- It counts only pairs of studies from the same scanner model (51 strata of manufacturer x model) and pools the counts
  over the strata before dividing.
- Pairs tied on the target are dropped, and ties in the prediction count one half, so a constant model scores 0.5.

Early in the competition, the same concordance computed over all pairs put our candidates in a different order from
the leaderboard, with some pairs inverted; the within-site c reproduced the leaderboard's order. After the deadline,
over 18 of our own bundles, it ordered the private scores with Spearman 0.998. That figure uses a correction for
members whose training labels differ from the labels the metric scores against.

**Reading the leaderboard.** One tick (0.001) is not the standard error. For each pair of submissions we estimated the
standard error of their difference by bootstrapping out-of-fold predictions on samples the size of the test split. On
the private side it ranged from 0.34 to 6.5 ticks, depending on the pair. After the deadline we counted a difference only
when it exceeded max(2 ticks, 2 x that pair's standard error). Of the 43 pairs we read that way, 10 went up, 4 went
down, and 29 could not be told apart.

## How our own bundle grew

These are the public and private scores of our own bundles, in the order we built them. They describe the path; they
do not attribute the gains, since most single steps are smaller than the noise of one pair.

| Submission | Change | Public | Private |
|---|---|---|---|
| 21 | DINOv3 ViT members (LoRA) and a CoAtNet member, trained on the first labels (the mean of Gemma 3 27B and Claude) | 0.903 | 0.900 |
| 31 | the CoAtNet member retrained, and the stacker refitted, on Claude-only labels | 0.938 | 0.925 |
| 33 | the ViT members dropped; a second seed of the CoAtNet member | 0.937 | 0.923 |
| 37 | + a coronal member | 0.945 | 0.931 |
| 42 | + an axial member; the second seed dropped | 0.947 | 0.932 |
| 43 | + ConvNeXt-Small twins on the coronal and axial stacks | 0.948 | 0.934 |
| 46 | + crop specialists (lateral compartment, notch) | 0.949 | 0.935 |
| 47 | + a medial crop | 0.949 | 0.936 |
| 48 | + three tiles (medial, lateral, patellofemoral) | 0.950 | 0.939 |
| L14 | the five whole-knee members retrained on drizlo's labels | 0.962 | 0.954 |
| L34 | the attention sharpened at inference (tau = 12 and 32, averaged) | 0.963 | 0.955 |

## What moved the score

1. **The teacher.** Retraining the five whole-knee members on drizlo's public labels lifted every one of them on the
   private board, by 18 to 24 ticks against the same member trained on our report-derived labels. drizlo's labels were
   made by an LLM that read the images; they were released after the deadline under CC BY 4.0. In the bundle the swap
   was worth +15 private and +12 public (L14 against submission 48's cell). There were earlier signs. A single CoAtNet
   member trained on six different report-derived label sets ranged from 0.846 to 0.918 private, and the switch to
   Claude-only labels came with the jump from submission 21 to 31.
2. **Our metric could not see it.** Against the report-derived labels, six of the seven teacher swaps moved the other
   way. A metric computed against one teacher can rank models trained on that teacher, but it cannot judge the teacher;
   only the leaderboard or true labels can.
3. **Slice order and laterality.** Early on, ordering each series' slices by ImagePositionPatient along the slice
   normal, and orienting every knee the same way, gained +0.055 macro AUC on the 58 labelled studies with no change to
   the model (10 seeds, paired, 10 of 10 positive). It is the largest single gain we measured, though on an early CNN
   that took the slices as input channels. The public recipe's decoder that the later members use orders slices the
   same way.
4. **Other planes as members, not as inputs.** Coronal and axial slices did not help as extra input channels of a
   sagittal model; we measured that several ways, with equal-parameter controls. As separate members they did help
   (submissions 37 and 42).

## What did not move it

After the deadline we compared single changes, on one member or on the bundle, with their own controls on the private
board. All of these could not be told apart from their controls, unless the item says otherwise:

- the training recipe: the backbone's learning rate, layer-wise learning-rate decay, a BCE + ranking loss, a
  Transformer across the windows, a query-token head, fine-tuning the whole network on all windows, and an attention
  loss towards boxes from a localiser;
- conditioning on acquisition parameters (FiLM), and site x finding biases at training time. Removing the FiLM
  conditioning at inference only was the one exception: +3 private, just over its cut of 2.68;
- refitting the heads on a frozen backbone. This one was worse (-5 private, cut 3.81);
- post-processing without training: decorrelating the findings, and sharpening the attention. L34 was +1 against L14,
  inside the noise, so its 0.955 is not "better" than L14's 0.954;
- other labels: drizlo's labels averaged with our own image-read labels (made after the deadline: Claude Opus 5.5 read an
  image mosaic and the report of each study, for $48.49 over 4,407 studies), the 1st place team's public labels for
  one member, and drizlo's labels for the crop and tile members as well.

Before the deadline we tested several more changes with within-site c on out-of-fold predictions. None of them
cleared the bar we had set, and each conclusion holds only for the setup it was measured on:
- frozen DINOv3 features with larger backbones or non-linear heads;
- content-dependent pooling over slices on frozen features (attention, LSE, max, top-k);
- more epochs and more regularisation on the early CNNs;
- SWA and translation augmentation for the CoAtNet members;
- a recurrent block across windows, and a video backbone.

## Lessons

1. **The ceiling was the teacher, and we explored teachers too narrowly.** Every label set we built before the deadline
   came from the report text. Twelve of the thirteen top write-ups we read used an LLM that read the images. Our own
   image-read labels cost $48.49 and were made only after the deadline.
2. **The metric and the selection rule were sound.** Within-site c ordered our private scores (Spearman 0.998). The rule
   for the final two was the best public score of each family, then the top two families. On the private board it
   beat two alternatives we had written down and tied with simply taking the top two public scores. The content fell
   short, not the measuring.
3. **Most single changes are below one pair's resolution.** Apart from the teacher swaps, 27 pairs were read after the
   deadline: 1 went up, 1 went down and 25 could not be told apart. Size a test before running it.
4. **The top of a grid is mostly selection bias.** In a one-factor grid of seven cells, nested cross-validation
   assigned +0.0017 of the leader's +0.0020 apparent gain to the selection itself.
5. **Held-out scanners are harder than random folds.** When a whole vendor group was held out, c fell by 0.009 to 0.013
   against random hold-outs of the same size.

## What is not here

- **Training code.** The training scripts of the research repository read study-level files derived from the
  competition data: the labels with their verbatim report quotes, out-of-fold predictions keyed by StudyInstanceUID,
  and tables of DICOM headers. The competition rules (section 4.b) and the RSNA MIRA agreement (clauses 4 and 6) do
  not allow publishing those, and the scripts do not run without them. Their checkpoints and caches were also deleted
  after the competition, so nothing here re-checks the training.
- **Our labels**, for the same reason.
- **Weights.** These are on Kaggle; see [inference/README.md](inference/README.md).

## Licence and terms

- The code in this repository is under the Apache License 2.0 ([LICENSE](LICENSE)). Third-party material is listed in
  [NOTICE](NOTICE).
- This repository contains no competition data: no images, reports, labels, predictions or study identifiers.
- For research and education only. Not for clinical use and not for diagnosis.

## Acknowledgements

Thanks to RSNA and Kaggle for the competition; to dreaddevelopment for the public notebooks that our decoder and recipe
started from; to drizlo (21st place) for publishing their labels under CC BY 4.0; and to goodpjw2008 and heliosli for
the public notebooks that our selected submissions used.
