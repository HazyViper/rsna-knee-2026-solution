"""Phase 26 (ADR-087, prompts/p26.md 0-4): the port of the public inference
notebook I's decoder, plus the 2.5D window generation of T / I.

THIS FILE IS WHAT RUNS.  The notebook is data (docs/ext/notebooks/...): every
function here was written by READING it and carries the `file:line` it was
copied from in its docstring; nothing is imported from or executed in the
notebook, and no child process (`subprocess` / `os.system` / `curl` / `wget`)
is ported.  The abbreviations are recipe_cards.md's:

    I = docs/ext/notebooks/dreaddevelopment__knee-mri-twelve-findings-from-a-single-model/
        knee-mri-twelve-findings-from-a-single-model.py           (460 lines, wc -l)
    T = docs/ext/notebooks/dreaddevelopment__knee-mri-training-the-twelve-finding-model/
        knee-mri-training-the-twelve-finding-model.py             (669 lines, wc -l)

The pixel path is I's, unchanged in its arithmetic: series picked per slot by
plane and Fluid_Sensitive preference (I:295-301), slices ordered by the
projection of ImagePositionPatient onto the plane normal (I:264-270, 275),
k slices at linspace over the span (I:319-320), modality LUT + MONOCHROME1
inversion (I:279-284), the selected slices POOLED for a p2 / p98 window
(I:330-331), clip to [0, 1] (I:337), a 140 mm box about the image centre from
PixelSpacing[0] capped at min(h, w) and resized to 336 with cv2.INTER_AREA
(I:93, 286-290, 271), then uint8 (I:339).

TWO GEOMETRIES, ONE FUNCTION (ADR-087 2-A-2 / 2-A-4, p26.md 0-4): the slot
table and the span are arguments.  The TRAINING geometry (44 slices = 12 / 10 /
8 / 6 / 8, span 15-85 %) is the published corpus's description (T:12-16 reads
that corpus; the slot counts and the span are the dataset-metadata text of
dreaddevelopment/knee-raptor-corpus, whose builder is not published -- card T
unknown (3); the span is applied with I's own formula at I:319, which I:316-318
says mirrors the corpus builder's SPAN_LO / SPAN_HI -- marked 推測 for the
training side).  The INFERENCE geometry (64 = 18 / 14 / 12 / 8 / 12, span
6-94 %) is I:96-98 and I:319.

NO LATERALITY CANONICALISATION.  I carries no flip and no order reversal
(recipe card I: the fact; ADR-087 上書き 3: the primary is reproduced without
one; the shipped submissions' canonicalisation is NOT added here).

THE WINDOWS.  Three physically adjacent slices become the three channels
(T:239 / I:216); window centres are drawn from the valid slices only, with both
neighbours valid (T:202-206 / I:198-205); training samples k centres at random
(T:207-211), evaluation and inference take k evenly spaced (T:212-214 /
I:206-207); the (3, 336, 336) tile is bilinearly resized to the backbone's
input and ImageNet-normalised (I:218-224 / T:216-222, 265-266).  I:99's
`K_EVAL = 42` and its comment are kept as they are (the 42 are a linspace
subset of the 62 centres a 64-slice stack holds; recorded, not repaired).
"""
import glob
import os
import random

import numpy as np

# ---------------------------------------------------------------- fixed config (I:91-104)
IMG = 336                    # I:92   the stored resolution
CROP_MM = 140.0              # I:93   the physical box
PS_MISSING_MM = 0.5          # I:271  PixelSpacing absent -> 0.5 mm
PCT_LO, PCT_HI = 2.0, 98.0   # I:331  the pooled intensity window
K_EVAL = 42                  # I:99   "every window position the volume holds, not an evenly spaced subset" -- comment kept verbatim; see module docstring
NORM = "imagenet"            # I:100  (T:360's default is "none"; the public weights' side is I's -- ADR-087 2-A-2 marks it 推測)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)   # I:103
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)    # I:104
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]     # I:101-102 / T:151-152

# ---------------------------------------------------------------- the two geometries
# (plane, fluid preference, k): fluid 1 = prefer Fluid_Sensitive == 1, 0 = prefer == 0, -1 = no preference (I:96-97, 297)
SLOTS_INFER = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]   # I:96-97
SPAN_INFER = (0.06, 0.94)                                                                                            # I:319
SLOTS_TRAIN = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]     # corpus description (12/10/8/6/8); same slot order and preferences as I:96-97 (推測: the builder is not published)
SPAN_TRAIN = (0.15, 0.85)                                                                                            # corpus description "15 to 85 percent"; applied with I:319's formula (推測)
K_TRAIN = 12                 # T:346  windows per study, training (random)
K_EVAL_T = 24                # T:347  windows per study, T's evaluation (even)


class Geometry:
    """A slot table and a span.  `total` = the stack depth (44 or 64)."""
    def __init__(self, name, slots, span):
        self.name, self.slots, self.span = name, list(slots), tuple(span)
        self.total = int(sum(s[2] for s in self.slots))          # I:98  MAXS = sum(s[2] for s in SLOTS)

    def as_dict(self):
        return {"name": self.name, "slots": [list(s) for s in self.slots], "span": list(self.span), "total": self.total}


GEOM_TRAIN = Geometry("train_44", SLOTS_TRAIN, SPAN_TRAIN)
GEOM_INFER = Geometry("infer_64", SLOTS_INFER, SPAN_INFER)
assert GEOM_TRAIN.total == 44 and GEOM_INFER.total == 64


# ---------------------------------------------------------------- the reader (I:255-292 `_make_reader`)
def order_and_meta(sdir):
    """I:259-277.  Every *.dcm of the series directory: header only
    (`stop_before_pixels=True`); pos = ImagePositionPatient . (row_dir x
    col_dir) from ImageOrientationPatient (I:264-268), else InstanceNumber
    (I:270); PixelSpacing[0] else 0.5 (I:271); a header that raises is kept as
    (0.0, f, 0.5) (I:273-274).  Sorted by pos (I:275).  Returns
    ([(file, ps)...], median ps) (I:276-277)."""
    import pydicom
    fs = glob.glob(sdir + "/*.dcm"); recs = []; ps_list = []
    for f in fs:
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None); ps = float(ps[0]) if ps is not None else PS_MISSING_MM
            ps_list.append(ps); recs.append((pos, f, ps))
        except Exception:
            recs.append((0.0, f, PS_MISSING_MM))
    recs.sort(key=lambda x: x[0])
    med_ps = float(np.median(ps_list)) if ps_list else PS_MISSING_MM
    return [(f, ps) for _, f, ps in recs], med_ps


def read_px(f):
    """I:279-284.  pixel_array through apply_modality_lut, float32;
    MONOCHROME1 inverted as max - a."""
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    d = pydicom.dcmread(f)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def mm_crop_resize(a, ps):
    """I:286-290.  cpx = round(140 / max(ps, 1e-3)) capped at min(h, w); a
    centre crop of cpx x cpx (integer-floor offsets); cv2.resize to (IMG, IMG)
    with INTER_AREA.  Returns (image, cpx) -- the cpx is returned beside the
    image for the record only; I returns the image alone."""
    import cv2
    h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
    cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
    a = a[y0:y0 + cpx, x0:x0 + cpx]
    return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA), cpx


# ---------------------------------------------------------------- the series picker (I:295-301)
def pick_series_for_slot(rows, plane, fluid, used):
    """I:295-301.  Candidates = the study's series rows whose Anatomical_Plane
    equals `plane` and whose SeriesInstanceUID is not yet used; if fluid is 0
    or 1, the first candidate with int(Fluid_Sensitive or 0) == fluid wins;
    else the first candidate (csv row order); None when the plane is absent."""
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def series_records(series_csv):
    """I:395-398.  `<split>_series.csv` -> {StudyInstanceUID: [row dicts]} with
    both UID columns as str; Fluid_Sensitive stays the csv's integer."""
    import pandas as pd
    tser = pd.read_csv(series_csv)
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    return {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}


# ---------------------------------------------------------------- the study (I:304-342 `build_study`)
def build_study(sid, ser_records, tsdir, geom, record=None):
    """I:304-342 with the slot table and the span as arguments (`geom`).

    vol = zeros (total, IMG, IMG) uint8 (I:307); for each slot (I:308): pick
    the series (I:309-312; absent -> the k positions stay zero, I:310-311),
    order its files (I:313; none -> skip, I:314-315), n = len(files),
    lo = int(n * span_lo), hi = int(n * span_hi) - 1, hi = max(hi, lo)
    (I:319 with the span as a parameter), picks = round(linspace(lo, hi, k))
    if n > 1 else [0] * k (I:320), read each pick (a failure -> None with the
    median ps, I:322-327), POOL the valid slices for the p2 / p98 window
    (I:328-333), and per slice: clip to [0, 1], 140 mm crop + resize with the
    slice's ps (else the median), x255 -> uint8 (I:334-339).  mask = 1 where
    the slot's slice holds any non-zero pixel (I:341).

    `record` (optional dict) receives the per-slot facts for the reproduction
    gate -- which series, n files, picks, ps, cpx, matrix, failures.  It is
    written beside the pixels and changes nothing in them."""
    rows = ser_records.get(sid, [])
    total = geom.total
    vol = np.zeros((total, IMG, IMG), np.uint8); idx = 0; used = set()
    slots_rec = []
    for plane, fluid, k in geom.slots:
        rec = {"plane": plane, "fluid": fluid, "k": k, "start": idx, "series": None}
        r = pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; slots_rec.append(rec); continue
        used.add(r["SeriesInstanceUID"])
        rec["series"] = r["SeriesInstanceUID"]
        rec["series_fluid_sensitive"] = int(r.get("Fluid_Sensitive", 0) or 0)
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        rec["n_files"] = len(files); rec["median_ps"] = med_ps
        if not files:
            idx += k; slots_rec.append(rec); continue
        n = len(files); lo, hi = int(n * geom.span[0]), int(n * geom.span[1]) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        rec["lo_hi"] = [int(lo), int(hi)]; rec["picks"] = [int(p) for p in picks]
        arrs = []; pss = []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        rec["n_read_failed"] = int(sum(a is None for a in arrs))
        rec["matrix"] = [int(valid[0].shape[0]), int(valid[0].shape[1])] if valid else None
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [PCT_LO, PCT_HI])
        else:
            loq, hiq = 0.0, 1.0
        rec["p2_p98"] = [float(loq), float(hiq)]; rec["ps"] = [float(p) for p in pss]; rec["cpx"] = []
        for a, ps in zip(arrs, pss):
            if idx >= total: break
            if a is None: idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw, cpx = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            rec["cpx"].append(int(cpx))
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        slots_rec.append(rec)
        if idx >= total: break
    mask = (vol.reshape(total, -1).sum(1) > 0).astype(np.uint8)
    if record is not None:
        record["slots"] = slots_rec
        record["n_series_used"] = len(used)
    return vol, mask


# ---------------------------------------------------------------- the windows (I:198-225 / T:202-222, 237-242, 265-266)
def valid_slices(mask, D):
    """T:229-230 / I:199-201.  The valid slice indices; fewer than 3 -> the
    first min(3, D) positions."""
    valid = np.where(np.asarray(mask) > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    return valid


def window_centers(valid, D, count, train, rng=random):
    """T:202-214 (and I:198-207 for train=False).  Centres are the positions
    c in (lo, hi) with both neighbours inside [lo, hi] of the valid range
    (T:204-205); none -> the one clamped midpoint (T:206).  train=True: the
    centres tiled to cover `count`, shuffled, the first `count` taken
    (T:207-211; T shuffles with the module-level `random` seeded by --seed,
    here an injectable `rng` with the same algorithm).  train=False:
    round(linspace(0, len(cs) - 1, count)) into cs (T:212-214 / I:206-207)."""
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    if train:
        reps = count // len(cs) + 1
        pool = (cs * reps)
        rng.shuffle(pool)
        return pool[:count]
    idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
    return [cs[i] for i in idx]


def windows(vol, centers, res, norm=NORM):
    """I:210-225 / T:237-242, 259-266 without the gain augmentation (that is
    the learner's, T:261-264).  For each centre c (clamped to [1, D-2],
    T:238 / I:215): the three slices c-1, c, c+1 as channels, /255 (T:239 /
    I:216), bilinear resize to (res, res) when needed (T:216-222 / I:218-220,
    torch F.interpolate align_corners=False), ImageNet normalisation when norm
    == "imagenet" (T:265-266 / I:223-224).  Returns a torch tensor
    (K, 3, res, res) float32."""
    import torch
    import torch.nn.functional as F
    D = vol.shape[0]
    wins = np.empty((len(centers), 3, res, res), np.float32)
    for j, c in enumerate(centers):
        c = max(1, min(int(c), D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != res or t.shape[-2] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear", align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - torch.from_numpy(IMAGENET_MEAN)) / torch.from_numpy(IMAGENET_STD)
    return x


def eval_windows(vol, mask, k, res, norm=NORM):
    """I:210-225 `eval_windows`: the k evenly spaced windows of a study."""
    D = vol.shape[0]
    cs = window_centers(valid_slices(mask, D), D, k, train=False)
    return windows(vol, cs, res, norm)
