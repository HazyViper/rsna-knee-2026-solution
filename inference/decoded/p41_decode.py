"""p41 (ADR-097 addendum 1 items 2 / 3; prompts/p41b_coronal_g40.md section 0): the coronal-only 40-slice decoder WITH the left-right
canonicalisation, and the training-side laterality.  New code; nothing here trains or scores.

build_study_canon(sid, ser_records, tsdir, geom, flip, record=None)
    scripts_p26/p26_decode.py build_study COPIED line for line (the series picker, the IPP order, the span formula, the picks, the per-slot
    pooled p2 / p98 window, the 140 mm crop, INTER_AREA to 336, uint8, the mask; module names prefixed DEC.; the one read line
    `arrs.append(read_px(fp)); pss.append(ps)` split in two to take the insertion) with ONE inserted operation: when `flip` is true, each
    slice read is mirrored IN PLANE (columns reversed, `a[:, ::-1]`) right after the read and before anything else -- the place and the
    axis of Phase 10-D's rule (scripts_p6/d5_dense_cache.py:116-117: `if right and plane in ("coronal", "axial"): a = a[:, ::-1]`, applied
    to the native-resolution slice before any resampling).  p26_decode.py is not edited (its sha256 is pinned by the shipped notebooks).
    The pooled percentile window is invariant to the mirror; the centre crop's integer offset (w - cpx) // 2 is not when (w - cpx) is
    odd, so the canonical stack is NOT fliplr of the non-canonical one on those slices (1 native column; counted in section 0).
    flip=False is anchored byte-equal to p26_decode.build_study (scripts_p41/p41_0_cache.py anchor (i')).

right_knee(ids)
    THE TRAINING-SIDE LATERALITY = Phase 10-D's rule by key: artifacts/step33_laterality_v4.parquet `lat_v3` == "R" (d5_dense_cache.py
    build_plan: `right = (r.lat_v3 == "R")`; a missing value is not right = not mirrored).  Section 0 asserts it equals
    artifacts/p10/p10_10_right_knee.npy (the canary record's right-knee mask) on the corpus order.  The shipped notebooks decide laterality
    at inference from the DICOM tags and the geometry (submission 21's worker `load`: the first Laterality / ImageLaterality tag of the
    sagittal series, else median IPP[0] > 0 -> L else R) -- the same arrangement as the ViT coronal members (trained on lat_v3, shipped
    with the tag / geometry rule); section 0 records the agreement on the training corpus.

slot_series(rows, geom)
    build_study's series-selection lines alone (no file is read): the series each slot of `geom` picks, in slot order.
"""
import numpy as np

import p26_decode as DEC

LAT_PARQUET = "/data/rsna_knee_profiling/artifacts/step33_laterality_v4.parquet"


def right_knee(ids):
    """lat_v3 == "R" on `ids` (corpus order); returns (bool array, counts)."""
    import pandas as pd
    lat = pd.read_parquet(LAT_PARQUET, columns=["StudyInstanceUID", "lat_v3"]).set_index("StudyInstanceUID")["lat_v3"]
    v = lat.reindex(np.asarray(ids).astype(str))
    right = (v == "R").to_numpy()
    return right, {"R": int((v == "R").sum()), "L": int((v == "L").sum()), "missing": int(v.isna().sum()), "n": int(len(v))}


def slot_series(rows, geom):
    """The series picked per slot (build_study's selection lines: pick_series_for_slot with the `used` set), None when absent."""
    used, out = set(), []
    for plane, fluid, k in geom.slots:
        r = DEC.pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            out.append(None); continue
        used.add(r["SeriesInstanceUID"])
        out.append(r["SeriesInstanceUID"])
    return out


def build_study_canon(sid, ser_records, tsdir, geom, flip, record=None):
    """p26_decode.build_study (I:304-342) verbatim + the in-plane mirror of every slice read when `flip` (module docstring)."""
    rows = ser_records.get(sid, [])
    total = geom.total
    vol = np.zeros((total, DEC.IMG, DEC.IMG), np.uint8); idx = 0; used = set()
    slots_rec = []
    for plane, fluid, k in geom.slots:
        rec = {"plane": plane, "fluid": fluid, "k": k, "start": idx, "series": None}
        r = DEC.pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; slots_rec.append(rec); continue
        used.add(r["SeriesInstanceUID"])
        rec["series"] = r["SeriesInstanceUID"]
        rec["series_fluid_sensitive"] = int(r.get("Fluid_Sensitive", 0) or 0)
        files, med_ps = DEC.order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
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
                a = DEC.read_px(fp)
                if flip:                                          # p41: Phase 10-D's in-plane mirror (d5_dense_cache.py:116-117)
                    a = np.ascontiguousarray(a[:, ::-1])
                arrs.append(a); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        rec["n_read_failed"] = int(sum(a is None for a in arrs))
        rec["matrix"] = [int(valid[0].shape[0]), int(valid[0].shape[1])] if valid else None
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [DEC.PCT_LO, DEC.PCT_HI])
        else:
            loq, hiq = 0.0, 1.0
        rec["p2_p98"] = [float(loq), float(hiq)]; rec["ps"] = [float(p) for p in pss]; rec["cpx"] = []
        for a, ps in zip(arrs, pss):
            if idx >= total: break
            if a is None: idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw, cpx = DEC.mm_crop_resize(aw, ps if ps > 0 else med_ps)
            rec["cpx"].append(int(cpx))
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        slots_rec.append(rec)
        if idx >= total: break
    mask = (vol.reshape(total, -1).sum(1) > 0).astype(np.uint8)
    if record is not None:
        record["slots"] = slots_rec
        record["n_series_used"] = len(used)
        record["flip"] = bool(flip)
    return vol, mask
