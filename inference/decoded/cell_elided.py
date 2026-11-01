# p83 L34 (kaggle/build_p83_l34.py): L14's cell (bundle v20 + W17, the five whole-knee slots on the 21st place's labels, the D's on T*) with the five whole-knee members' attention sharpened at inference only (3-2), TWO-WORKER form.
# p83 L14 (kaggle/build_p83_l14.py): L0w's cell (bundle v20 + W17) with the five whole-knee slots = their 21st-place-label versions and the D's retrained (p82's five-member swap bundle (items 27-31 in bundle v20)), TWO-WORKER form.
# p83 L0w (kaggle/build_p83_late.py): this cell + W17 (p73 addendum 7's loading heartbeat on the nine members' fold loops; nothing computed differently), TWO-WORKER form.
# Submission 48 (p66, ADR-118 bundle v20 = V20_tiles: submission 47's graph V19_x4 + the tiles Xmedt, Xlatt, Xpft on their own labels, F20's own-label reweighter on T*; kaggle/build_sub48_2w.py) in the TWO-WORKER form (ADR-091 2-A; kaggle/build_sub18_2w.py).
# The scores are submission 18's: its notebook runs unchanged in each worker (named edits: environment hooks for the
# local verifier, a per-process csv name, the loop body as a function).  The parent spreads test.csv's studies over the
# two T4s by parity, watches the workers (heartbeat / deadline from the p28b probe's measured values), re-runs a dead
# worker's incomplete chunks on the survivor, composes by position, applies R-6 / R-7 to the whole and writes
# submission.csv.  One visible card -> one worker (the one-GPU budget, ADR-091 2-D).
import os, sys, glob, json, time, base64, hashlib
T_KERNEL0 = time.time()
ORCH_SRC = base64.b64decode("<elided: decoded/p28s_orch.py>").decode("utf-8")
assert hashlib.sha256(ORCH_SRC.encode("utf-8")).hexdigest() == "a86c1577abdd1c2039f33e46aa19477660616e9772cb2eebfb0da0694f8180d9", "the embedded orchestration differs"
WORKER_SRC = base64.b64decode("<elided: decoded/p28s_worker.py>").decode("utf-8")
assert hashlib.sha256(WORKER_SRC.encode("utf-8")).hexdigest() == "46c6c0fea7c9bf24d56137d182b3e51317fefcfc0b1b71d71dad1212ab45429c", "the embedded worker text differs"
ORCH = {"__name__": "p28s_orch"}
exec(compile(ORCH_SRC, "p28s_orch.py", "exec"), ORCH)
CONST = json.loads(r"""{"CHUNK": 32, "T_limit_s": 32400.0, "threads": {"worker_torch": 2, "worker_decode": 2, "survivor_or_fallback_torch": 4, "survivor_or_fallback_decode": 4}, "p90_fb_s": 17.230292692046536, "p90_w_s": 21.71718363800776, "T_ovh_s": 207.07800793647766, "T_hb_s": 2084.8496292487453, "from": {"CHUNK / T_limit / threads / heartbeat multiplier": "artifacts/p28b/p28b_1_budget_rule.json parsed_from_the_adr", "p90_fb / p90_w / T_ovh": "artifacts/p67/p67_3_probe_F.json measured (submission 48 bundle v20 + R1's own two-worker probe)"}}""")     # read by key at build: the sealed rule's CHUNK / T_limit, the probe's p90_fb / p90_w / T_ovh

def find_root():
    for d in range(1, 6):
        for hit in sorted(glob.glob("/kaggle/input/" + "*/" * d + "test.csv")):
            return os.path.dirname(hit)
    raise FileNotFoundError("test.csv not found")

ROOT = os.environ.get("P28S_ROOT") or find_root()
OUTCSV = "/kaggle/working/submission.csv" if os.path.isdir("/kaggle/working") else "submission.csv"
import pandas as pd
STUDIES = pd.read_csv(f"{ROOT}/test.csv").StudyInstanceUID.tolist()
import torch
ND = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
N_WORKERS = int(os.environ.get("P28S_FORCE_WORKERS") or (2 if ND >= 2 else 1))      # P28S_FORCE_WORKERS: the local verifier only
CUDA = [0, 1] if (N_WORKERS == 2 and ND >= 2) else [0] * N_WORKERS
THREADS = [CONST["threads"]["worker_torch"], CONST["threads"]["worker_decode"]] if N_WORKERS == 2 else \
          [CONST["threads"]["survivor_or_fallback_torch"], CONST["threads"]["survivor_or_fallback_decode"]]
RUN = os.environ.get("P28S_RUN") or "/tmp/p28s_run"
os.makedirs(RUN, exist_ok=True)
WORKER_PY = os.path.join(RUN, "p28s_worker.py")
with open(WORKER_PY, "w") as f:
    f.write(WORKER_SRC)
print(f"ROOT={ROOT}  studies {len(STUDIES)}  visible GPUs {ND} -> {N_WORKERS} worker(s) on cards {CUDA}, threads {THREADS}", flush=True)
SEC = {"run_dir": os.path.join(RUN, "section"), "studies": STUDIES, "n_workers": N_WORKERS, "worker_py": WORKER_PY,
       "cuda_devices": CUDA, "threads": THREADS, "chunk": CONST["CHUNK"], "swap": False, "serial": False,
       "T_hb_s": CONST["T_hb_s"], "T_limit_s": CONST["T_limit_s"] - (time.time() - T_KERNEL0), "T_ovh_est_s": CONST["T_ovh_s"],
       "p90_fb_placeholder_s": CONST["p90_fb_s"], "hooks": {}, "env": {}, "poll_s": 1.0, "csv": OUTCSV, "pred_dtype": "float32"}
try:
    REC = ORCH["run_section"](SEC)
except BaseException:
    for d in range(N_WORKERS):
        lp = os.path.join(SEC["run_dir"], f"w{d}.log")
        if os.path.exists(lp):
            print(f"---- worker {d} log (tail) ----\n" + open(lp).read()[-6000:], flush=True)
    raise
g, c = REC["guard"], REC["counters"]
print(f"R-6/R-7 on the composed whole: E absent {c['n_e_absent']}/{len(STUDIES)} ({g['rate_e_absent']:.4f}), C absent "
      f"{c['n_c_absent']}/{len(STUDIES)}, A absent {c['n_a_absent']}/{len(STUDIES)}, Xlat absent {c['n_xlat_absent']}/{len(STUDIES)}, Xnotcht absent {c['n_xnotcht_absent']}/{len(STUDIES)}, Xmed absent {c['n_xmed_absent']}/{len(STUDIES)}, Xmedt absent {c['n_xmedt_absent']}/{len(STUDIES)}, Xlatt absent {c['n_xlatt_absent']}/{len(STUDIES)}, Xpft absent {c['n_xpft_absent']}/{len(STUDIES)}, rows left at PRIOR "
      f"{g['n_prior_rows']} ({g['rate_prior_rows']:.4f}), non-finite cells repaired: member-mean {c['n_fix_member']}, prior "
      f"{c['n_fix_prior'] + g['n_fix_late']}", flush=True)
print(f"kills {[(k['wid'], k['why']) for k in REC['kills']]}  return codes {REC['return_codes']}  every study composed {REC['every_study_composed']}", flush=True)
sub = pd.read_csv(OUTCSV)
print(f"\nwrote {OUTCSV}  rows {len(sub)}  cols {list(sub.columns)}  total {time.time() - T_KERNEL0:.0f}s", flush=True)
