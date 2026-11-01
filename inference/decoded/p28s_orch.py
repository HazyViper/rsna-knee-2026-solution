"""p28s: THE SHIPPING VARIANT (kaggle/build_sub18_2w.py applies three named edits to the probe's text below).
p28b: the two-worker orchestration of the budget probe (ADR-091 2-A + the addendum 「wall_2w の束ね方」,
sealed in artifacts/p28b/p28b_1_budget_rule.json).

THIS FILE IS EMBEDDED VERBATIM in the probe kernel (kaggle/build_p28budget.py, sha256 asserted) and imported as
it is by the local verifier (scripts_p28b/verify_p28budget.py).  It holds no model and no pixel code: the
per-study work (`load`, `gpu_out`) comes from the worker text, which is the shipped 5-D kernel's body.

    worker side   run_worker(): the chunk loop of submission 18 (a chunk's decode in a thread pool, then the
                  GPU one study at a time) with the timers of the sealed rule, the atomic chunk files, the
                  progress file, and the parent's control files (a reassignment, the solo switch to 4 / 4
                  threads, the release).
    parent side   run_section(): the parity split, the children (CUDA_VISIBLE_DEVICES), the monitor
                  (heartbeat / deadline / non-zero exit -> SIGKILL of the process group, the dead worker's
                  INCOMPLETE chunks re-run by the survivor after its own), the composition by position, the
                  R-6 sweep and the R-7 guard over the COMPOSED whole, the csv.
    timing        allocate_t() / loop_summary(): t_i = R_c x tau_i / sum tau + g_i (the human's decision
                  2026-09-24, 按分（スレッド時間比）), so sum t = the loop wall by construction.

Verification hooks (a hang, a slow worker, short heartbeat / limit values, children one after the other on one
GPU, a re-score across the thread switch) are read from the worker / section configuration and are OFF unless
the local verifier sets them.  The probe kernel never sets them.
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
PRIOR = {"ACL": 0.4099, "MCL": 0.3122, "Medial Meniscus": 0.507, "Lateral Meniscus": 0.3533, "Medial OA": 0.2974,
         "Lateral OA": 0.2056, "PF OA": 0.3465, "Effusion": 0.4191, "Synovitis": 0.2575, "Baker's": 0.2605,
         "Contusion": 0.2388, "Fracture": 0.1264}                       # submission 18's notebook, verbatim
R7_E_ABSENT_MAX, R7_PRIOR_ROW_MAX = 0.05, 0.05                          # ADR-088 addendum R-7; submission 33 (ADR-096 2-C): the drop form's PRIOR-row bound 0.02 -> 0.05
N_Q, N_VDIG = 12, 15                                                    # a study's output: 12 labels + 15 ViT digests


# ------------------------------------------------------------------ files (atomic)
def write_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=float)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_chunk(path, ids, q, vdig, meta):
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, q=np.asarray(q, np.float64), vdig=np.asarray(vdig, np.float64),
             ids=np.asarray(ids), meta=np.asarray(json.dumps(meta, default=float)))
    os.replace(tmp, path)


def load_chunk(path):
    z = np.load(path, allow_pickle=False)
    return [str(s) for s in z["ids"]], z["q"], z["vdig"], json.loads(str(z["meta"]))


def chunk_keys(ids, chunk, wid):
    return [{"key": f"w{wid}c{k // chunk:03d}", "ids": list(ids[k:k + chunk])} for k in range(0, len(ids), chunk)]


# ------------------------------------------------------------------ timing (the sealed rule's definitions)
def allocate_t(records):
    """records: the chunks of ONE loop in order, each {T_start, T_end, ids, tau, g}.  Returns (ids, t, tau+g sums).
    t_i = R_c x tau_i / sum_{j in c} tau_j + g_i with R_c = (T_end - T_start) - sum g  (equal shares if sum tau == 0)."""
    ids, t, s = [], [], []
    for r in records:
        tau, g = np.asarray(r["tau"], float), np.asarray(r["g"], float)
        R_c = (r["T_end"] - r["T_start"]) - g.sum()
        share = tau / tau.sum() if tau.sum() > 0 else np.full(len(tau), 1.0 / max(len(tau), 1))
        t += list(R_c * share + g)
        s += list(tau + g)
        ids += list(r["ids"])
    return ids, np.asarray(t, float), np.asarray(s, float)


def loop_summary(records):
    """The per-loop quantities: W (sum of chunk walls = T_C - T_0 when contiguous), idle (gaps between chunks, 0 in a
    normal run), t / its mean / p90 / rho, sum t - W, the beside per-study sums and D_c / bookkeeping."""
    ids, t, s = allocate_t(records)
    W = float(sum(r["T_end"] - r["T_start"] for r in records))
    span = float(records[-1]["T_end"] - records[0]["T_start"]) if records else 0.0
    out = {"n": len(t), "W_s": W, "span_T0_to_TC_s": span, "idle_s": span - W,
           "sum_t_minus_W_s": float(t.sum() - W) if len(t) else 0.0,
           "t_mean_s": float(t.mean()) if len(t) else None, "t_p90_s": float(np.percentile(t, 90)) if len(t) else None,
           "rho": float(np.percentile(t, 90) / t.mean()) if len(t) else None,
           "beside_per_study_sum_tau_plus_g": {"mean_s": float(s.mean()) if len(s) else None,
                                               "p90_s": float(np.percentile(s, 90)) if len(s) else None},
           "beside_chunks": [{"key": r["key"], "n": len(r["ids"]), "wall_s": r["T_end"] - r["T_start"], "D_c_pool_map_s": r["D_c"],
                              "R_c_s": (r["T_end"] - r["T_start"]) - float(np.sum(r["g"])),
                              "bookkeeping_s": (r["T_end"] - r["T_start"]) - float(np.sum(r["g"])) - r["D_c"],
                              "threads": r["threads"]} for r in records],
           "ids": ids, "t_s": [float(x) for x in t]}
    return out


def p90_estimate(W, n, t):
    """(W / n) x p90(t) / mean(t) -- the ADR-076 2-D-1 kind (mean x the same sample's p90/mean)."""
    t = np.asarray(t, float)
    return (W / n) * float(np.percentile(t, 90)) / float(t.mean())


# ------------------------------------------------------------------ worker side
def run_worker(cfg, load, gpu_out, set_torch_threads, sync):
    """The chunk loop of ONE worker.  cfg: {wid, run_dir, chunks: [{key, ids}], threads: [torch_n or None, decode_n],
    hooks: {...}}.  `load(st) -> (st, g)` and `gpu_out(g) -> (q (12,), vdig (15,), info)` are the worker text's.
    Writes <run_dir>/w<wid>/: chunk files, progress.json, done.json."""
    from concurrent.futures import ThreadPoolExecutor
    wid, run_dir = cfg["wid"], cfg["run_dir"]
    wd = f"{run_dir}/w{wid}"
    os.makedirs(wd, exist_ok=True)
    hooks = cfg.get("hooks") or {}
    torch_n, dec_n = cfg["threads"]
    if torch_n is not None:
        set_torch_threads(int(torch_n))
    queue = [dict(c) for c in cfg["chunks"]]
    seen_reassign = 0
    records, ran_keys, rescore = [], [], []
    counters = {"n_ok": 0, "n_e_absent": 0, "n_c_absent": 0, "n_a_absent": 0, "n_xlat_absent": 0, "n_xnotcht_absent": 0, "n_xmed_absent": 0, "n_xmedt_absent": 0, "n_xlatt_absent": 0, "n_xpft_absent": 0, "n_fix_member": 0, "n_fix_prior": 0, "n_load_none": 0}      # p41: + n_c_absent; p47: + n_a_absent; p62: + n_xlat_absent / n_xnotcht_absent; p64: + n_xmed_absent; p66: + the tiles
    first_scored = None
    solo_applied, solo_at_chunk = False, None
    ready = time.time()
    prog = {"wid": wid, "pid": os.getpid(), "ready": ready, "state": "running", "done": [], "last_update": ready}
    write_json(f"{wd}/progress.json", prog)
    T_prev = time.time()
    n_chunks_done = 0
    while True:
        # the parent's control files are read at chunk boundaries only
        ra = read_json(f"{wd}/reassign.json", {"chunks": []})
        if len(ra["chunks"]) > seen_reassign:
            queue += ra["chunks"][seen_reassign:]
            seen_reassign = len(ra["chunks"])
        if not solo_applied and os.path.exists(f"{wd}/solo.flag"):
            set_torch_threads(4)
            torch_n, dec_n = 4, 4
            solo_applied, solo_at_chunk = True, n_chunks_done
            if hooks.get("rescore") and first_scored is not None:
                st, q_before = first_scored
                _, g = load(st)
                q_after, _, _ = gpu_out(g)
                rescore.append({"study": st, "max_abs_diff": float(np.max(np.abs(np.asarray(q_after, float) - q_before))),
                                "before_threads": list(cfg["threads"]), "after_threads": [4, 4]})
            T_prev = time.time()                  # the switch (and a verification re-score) is not a study's time
        if not queue:
            prog.update(state="own_done", last_update=time.time())
            write_json(f"{wd}/progress.json", prog)
            while not queue:
                if os.path.exists(f"{run_dir}/release.flag"):
                    break
                ra = read_json(f"{wd}/reassign.json", {"chunks": []})
                if len(ra["chunks"]) > seen_reassign:
                    queue += ra["chunks"][seen_reassign:]
                    seen_reassign = len(ra["chunks"])
                    break
                time.sleep(0.2)
            if not queue:
                break
            prog.update(state="running", last_update=time.time())
            write_json(f"{wd}/progress.json", prog)
            continue                              # back to the boundary: the solo switch is read there
        if hooks.get("hang_after") is not None and wid == hooks.get("hang_wid", 0) and n_chunks_done == int(hooks["hang_after"]):
            while True:                           # a hung worker: no progress until the parent kills the group
                time.sleep(1.0)
        c = queue.pop(0)
        T_start = T_prev
        tau = {}

        def load_timed(st):
            t0 = time.perf_counter()
            r = load(st)
            tau[st] = time.perf_counter() - t0
            return r

        with ThreadPoolExecutor(int(dec_n)) as ex:
            got = dict(ex.map(load_timed, c["ids"]))
        D_c = time.time() - T_start
        q_all, v_all, g_all = [], [], []
        cnt = {"n_ok": 0, "n_e_absent": 0, "n_c_absent": 0, "n_a_absent": 0, "n_xlat_absent": 0, "n_xnotcht_absent": 0, "n_xmed_absent": 0, "n_xmedt_absent": 0, "n_xlatt_absent": 0, "n_xpft_absent": 0, "n_fix_member": 0, "n_fix_prior": 0, "n_load_none": 0}      # p41: + n_c_absent; p47: + n_a_absent; p62: + n_xlat_absent / n_xnotcht_absent; p64: + n_xmed_absent; p66: + the tiles
        for st in c["ids"]:
            g = got.get(st)
            t1 = time.time()
            if g is None:
                cnt["n_load_none"] += 1
                q, v, info = np.full(N_Q, np.nan), np.full(N_VDIG, np.nan), {"absent": True}
            else:
                q, v, info = gpu_out(g)
                sync()
                cnt["n_ok"] += 0 if info.get("no_member") else 1     # p28s: submission 18 does not score a study with no member
                cnt["n_e_absent"] += int(info.get("e_absent", 0))
                cnt["n_c_absent"] += int(info.get("c_absent", 0))    # p41: the coronal member C absent on this study
                cnt["n_a_absent"] += int(info.get("a_absent", 0))    # p47: the axial member A absent on this study
                cnt["n_xlat_absent"] += int(info.get("xlat_absent", 0))    # p62: the crop member X_lat absent on this study
                cnt["n_xnotcht_absent"] += int(info.get("xnotcht_absent", 0))    # p62: the crop member X_notch_t absent on this study
                cnt["n_xmed_absent"] += int(info.get("xmed_absent", 0))    # p64: the crop member X_med absent on this study
                cnt["n_xmedt_absent"] += int(info.get("xmedt_absent", 0))    # p66: the tile member X_med_t absent on this study
                cnt["n_xlatt_absent"] += int(info.get("xlatt_absent", 0))    # p66: the tile member X_lat_t absent on this study
                cnt["n_xpft_absent"] += int(info.get("xpft_absent", 0))    # p66: the tile member X_pf_t absent on this study
                cnt["n_fix_member"] += int(info.get("fix_member", 0))
                cnt["n_fix_prior"] += int(info.get("fix_prior", 0))
                if first_scored is None:
                    first_scored = (st, np.asarray(q, float))
            g_all.append(time.time() - t1)
            q_all.append(np.asarray(q, float)); v_all.append(np.asarray(v, float))
        if hooks.get("slow_s") and wid == hooks.get("slow_wid", 0):
            time.sleep(float(hooks["slow_s"]))
        del got
        for k in counters:
            counters[k] += cnt[k]
        save_chunk(f"{wd}/{c['key']}.npz", c["ids"], np.stack(q_all), np.stack(v_all), {"key": c["key"], "wid": wid, "counters": cnt,
                                                                                       "threads": [torch_n, dec_n]})
        prog["done"].append(c["key"]); prog["last_update"] = time.time()
        write_json(f"{wd}/progress.json", prog)
        T_end = time.time()
        records.append({"key": c["key"], "ids": list(c["ids"]), "T_start": T_start, "T_end": T_end, "D_c": D_c,
                        "tau": [tau.get(st, 0.0) for st in c["ids"]], "g": g_all, "threads": [torch_n, dec_n]})
        ran_keys.append(c["key"])
        T_prev = T_end
        n_chunks_done += 1
    summ = loop_summary(records)
    done = {"wid": wid, "ready": ready, "records": records, "ran_keys": ran_keys, "counters": counters,
            "summary": {k: v for k, v in summ.items() if k not in ("ids", "t_s")}, "ids": summ["ids"], "t_s": summ["t_s"],
            "solo_applied": solo_applied, "solo_at_chunk": solo_at_chunk, "rescore": rescore,
            "extra": cfg.get("extra_done", {})}
    return done


# ------------------------------------------------------------------ parent side
class CpuSampler(threading.Thread):
    """/proc/stat every second: the busy fraction of all CPUs (user + nice + system + irq + softirq + steal)."""
    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period, self.samples, self.stop_evt = period, [], threading.Event()

    @staticmethod
    def read():
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        idle = v[3] + (v[4] if len(v) > 4 else 0)
        return sum(v), idle

    def run(self):
        try:
            a = self.read()
        except OSError:
            return
        while not self.stop_evt.wait(self.period):
            b = self.read()
            dt, di = b[0] - a[0], b[1] - a[1]
            if dt > 0:
                self.samples.append((time.time(), 1.0 - di / dt))
            a = b

    def stop(self):
        self.stop_evt.set()


def split_parity(studies, swap=False):
    even, odd = list(studies[0::2]), list(studies[1::2])
    return (odd, even) if swap else (even, odd)


def run_section(sec):
    """sec: {run_dir, studies, n_workers (1 or 2), worker_py, cuda_devices [..], threads [torch, decode], chunk,
    swap, serial, T_hb_s, T_limit_s, T_ovh_est_s, p90_fb_placeholder_s, hooks, env, poll_s}.
    Returns the section record (timestamps, per-worker done records, kills, composed arrays, csv path)."""
    run_dir = sec["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    studies = list(sec["studies"])
    POS = {s: k for k, s in enumerate(studies)}
    halves = split_parity(studies, sec.get("swap", False)) if sec["n_workers"] == 2 else (studies,)
    plans = {d: chunk_keys(h, sec["chunk"], d) for d, h in enumerate(halves)}
    T_parent_start = time.time()
    procs, logs, spawn_t, kills, dead = {}, {}, {}, [], set()
    cpu = CpuSampler(); cpu.start()

    def spawn(d):
        cfgp = f"{run_dir}/w{d}_cfg.json"
        os.makedirs(f"{run_dir}/w{d}", exist_ok=True)
        write_json(cfgp, {"wid": d, "run_dir": run_dir, "chunks": plans[d], "threads": sec["threads"], "hooks": sec.get("hooks") or {}})
        env = dict(os.environ, **(sec.get("env") or {}))
        env["CUDA_VISIBLE_DEVICES"] = str(sec["cuda_devices"][d])
        logs[d] = open(f"{run_dir}/w{d}.log", "w")
        procs[d] = subprocess.Popen([sys.executable, sec["worker_py"], cfgp], env=env, stdout=logs[d], stderr=subprocess.STDOUT,
                                    start_new_session=True)
        spawn_t[d] = time.time()

    serial = bool(sec.get("serial")) and sec["n_workers"] == 2
    spawn(0)
    if sec["n_workers"] == 2 and not serial:
        spawn(1)
    all_keys = {c["key"] for d in plans for c in plans[d]}
    n_of = {c["key"]: len(c["ids"]) for d in plans for c in plans[d]}
    owner = {c["key"]: d for d in plans for c in plans[d]}
    deadline_eval = None

    def done_keys():
        out = set()
        for d in plans:
            out |= {k for k in os.listdir(f"{run_dir}/w{d}") if k.endswith(".npz") and ".tmp" not in k} if os.path.isdir(f"{run_dir}/w{d}") else set()
        return {k[:-4] for k in out}

    def kill(d, why, terms=None):
        p = procs.get(d)
        if p is not None and p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
        dead.add(d)
        finished = done_keys()
        incomplete = [c for c in plans[d] if c["key"] not in finished]
        done_by_d = sorted(k for k in finished if owner.get(k) == d)
        kills.append({"wid": d, "why": why, "t": time.time(), "incomplete": [c["key"] for c in incomplete],
                      "completed_before_kill": done_by_d, "terms": terms,
                      "completed_sha256_at_kill": {k: sha256_file(f"{run_dir}/w{d}/{k}.npz") for k in done_by_d}})
        survivors = [x for x in plans if x != d and x not in dead]
        assert survivors, "both workers are dead -- stop"
        s = survivors[0]
        os.makedirs(f"{run_dir}/w{s}", exist_ok=True)
        if s not in procs:                        # serial verification mode: the survivor has not started yet
            plans[s] = plans[s] + incomplete
            write_json(f"{run_dir}/w{s}/reassign.json", {"chunks": []})
            open(f"{run_dir}/w{s}/solo.flag", "w").close()
        else:
            ra = read_json(f"{run_dir}/w{s}/reassign.json", {"chunks": []})
            ra["chunks"] += incomplete
            write_json(f"{run_dir}/w{s}/reassign.json", ra)
            open(f"{run_dir}/w{s}/solo.flag", "w").close()
        for c in incomplete:
            owner[c["key"]] = s

    T_hb = float(sec["T_hb_s"])
    T_limit_abs = T_parent_start + float(sec["T_limit_s"])
    while True:
        now = time.time()
        fin = done_keys()
        if fin >= all_keys:
            break
        for d in list(procs):
            if d in dead:
                continue
            p = procs[d]
            prog = read_json(f"{run_dir}/w{d}/progress.json", {})
            last = prog.get("last_update") or spawn_t[d]
            if p.poll() is not None and p.returncode != 0:
                kill(d, f"non-zero exit {p.returncode}"); continue
            if p.poll() is not None and p.returncode == 0 and not ({c["key"] for c in plans[d]} <= fin):
                kill(d, "exited 0 with chunks left"); continue
            if prog.get("state") != "own_done" and now - last >= T_hb:
                kill(d, "heartbeat", {"T_hb_s": T_hb, "since_last_progress_s": now - last}); continue
            # the deadline (ADR-091 2-A-3): T_dead(t) = T_limit - T_ovh - (n_rem_s + n_rem_d) x p90_fb
            if sec["n_workers"] == 2 and prog.get("state") != "own_done":
                others = [x for x in plans if x != d and x not in dead]
                if others:
                    s = others[0]
                    n_rem_d = sum(n_of[c["key"]] for c in plans[d] if c["key"] not in fin)
                    n_rem_s = sum(n_of[k] for k, o in owner.items() if o == s and k not in fin)
                    T_dead = T_limit_abs - float(sec["T_ovh_est_s"]) - (n_rem_s + n_rem_d) * float(sec["p90_fb_placeholder_s"])
                    if now >= T_dead:
                        deadline_eval = {"t": now, "T_limit_abs": T_limit_abs, "T_ovh_est_s": sec["T_ovh_est_s"],
                                         "n_rem_s": n_rem_s, "n_rem_d": n_rem_d, "p90_fb_s": sec["p90_fb_placeholder_s"],
                                         "T_dead": T_dead,
                                         "T_dead_recomputed": T_limit_abs - float(sec["T_ovh_est_s"]) - (n_rem_s + n_rem_d) * float(sec["p90_fb_placeholder_s"])}
                        kill(d, "deadline", deadline_eval); continue
        if serial and 1 not in procs:
            p0 = read_json(f"{run_dir}/w0/progress.json", {}) or {}
            if 0 in dead or procs[0].poll() is not None or p0.get("state") == "own_done":
                spawn(1)
        time.sleep(float(sec.get("poll_s", 0.5)))
    T_all_chunks = time.time()
    open(f"{run_dir}/release.flag", "w").close()
    rcs = {}
    for d, p in procs.items():
        if d in dead:
            rcs[d] = "killed"; continue
        try:
            rcs[d] = p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL); rcs[d] = "timeout"
    for f in logs.values():
        f.close()
    cpu.stop()
    workers = {d: read_json(f"{run_dir}/w{d}/done.json") for d in plans}

    # ---- the composition by position (submission 18's POS / PRIOR), then R-6 and R-7 over the WHOLE
    t_c0 = time.time()
    pred, vdig, counters, got, sources = compose(run_dir, studies, owner, list(plans), sec.get("pred_dtype", "float64"))
    guard = compose_guard(pred, len(studies), counters)
    csv_path = sec.get("csv")
    if csv_path:
        write_csv(csv_path, studies, pred)
    T_csv = time.time()
    return {"T_parent_start": T_parent_start, "spawn_t": spawn_t, "T_all_chunks": T_all_chunks, "T_compose_start": t_c0,
            "T_csv": T_csv, "workers": workers, "kills": kills, "deadline_eval": deadline_eval, "return_codes": rcs,
            "plans": {d: [c["key"] for c in v] for d, v in plans.items()}, "owner": owner, "every_study_composed": bool(got.all()),
            "composed_from": sources,
            "pred": {lb: pred[lb].tolist() for lb in LABELS}, "vdig": vdig.tolist(), "studies": studies, "guard": guard,
            "counters": counters, "cpu_util": cpu.samples, "csv": csv_path}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def compose(run_dir, studies, owner, wids, pred_dtype="float64"):
    """Every chunk file into submission 18's arrays by position: pred (PRIOR where a study has no output), the ViT
    digests, and the counters summed over the COMPOSED chunk files (exact whatever worker wrote them)."""
    POS = {s: k for k, s in enumerate(studies)}
    pred = {lb: np.full(len(studies), PRIOR[lb], np.dtype(pred_dtype)) for lb in LABELS}   # p28s: submission 18's float32
    counters = {k: 0 for k in ("n_ok", "n_e_absent", "n_c_absent", "n_a_absent", "n_xlat_absent", "n_xnotcht_absent", "n_xmed_absent", "n_xmedt_absent", "n_xlatt_absent", "n_xpft_absent", "n_fix_member", "n_fix_prior", "n_load_none")}   # p41: + n_c_absent; p47: + n_a_absent; p62: + the crop members; p64: + X_med; p66: + the tiles
    vdig = np.full((len(studies), N_VDIG), np.nan)
    got = np.zeros(len(studies), bool)
    sources = {}
    for key in sorted(owner):
        path = f"{run_dir}/w{owner[key]}/{key}.npz"
        if not os.path.exists(path):                           # written by the original owner before its death
            path = next(f"{run_dir}/w{x}/{key}.npz" for x in wids if os.path.exists(f"{run_dir}/w{x}/{key}.npz"))
        sources[key] = path
        ids, q, v, meta = load_chunk(path)
        for k_ in counters:
            counters[k_] += int(meta["counters"][k_])
        for i, st in enumerate(ids):
            k = POS[st]
            got[k] = True
            vdig[k] = v[i]
            if np.isfinite(q[i]).any():
                for j, lb in enumerate(LABELS):
                    pred[lb][k] = q[i][j]
    return pred, vdig, counters, got, sources


def compose_guard(pred, n_studies, counters):
    """Submission 18's R-6 sweep and R-7 guard, applied to the COMPOSED arrays (in place)."""
    n_fix_late = 0
    for lb in LABELS:                                       # R-6: nothing non-finite may reach the csv
        bad = ~np.isfinite(pred[lb])
        if bad.any():
            n_fix_late += int(bad.sum())
            pred[lb][bad] = PRIOR[lb]
    n_prior_rows = n_studies - counters["n_ok"]
    rate_e, rate_prior = counters["n_e_absent"] / max(n_studies, 1), n_prior_rows / max(n_studies, 1)
    assert rate_e <= R7_E_ABSENT_MAX, f"R-7: E absent on {rate_e:.4f} of studies (> {R7_E_ABSENT_MAX})"
    assert rate_prior <= R7_PRIOR_ROW_MAX, f"R-7: {rate_prior:.4f} of rows left at PRIOR (> {R7_PRIOR_ROW_MAX})"
    return {"n_fix_late": n_fix_late, "n_prior_rows": n_prior_rows, "rate_e_absent": rate_e, "rate_prior_rows": rate_prior}


def write_csv(path, studies, pred):
    import pandas as pd
    sub = pd.DataFrame({"StudyInstanceUID": studies})
    for lb in LABELS:
        sub[lb] = pred[lb]
    assert np.isfinite(sub[LABELS].to_numpy()).all(), "a non-finite cell reached the csv"
    sub.to_csv(path, index=False)


def two_worker_reading(sec_rec):
    """The sealed quantities of the two-worker loop: W_loop (both ready -> the last study's end), n_2w, w*, rho, wall_2w,
    p90_w and its worker, T_ovh (both ready - the parent's start + the csv - the last study's end)."""
    ws = {d: w for d, w in sec_rec["workers"].items() if w is not None and w["records"]}
    readys = {d: w["ready"] for d, w in sec_rec["workers"].items() if w is not None}
    T_both_ready = max(readys.values())
    T_last_end = max(w["records"][-1]["T_end"] for w in ws.values())
    n_2w = len(sec_rec["studies"])
    W_loop = T_last_end - T_both_ready
    per = {}
    for d, w in ws.items():
        t = np.asarray(w["t_s"], float)
        per[d] = {"W_w_s": w["summary"]["W_s"], "n": len(t), "t_mean_s": float(t.mean()), "t_p90_s": float(np.percentile(t, 90)),
                  "rho": float(np.percentile(t, 90) / t.mean()), "sum_t_minus_W_s": w["summary"]["sum_t_minus_W_s"],
                  "idle_s": w["summary"]["idle_s"]}
    wstar = max(per, key=lambda d: (per[d]["W_w_s"], per[d]["t_p90_s"]))
    p90w_d = max(per, key=lambda d: per[d]["t_p90_s"])
    rho = per[wstar]["rho"]
    return {"T_both_ready": T_both_ready, "T_last_end": T_last_end, "W_loop_s": W_loop, "n_2w": n_2w, "waited_worker": wstar,
            "rho": rho, "wall_2w_s": (W_loop / n_2w) * rho, "wall_2w_mean_s": W_loop / n_2w,
            "p90_w_s": per[p90w_d]["t_p90_s"], "p90_w_worker": p90w_d, "per_worker": per,
            "T_ovh_s": (T_both_ready - sec_rec["T_parent_start"]) + (sec_rec["T_csv"] - T_last_end),
            "T_ovh_parts_s": {"start_to_both_ready": T_both_ready - sec_rec["T_parent_start"],
                              "last_end_to_csv": sec_rec["T_csv"] - T_last_end}}


def one_worker_reading(w):
    """p90 of one loop in the addendum's kind: (W / n) x p90(t) / mean(t); with sum t = W it is p90(t) itself."""
    t = np.asarray(w["t_s"], float)
    W, n = w["summary"]["W_s"], len(t)
    est = p90_estimate(W, n, t)
    return {"W_s": W, "n": n, "t_mean_s": float(t.mean()), "t_p90_s": float(np.percentile(t, 90)), "rho": float(np.percentile(t, 90) / t.mean()),
            "p90_estimate_s": est, "equals_p90_t_rel": abs(est - float(np.percentile(t, 90))) / float(np.percentile(t, 90)),
            "sum_t_minus_W_s": w["summary"]["sum_t_minus_W_s"], "idle_s": w["summary"]["idle_s"]}


def worker_main(cfg_path, ns):
    """The child's entry: `ns` is the worker text's namespace (load / gpu_out / torch)."""
    cfg = read_json(cfg_path)
    torch = ns["torch"]
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    done = run_worker(cfg, ns["load"], ns["gpu_out"], torch.set_num_threads, sync)
    done["extra"] = {"peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 4) if torch.cuda.is_available() else None,
                     "torch_threads_end": torch.get_num_threads(), "device": ns.get("DEV"),
                     "stop_conditions": [], "t_since_worker_start_s": time.time() - ns.get("t00", time.time())}   # p28s: submission 18's names
    write_json(f"{cfg['run_dir']}/w{cfg['wid']}/done.json", done)
