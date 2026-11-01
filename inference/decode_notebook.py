"""Write out the sources embedded in late-l34-tau-l14.ipynb, for reading.

The notebook is one code cell.  The cell base64-decodes an orchestrator and a
worker, checks each against a sha256 written in the cell, and runs them; the
worker carries six more embedded texts (the image decoder, the coronal / axial
canonicaliser, three reweighter tables and the crop helpers), checked the same
way.  This script decodes all of them into decoded/ and checks every decoded
text against the sha256 written in the text that embeds it.  Nothing is run.

The *_elided.py files are the cell and the worker with each base64 literal
replaced by the name of the file it decodes to.  They are for reading only.

Usage:
    python inference/decode_notebook.py            # (re)write decoded/
    python inference/decode_notebook.py --check    # compare decoded/ with the notebook
"""
import argparse
import base64
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "late-l34-tau-l14.ipynb")
OUT = os.path.join(HERE, "decoded")

# variable -> file name: the names the notebook compiles the sources under, and
# for the three tables the file names the worker records next to their sha256
IN_CELL = {"ORCH_SRC": "p28s_orch.py", "WORKER_SRC": "p28s_worker.py"}
IN_WORKER = {"DEC_SRC": "p26_decode.py", "CANON_SRC": "p41_decode.py",
             "RW_SRC": "reweight_g1_drz5.json", "RWF_SRC": "reweight_f19_drz5.json",
             "RWG_SRC": "reweight_f20_drz5.json", "XSRC_TXT": "crop_functions.json",
             "ORCH_SRC": "p28s_orch.py"}     # the worker's own copy: must equal the cell's

BLOB = re.compile(r'^(\w+) = base64\.b64decode\("([A-Za-z0-9+/=]+)"\)\.decode\(["\']utf-8["\']\)', re.M)


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unpack(text, names, where):
    """{file name: decoded text} for the variables in `names`, plus `text` with the literals elided."""
    found = {m.group(1): m for m in BLOB.finditer(text)}
    assert set(found) == set(names), f"{where}: embedded {sorted(found)}, expected {sorted(names)}"
    out, elided = {}, text
    for var, fname in names.items():
        decoded = base64.b64decode(found[var].group(2)).decode("utf-8")
        assert sha256(decoded) in text, f"{where}: {var}'s sha256 is not the one {where} asserts"
        assert out.get(fname, decoded) == decoded, f"{where}: {var} differs from the other copy of {fname}"
        out[fname] = decoded
        elided = elided.replace(found[var].group(2), f"<elided: decoded/{fname}>")
    return out, elided


def decode():
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    assert len(nb["cells"]) == 1 and nb["cells"][0]["cell_type"] == "code", "expected one code cell"
    cell = "".join(nb["cells"][0]["source"])
    files, cell_elided = unpack(cell, IN_CELL, "the cell")
    inner, worker_elided = unpack(files["p28s_worker.py"], IN_WORKER, "the worker")
    assert inner["p28s_orch.py"] == files["p28s_orch.py"], "the worker's orchestrator differs from the cell's"
    files.update(inner)
    files["cell_elided.py"] = cell_elided
    files["p28s_worker_elided.py"] = worker_elided
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare decoded/ with the notebook; write nothing")
    a = ap.parse_args()
    files = decode()
    bad = 0
    for fname, text in sorted(files.items()):
        path = os.path.join(OUT, fname)
        if a.check:
            same = os.path.exists(path) and open(path, encoding="utf-8", newline="").read() == text
            bad += not same
            print(f"{'ok  ' if same else 'DIFF'}  {fname}")
        else:
            os.makedirs(OUT, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            print(f"wrote {fname}  {len(text.encode('utf-8')):>7} B  sha256 {sha256(text)[:12]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
