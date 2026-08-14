#!/usr/bin/env python3
"""Chunked regeneration of ORR lambda examples with the unit-fixed pipeline code.

Runs on the device VM (no torch): torch is stubbed only to satisfy module
imports; example building itself is pure python/numpy and uses the canonical
build_examples_from_results from the orchestrator, one record at a time,
resuming from a byte-offset cursor between invocations.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

STATE = Path(sys.argv[1])
SRC = Path(sys.argv[2])
OUT = Path(sys.argv[3])
FAIL = Path(sys.argv[4])
REPO = Path(sys.argv[5])
BUDGET = int(sys.argv[6]) if len(sys.argv) > 6 else 200 * 1024 * 1024

stub = types.ModuleType("torch")
stub.nn = types.ModuleType("torch.nn")
class _StubModule:  # noqa: N801
    pass
stub.nn.Module = _StubModule
stub.Tensor = object
stub.manual_seed = lambda *a, **k: None
sys.modules.setdefault("torch", stub)
sys.modules.setdefault("torch.nn", stub.nn)

sys.path.insert(0, str(REPO / "pipeline"))
spec = importlib.util.spec_from_file_location(
    "orch", REPO / "pipeline" / "run_oncology_retrospective_lambda_training.py"
)
orch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orch)

state = (
    json.loads(STATE.read_text())
    if STATE.exists()
    else {"offset": 0, "n_records": 0, "n_examples": 0, "n_failures": 0, "done": False}
)
if state["done"]:
    print(json.dumps(state))
    sys.exit(0)

mode = "a" if state["offset"] else "w"
processed = 0
with open(SRC, "rb") as fh, open(OUT, mode, encoding="utf-8") as out, open(FAIL, mode, encoding="utf-8") as fail:
    fh.seek(state["offset"])
    while processed < BUDGET:
        line = fh.readline()
        if not line:
            state["done"] = True
            break
        processed += len(line)
        text = line.decode("utf-8").strip()
        if not text:
            continue
        result = json.loads(text)
        state["n_records"] += 1
        exs, fls = orch.build_examples_from_results([result], "ORR", 0.2)
        for e in exs:
            out.write(json.dumps(e, ensure_ascii=False) + "\n")
            state["n_examples"] += 1
        for f in fls:
            fail.write(json.dumps(f, ensure_ascii=False) + "\n")
            state["n_failures"] += 1
    state["offset"] = fh.tell()

STATE.write_text(json.dumps(state))
print(json.dumps(state))
