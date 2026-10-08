# Export the training history stored inside the released weights.
#
#   python scripts/12_training_logs.py
#
# Ultralytics stores the whole per-epoch results table (its results.csv) and the
# training arguments inside every checkpoint it saves, so weights/best.pt and
# weights/crop_classifier.pt carry their own training logs. This writes them out:
#   results/detector_training_log.csv     one row per epoch of the detector
#   results/classifier_training_log.csv   one row per epoch of the crop classifier
#   results/training_summary.json         epochs, training sessions, best epoch,
#                                         patience and whether early stopping fired
#
# Training sessions: the "time" column counts seconds since the current session
# started, so it falls back whenever training was resumed from a checkpoint. Each
# fall starts a new session.
#
# Best epoch: the checkpoint also stores the validation metrics of the epoch it
# was saved from, including Ultralytics' fitness score. The best epoch is the
# row whose metrics equal those.
#
# No dataset, no GPU, a few seconds on a CPU.
#
# Env:
#   WEIGHTS_PATH=weights/best.pt   CLS_WEIGHTS=weights/crop_classifier.pt
#   RESULTS_DIR=results

import os
import csv
import json
import platform
from pathlib import Path

import torch
import ultralytics

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", str(ROOT / "weights" / "best.pt"))
CLS_WEIGHTS = os.environ.get("CLS_WEIGHTS", str(ROOT / "weights" / "crop_classifier.pt"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))


def export(weights, csv_name):
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    table = ckpt["train_results"]
    args = ckpt["train_args"]
    saved = ckpt["train_metrics"]
    cols = list(table)
    n = len(table["epoch"])

    session, sessions = 1, []
    for i in range(n):
        if i and table["time"][i] < table["time"][i - 1]:
            session += 1
        sessions.append(session)

    metric_cols = [c for c in saved if c in table]
    best = [i for i in range(n)
            if all(abs(table[c][i] - saved[c]) < 5e-6 for c in metric_cols)]
    if len(best) != 1:
        raise SystemExit(f"{weights}: {len(best)} epochs match the saved metrics")
    best_epoch = int(table["epoch"][best[0]])
    last_epoch = int(table["epoch"][-1])

    with open(RESULTS_DIR / csv_name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session"] + cols)
        for i in range(n):
            w.writerow([sessions[i]] + [table[c][i] for c in cols])

    resumed = [int(table["epoch"][i]) for i in range(1, n) if sessions[i] != sessions[i - 1]]
    patience = int(args["patience"])
    summary = {
        "weights": Path(weights).relative_to(ROOT).as_posix() if Path(weights).is_relative_to(ROOT) else weights,
        "log": f"results/{csv_name}",
        "task": args.get("task"),
        "epochs_configured": int(args["epochs"]),
        "epochs_trained": last_epoch,
        "imgsz": args.get("imgsz"),
        "batch": args.get("batch"),
        "sessions": session,
        "resumes": session - 1,
        "resumed_at_epochs": resumed,
        "best_epoch": best_epoch,
        "best_fitness": round(float(saved["fitness"]), 5),
        "best_metrics": {c: round(float(saved[c]), 5) for c in metric_cols if c.startswith("metrics/")},
        "patience": patience,
        "early_stopping_triggered": last_epoch < int(args["epochs"]) and last_epoch - best_epoch >= patience,
    }
    print(f"{csv_name}: {n} epochs, {session} sessions (resumed at epochs {resumed}), "
          f"best epoch {best_epoch}, fitness {summary['best_fitness']}, "
          f"early stopping {'triggered' if summary['early_stopping_triggered'] else 'not triggered'}")
    return summary


RESULTS_DIR.mkdir(parents=True, exist_ok=True)
report = {
    "detector": export(WEIGHTS_PATH, "detector_training_log.csv"),
    "classifier": export(CLS_WEIGHTS, "classifier_training_log.csv"),
    "versions": {"python": platform.python_version(), "torch": torch.__version__,
                 "ultralytics": ultralytics.__version__},
}
out = RESULTS_DIR / "training_summary.json"
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
