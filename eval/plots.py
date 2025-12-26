"""Plotting utilities for training/validation analytics."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from statistics import mean, median
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WH_TYPES = ["who", "when", "where", "what", "why", "how"]


def load_train_logs(run_dir: str) -> dict:
    """Load training/validation logs from a run directory.

    The loader is tolerant to multiple possible file formats:
    - ``train_log.csv`` with columns step, train_loss, val_loss, rougeL
    - HuggingFace ``trainer_state.json`` with ``log_history`` entries
    - ``events.json`` containing one JSON object per line similar to log_history
    Returns a dictionary with lists of (step, value) tuples for each metric.
    """

    logs: dict[str, List[tuple[int, float]]] = {
        "train_loss": [],
        "val_loss": [],
        "rougeL": [],
        "exp_name": os.path.basename(run_dir.rstrip("/")),
    }

    csv_path = os.path.join(run_dir, "train_log.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = row.get("step")
                if step is None:
                    continue
                try:
                    step_i = int(float(step))
                except ValueError:
                    continue
                if "train_loss" in row and row["train_loss"]:
                    try:
                        logs["train_loss"].append((step_i, float(row["train_loss"])))
                    except ValueError:
                        pass
                val_key = "val_loss" if "val_loss" in row else "eval_loss"
                if val_key in row and row[val_key]:
                    try:
                        logs["val_loss"].append((step_i, float(row[val_key])))
                    except ValueError:
                        pass
                rouge_key = "rougeL" if "rougeL" in row else "eval_rougeL"
                if rouge_key in row and row[rouge_key]:
                    try:
                        logs["rougeL"].append((step_i, float(row[rouge_key])))
                    except ValueError:
                        pass
        return logs

    state_path = os.path.join(run_dir, "trainer_state.json")
    events_path = os.path.join(run_dir, "events.json")
    log_entries: list[dict] = []

    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            try:
                state = json.load(f)
                log_entries = state.get("log_history", [])
            except json.JSONDecodeError:
                print(f"Warning: failed to parse {state_path}")

    elif os.path.exists(events_path):
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    log_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    for entry in log_entries:
        step = entry.get("step")
        if step is None:
            continue
        try:
            step_i = int(step)
        except (TypeError, ValueError):
            continue
        if "loss" in entry:
            try:
                logs["train_loss"].append((step_i, float(entry["loss"])))
            except (TypeError, ValueError):
                pass
        if "eval_loss" in entry:
            try:
                logs["val_loss"].append((step_i, float(entry["eval_loss"])))
            except (TypeError, ValueError):
                pass
        rouge_key = "eval_rougeL" if "eval_rougeL" in entry else None
        if rouge_key:
            try:
                logs["rougeL"].append((step_i, float(entry[rouge_key])))
            except (TypeError, ValueError):
                pass

    if not any(logs[k] for k in ("train_loss", "val_loss", "rougeL")):
        print(f"Warning: no training logs found in {run_dir}")
    return logs


def load_qg2qa_details(run_dir: str) -> list[dict]:
    details_path = os.path.join(run_dir, "qg2qa_details.jsonl")
    details: list[dict] = []
    if not os.path.exists(details_path):
        print(f"Warning: missing qg2qa_details.jsonl in {run_dir}")
        return details

    with open(details_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                details.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return details


def _prepare_xy(pairs: list[tuple[int, float]]):
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    x, y = zip(*pairs_sorted) if pairs_sorted else ([], [])
    return x, y


def plot_loss_curves(logs: dict, out_path: str) -> None:
    plt.figure()
    exp_name = logs.get("exp_name", "Loss curves")
    train_x, train_y = _prepare_xy(logs.get("train_loss", []))
    val_x, val_y = _prepare_xy(logs.get("val_loss", []))

    if train_x:
        plt.plot(train_x, train_y, label="Train loss")
    if val_x:
        plt.plot(val_x, val_y, label="Val loss")

    plt.title(exp_name)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    if train_x or val_x:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_rouge_curve(logs: dict, out_path: str) -> None:
    plt.figure()
    exp_name = logs.get("exp_name", "Validation ROUGE-L")
    rouge_x, rouge_y = _prepare_xy(logs.get("rougeL", []))

    if rouge_x:
        plt.plot(rouge_x, rouge_y, label="Val ROUGE-L")
    plt.title(exp_name)
    plt.xlabel("Step")
    plt.ylabel("ROUGE-L")
    if rouge_x:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_f1_hist(details: list[dict], lang: str, out_path: str) -> None:
    scores = [d.get("f1") for d in details if d.get("lang") == lang and d.get("f1") is not None]
    plt.figure()
    bins = [i / 20 for i in range(21)]
    plt.hist(scores, bins=bins, edgecolor="black")
    if scores:
        plt.legend([f"mean={mean(scores):.3f}, median={median(scores):.3f}"])
    plt.title(f"QG→QA F1 distribution ({lang})")
    plt.xlabel("F1")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_passrate_bar(per_exp_pass: dict[str, float], out_path: str) -> None:
    plt.figure()
    exp_names = sorted(per_exp_pass.keys())
    values = [per_exp_pass[name] * 100 for name in exp_names]
    plt.bar(exp_names, values)
    plt.title("QG→QA pass-rate")
    plt.ylabel("Pass rate (%)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_qlen_box(per_exp_lengths: dict[str, list[int]], out_path: str) -> None:
    plt.figure()
    exp_names = sorted(per_exp_lengths.keys())
    data = [per_exp_lengths[name] for name in exp_names]
    plt.boxplot(data, labels=exp_names, showfliers=False)
    plt.title("Question length distribution")
    plt.ylabel("Length (tokens)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_wh_heatmap(per_exp_wh: dict[str, dict[str, float]], out_path: str) -> None:
    plt.figure()
    exp_names = sorted(per_exp_wh.keys())
    matrix = [[per_exp_wh[exp].get(wh, 0.0) for wh in WH_TYPES] for exp in exp_names]
    im = plt.imshow(matrix, aspect="auto", vmin=0, vmax=100)
    plt.colorbar(im, label="Percentage")
    plt.yticks(range(len(exp_names)), exp_names)
    plt.xticks(range(len(WH_TYPES)), WH_TYPES)
    plt.title("WH-type distribution (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def append_to_model_compare(figure_dir: str, figure_names: list[str]) -> None:
    md_path = os.path.join("docs", "model_compare.md")
    lines = [
        "\n## Візуалізації навчання та якості\n",
        "Нижче наведені ключові графіки для моніторингу навчання та якості моделей.\n",
    ]
    for name in figure_names:
        rel_path = os.path.join("figures", name)
        short_comment = ""
        if name.startswith("loss_curves"):
            short_comment = "Криві train/val loss показують збіжність та можливе перенавчання."
        elif name.startswith("rougeL_curve"):
            short_comment = "ROUGE-L по кроках/епохах відображає стабілізацію якості."
        elif name.startswith("f1_hist_en"):
            short_comment = "Гістограма F1 (EN) ілюструє розкид якості QG→QA."
        elif name.startswith("f1_hist_ua"):
            short_comment = "Гістограма F1 (UA) демонструє узгодженість якості україномовних питань."
        elif name.startswith("passrate_bar"):
            short_comment = "Pass-rate по експериментах виявляє стабільність генерації валідних питань."
        elif name.startswith("qlen_box"):
            short_comment = "Boxplot довжин питань допомагає відслідковувати стислість або багатослівність."
        elif name.startswith("wh_heatmap"):
            short_comment = "Heatmap WH-типів показує баланс типів питань між моделями."
        lines.append(f"![{name}]({rel_path})\n")
        if short_comment:
            lines.append(f"- {short_comment}\n")
    with open(md_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate training/quality plots.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories with logs")
    parser.add_argument("--out-dir", default=os.path.join("docs", "figures"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs_sorted = sorted(args.runs)

    all_details: list[dict] = []
    per_exp_pass: Dict[str, float] = {}
    per_exp_lengths: Dict[str, list[int]] = {}
    per_exp_wh_counts: Dict[str, Counter] = {}
    generated_figures: list[str] = []

    for run_dir in runs_sorted:
        exp_name = os.path.basename(run_dir.rstrip("/"))
        logs = load_train_logs(run_dir)
        logs["exp_name"] = exp_name
        suffix = f"_{exp_name}" if len(runs_sorted) > 1 else ""

        if logs.get("train_loss") or logs.get("val_loss"):
            loss_path = os.path.join(args.out_dir, f"loss_curves{suffix}.png")
            plot_loss_curves(logs, loss_path)
            generated_figures.append(os.path.basename(loss_path))
        if logs.get("rougeL"):
            rouge_path = os.path.join(args.out_dir, f"rougeL_curve{suffix}.png")
            plot_rouge_curve(logs, rouge_path)
            generated_figures.append(os.path.basename(rouge_path))

        details = load_qg2qa_details(run_dir)
        if not details:
            continue
        all_details.extend(details)

        passes = [1.0 if d.get("passed") else 0.0 for d in details if "passed" in d]
        if passes:
            per_exp_pass[exp_name] = sum(passes) / len(passes)

        lengths = [d.get("question_len") for d in details if isinstance(d.get("question_len"), (int, float))]
        if lengths:
            per_exp_lengths[exp_name] = [int(l) for l in lengths]

        wh_counter: Counter = Counter()
        for d in details:
            wt = d.get("wh_type")
            if wt:
                wh_counter[str(wt).lower()] += 1
        per_exp_wh_counts[exp_name] = wh_counter

    # Aggregated plots
    for lang in ("en", "ua"):
        lang_details = [d for d in all_details if d.get("lang") == lang]
        if lang_details:
            hist_path = os.path.join(args.out_dir, f"f1_hist_{lang}.png")
            plot_f1_hist(lang_details, lang, hist_path)
            generated_figures.append(os.path.basename(hist_path))
        else:
            print(f"Warning: no details for language {lang}, skipping histogram")

    if per_exp_pass:
        pass_path = os.path.join(args.out_dir, "passrate_bar.png")
        plot_passrate_bar(per_exp_pass, pass_path)
        generated_figures.append(os.path.basename(pass_path))
    else:
        print("Warning: no pass-rate data to plot")

    if per_exp_lengths:
        qlen_path = os.path.join(args.out_dir, "qlen_box.png")
        plot_qlen_box(per_exp_lengths, qlen_path)
        generated_figures.append(os.path.basename(qlen_path))
    else:
        print("Warning: no question length data to plot")

    if per_exp_wh_counts:
        per_exp_wh: Dict[str, Dict[str, float]] = {}
        for exp_name, counter in per_exp_wh_counts.items():
            total = sum(counter.values())
            if total == 0:
                continue
            per_exp_wh[exp_name] = {wh: counter.get(wh, 0) * 100 / total for wh in WH_TYPES}
        if per_exp_wh:
            heatmap_path = os.path.join(args.out_dir, "wh_heatmap.png")
            plot_wh_heatmap(per_exp_wh, heatmap_path)
            generated_figures.append(os.path.basename(heatmap_path))
        else:
            print("Warning: no WH-type data to plot")

    if generated_figures:
        append_to_model_compare(args.out_dir, generated_figures)
        print(f"Saved figures: {', '.join(generated_figures)}")
    else:
        print("No figures were generated.")


if __name__ == "__main__":
    main()
