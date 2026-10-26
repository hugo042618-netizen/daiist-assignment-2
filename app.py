"""
Assignment 2 dashboard: hard hat compliance from site camera snapshots.

    uv run python main.py app

Loads the committed weights in weights/ and the numbers train.py saved in
artifacts/. It never trains anything. Every number that moves with the sliders
is recomputed from the saved test-split scores with the same cost code
(costing.py) that train.py used to choose the thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import costing

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "best.pt"


def load_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text())


MODEL = YOLO(str(WEIGHTS))
METRICS = load_json("metrics.json")
OP = load_json("operating_point.json")
STATS = load_json("dataset_stats.json")
TRAIN_CFG = load_json("training_config.json")
SCORES = {s: pd.read_csv(ARTIFACTS / f"scores_{s}.csv") for s in ("val", "test")}
HISTORY = pd.read_csv(ARTIFACTS / "training_history.csv")
SENSITIVITY = pd.read_csv(ARTIFACTS / "sensitivity.csv")
BASE_COSTS = costing.Costs.from_dict(OP["costs"])
IMGSZ = int(TRAIN_CFG["imgsz"])
POSITIVE = STATS["positive_classes"]
NAMES = MODEL.names
POS_IDS = [k for k, v in NAMES.items() if v in POSITIVE]
EXAMPLES = sorted(str(p) for p in (ROOT / "examples").glob("*.jpg"))

OUTCOME_COLOURS = {"PASS": "#1a7f37", "REVIEW": "#b26b00", "FAIL": "#c62828"}


def eur(x: float) -> str:
    return f"EUR {x:,.0f}"


def decide(score: float, t_low: float, t_high: float) -> str:
    t_high = max(t_high, t_low)
    if score >= t_high:
        return "FAIL"
    if score >= t_low:
        return "REVIEW"
    return "PASS"


# --------------------------------------------------------------- inspect ---
OUTCOME_MEANING = {"PASS": "no action", "REVIEW": "the monitoring operator looks at this snapshot",
                   "FAIL": "alert: the supervisor goes out to stop the worker"}


def inspect(image: np.ndarray | None, t_low: float, t_high: float):
    if image is None:
        return None, "Upload a site photo or pick an example."
    # The dataset images are raw (not resized by Roboflow), so predict on the photo as it is.
    # Ultralytics reads numpy arrays as BGR, Gradio hands over RGB.
    result = MODEL.predict(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), conf=0.001, iou=0.7, imgsz=IMGSZ,
                           max_det=300, verbose=False)[0]
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    bare = np.isin(cls, POS_IDS)
    score = float(conf[bare].max()) if bare.any() else 0.0
    outcome = decide(score, t_low, t_high)

    show = conf >= max(t_low, 0.05)  # boxes that matter at the current lower threshold
    drawn = cv2.cvtColor(result[np.flatnonzero(show)].plot(line_width=2), cv2.COLOR_BGR2RGB)

    lines = [
        f"## <span style='color:{OUTCOME_COLOURS[outcome]}'>{outcome}</span>: {OUTCOME_MEANING[outcome]}",
        f"Bare-head score **{score:.3f}** (highest confidence of any {', '.join(POSITIVE)} box).",
        f"Pass below {t_low:.3g}, review from {t_low:.3g} to {max(t_high, t_low):.3g}, fail (alert) at {max(t_high, t_low):.3g} or above.",
        "",
        "| class | confidence |",
        "|---|---|",
    ]
    order = np.argsort(-conf)
    listed = [i for i in order if show[i]][:12]
    lines += [f"| {NAMES[cls[i]]} | {conf[i]:.3f} |" for i in listed] or ["| none above the lower threshold | |"]
    return drawn, "\n".join(lines)


# ------------------------------------------------------- operating point ---
def costs_from_inputs(base_rate, c_fn, c_fp, c_tp, hourly, cases, review_s) -> costing.Costs:
    return costing.Costs.from_dict({
        **BASE_COSTS.to_dict(),
        "base_rate": (base_rate or 0) / 100,
        "fn": c_fn or 0, "fp": c_fp or 0, "tp": c_tp or 0, "hourly_cost": hourly or 0,
        "cases_per_day": int(cases or 0), "review_seconds": review_s or 0,
    })


def confusion_figure(counts: pd.DataFrame, title: str) -> go.Figure:
    z = counts.to_numpy()
    share = z / np.maximum(z.sum(axis=1, keepdims=True), 1)
    text = [[f"{z[i, j]}<br>({share[i, j]:.0%} of row)" for j in range(z.shape[1])] for i in range(z.shape[0])]
    fig = go.Figure(go.Heatmap(z=share, x=list(counts.columns), y=["bare head in snapshot", "all heads helmeted"],
                               text=text, texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=1,
                               showscale=False))
    fig.update_layout(title=title, height=300, margin=dict(l=10, r=10, t=50, b=10),
                      yaxis=dict(autorange="reversed"))
    return fig


def cost_curve_figure(c: costing.Costs, t_low: float, t_high: float, band_cost: float) -> go.Figure:
    fig = go.Figure()
    for split, dash in (("val", "solid"), ("test", "dash")):
        s = SCORES[split]
        sweep = costing.threshold_sweep(s["score"], s["positive"], c)
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["cost_per_1000"], mode="lines",
                                 name=f"{split} split", line=dict(dash=dash)))
    fig.add_hline(y=1000 * c.base_rate * c.fn, line_dash="dot", line_color="grey",
                  annotation_text="do nothing", annotation_position="top left")
    check_all = 1000 * (c.review + c.base_rate * c.tp + (1 - c.base_rate) * c.tn)
    fig.add_hline(y=check_all, line_dash="dash", line_color="green",
                  annotation_text="operator checks every snapshot", annotation_position="bottom left")
    fig.add_hline(y=band_cost, line_dash="dashdot", line_color="darkorange",
                  annotation_text="current thresholds (model + review band), test", annotation_position="top right")
    fig.add_vrect(x0=t_low, x1=max(t_high, t_low), fillcolor="orange", opacity=0.15, line_width=0,
                  annotation_text="current review band", annotation_position="top right")
    fig.add_vline(x=OP["t_single"], line_color="black",
                  annotation_text=f"chosen single threshold {OP['t_single']:.3g}", annotation_position="bottom right")
    fig.update_layout(title="Expected cost per 1,000 snapshots when the model decides alone",
                      xaxis_title="threshold on bare-head score", yaxis_title="EUR per 1,000 snapshots",
                      height=420, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def operating_view(t_low, t_high, base_rate, c_fn, c_fp, c_tp, hourly, cases, review_s):
    c = costs_from_inputs(base_rate, c_fn, c_fp, c_tp, hourly, cases, review_s)
    test = SCORES["test"]
    res = costing.evaluate_band(test["score"], test["positive"], t_low, t_high, c)
    counts = costing.outcome_counts(test["score"], test["positive"], t_low, t_high)
    nothing = 1000 * c.base_rate * c.fn
    kpis = (
        f"### {eur(res['cost_per_1000'])} expected cost per 1,000 snapshots\n"
        f"Do nothing costs {eur(nothing)}. Saving: **{eur(nothing - res['cost_per_1000'])}** per 1,000, "
        f"{eur((nothing - res['cost_per_1000']) * c.cases_per_day / 1000)} a day.\n\n"
        f"| Measure | Value |\n|---|---|\n"
        f"| Share of snapshots sent to the operator | **{res['review_share']:.1%}** |\n"
        f"| Reviews per day ({c.cases_per_day:,} snapshots) | {res['reviews_per_day']:,.0f} |\n"
        f"| Review hours per day ({c.review_seconds:.0f} s each, EUR {c.review:.2f}) | {res['review_hours_per_day']:.1f} |\n"
        f"| Operators needed ({c.reviewer_hours_per_day:g} h shifts) | **{res['reviewers_needed']:.2f}** |\n"
        f"| Violations caught (model or operator) | {res['recall_violation']:.1%} |\n"
        f"| Compliant snapshots that send the supervisor out | {res['false_alert_rate']:.1%} |\n"
    )
    alt = costing.alternatives(test["score"], test["positive"], c, OP["t_single"], t_low, t_high)
    alt["cost_per_1000"] = alt["cost_per_1000"].round(0)
    alt["cost_per_day"] = alt["cost_per_day"].round(0)
    alt["review_share"] = (100 * alt["review_share"]).round(1).astype(str) + "%"
    alt["reviewers_needed"] = alt["reviewers_needed"].round(2)
    alt = alt.rename(columns={"cost_per_1000": "EUR per 1,000 snapshots", "cost_per_day": "EUR per day",
                              "review_share": "sent to operator", "reviewers_needed": "operators needed"})
    cm = confusion_figure(counts, f"Test split: {len(test)} snapshots, decision at {t_low:.3g} / {max(t_high, t_low):.3g}")
    return cm, kpis, alt, cost_curve_figure(c, t_low, t_high, res["cost_per_1000"])


# ----------------------------------------------------------- model tables ---
def baseline_table() -> pd.DataFrame:
    rows = []
    for key, label in (("finetuned", "Finetuned yolo11n"),
                       ("zero_shot_coco", "COCO yolo11n zero-shot (no head class; snapshot scored by its person boxes)"),
                       ("predict_nothing", "Predict nothing (trivial)")):
        m = METRICS[key]
        img = m["image_level_at_op"]
        rows.append({
            "model": label,
            "mAP50": round(m["overall"]["map50"], 3),
            "mAP50-95": round(m["overall"]["map50_95"], 3),
            f"violation recall @ {img['threshold']:.3g}": round(img["recall_violation"], 3),
            f"violation precision @ {img['threshold']:.3g}": round(img["precision_violation"], 3),
            "snapshot ROC-AUC": None if img.get("roc_auc") is None else round(img["roc_auc"], 3),
            f"EUR per 1,000 @ {img['threshold']:.3g}": round(img["cost_per_1000"], 0),
            "EUR per 1,000, its best threshold": round(img.get("best_cost_per_1000_any_threshold", float("nan")), 0),
        })
    return pd.DataFrame(rows)


def per_class_table(key: str = "finetuned") -> pd.DataFrame:
    rows = []
    for name, m in METRICS[key]["per_class"].items():
        rows.append({
            "class": name + (" (positive: bare head)" if name in POSITIVE else ""),
            "test instances": m["instances"],
            "precision": round(m["precision"], 3),
            "recall": round(m["recall"], 3),
            "AP50": round(m["ap50"], 3),
            "AP50-95": round(m["ap50_95"], 3),
            f"precision @ {OP['t_single']:.3g}": round(m.get("precision_at_op", 0.0), 3),
            f"recall @ {OP['t_single']:.3g}": round(m.get("recall_at_op", 0.0), 3),
        })
    return pd.DataFrame(rows)


def breakdown_table() -> pd.DataFrame:
    b = METRICS.get("head_recall_breakdown")
    if not b:
        return pd.DataFrame()
    rows = []
    for group, r25 in b["at_conf_0.25"].items():
        rop = b["at_operating_threshold"][group]
        rows.append({
            "bare heads in": group,
            "labelled bare heads": r25["total"],
            "found @ 0.25": None if r25["recall"] is None else f"{r25['recall']:.0%}",
            f"found @ {OP['t_single']:.3g}": None if rop["recall"] is None else f"{rop['recall']:.0%}",
        })
    return pd.DataFrame(rows)


def detection_cm_figure(which: str) -> go.Figure:
    cm = METRICS["confusion_matrix"]
    labels = cm["labels"]
    z = np.array(cm[which])
    fig = go.Figure(go.Heatmap(z=np.log1p(z), x=labels, y=labels, text=z, texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    fig.update_layout(title="Box confusion matrix on test (rows predicted, columns true, IoU 0.5)",
                      xaxis_title="true", yaxis_title="predicted", height=420,
                      yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def dataset_table() -> pd.DataFrame:
    rows = []
    for split, s in STATS["splits"].items():
        row = {"split": split, "images": s["images"], "with a bare head": s["positive_images"],
               "all helmeted": s["negative_images"], "bare-head share": f"{s['positive_share']:.0%}"}
        row.update({f"{k} boxes": v for k, v in s["instances"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_table() -> pd.DataFrame:
    t = SENSITIVITY.copy()
    for col in ("test_cost_band", "test_cost_model_alone", "cost_do_nothing", "cost_check_everything"):
        if col in t:
            t[col] = t[col].round(0)
    for col in ("review_share", "recall_violation"):
        if col in t:
            t[col] = (100 * t[col]).round(1).astype(str) + "%"
    if "reviewers_needed" in t:
        t["reviewers_needed"] = t["reviewers_needed"].round(2)
    return t.rename(columns={
        "t_single": "single threshold", "t_low": "lower", "t_high": "upper",
        "test_cost_band": "EUR/1,000 model + review", "test_cost_model_alone": "EUR/1,000 model alone",
        "cost_do_nothing": "EUR/1,000 do nothing", "cost_check_everything": "EUR/1,000 check everything",
        "review_share": "sent to operator", "reviewers_needed": "operators", "recall_violation": "violations caught",
    })


def break_even_text() -> str:
    be = OP.get("break_even", {})
    lines = []
    pretty = {
        "base_rate_vs_do_nothing": ("base rate", "doing nothing", lambda v: f"{v:.2%}"),
        "base_rate_vs_check_everything": ("base rate", "an operator checking every snapshot", lambda v: f"{v:.1%}"),
        "fn_vs_check_everything": ("cost of a missed bare head", "an operator checking every snapshot",
                                   lambda v: f"EUR {v:.0f}"),
        "fn_vs_do_nothing": ("cost of a missed bare head", "doing nothing", lambda v: f"EUR {v:.1f}"),
    }
    for key, (what, against, fmt) in pretty.items():
        r = be.get(key)
        if not r:
            continue
        if r["value"] is None:
            verdict = "always" if r["design_cheaper"] == "always" else "never"
            lines.append(f"- Against {against}: the chosen design is {verdict} cheaper over the {what} range searched.")
        else:
            lines.append(f"- Against {against}: the chosen design is cheaper when the {what} is "
                         f"{r['design_cheaper']} **{fmt(r['value'])}**.")
    if not lines:
        return ""
    return ("**Break-even points** (base costs otherwise; thresholds re-chosen on validation each time, "
            "one-operator cap kept, cost on test):\n" + "\n".join(lines))


def history_figure() -> go.Figure:
    fig = go.Figure()
    for col, label in (("metrics/mAP50(B)", "val mAP50"), ("metrics/mAP50-95(B)", "val mAP50-95")):
        if col in HISTORY:
            fig.add_trace(go.Scatter(x=HISTORY["epoch"], y=HISTORY[col], name=label))
    fig.update_layout(title="Validation mAP during finetuning", xaxis_title="epoch", height=320,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ------------------------------------------------------------------ layout ---
with gr.Blocks(title="Hard hat compliance") as demo:
    gr.Markdown(
        "# Hard hat compliance from site camera snapshots\n"
        f"Finetuned yolo11n on Roboflow `{METRICS['dataset']['workspace']}/{METRICS['dataset']['project']}` "
        f"v{METRICS['dataset']['version']}. Positive class: **violation**, a snapshot with at least one bare head "
        f"(a `{'` or `'.join(POSITIVE)}` box). PASS = no action, REVIEW = the monitoring operator looks, "
        "FAIL = alert, the supervisor goes out to stop the worker. "
        "Thresholds were chosen on the validation split; everything below is the held-out test split."
    )
    with gr.Row():
        t_low = gr.Slider(0, 1, value=OP["t_low"], step=0.001, label="Lower threshold: pass below this")
        t_high = gr.Slider(0, 1, value=OP["t_high"], step=0.001, label="Upper threshold: fail (alert) at or above this")
        reset = gr.Button("Reset to chosen thresholds", scale=0)

    with gr.Tab("Operating point and cost"):
        with gr.Accordion("Cost assumptions (defaults are the ones defended in REPORT.md)", open=False):
            with gr.Row():
                base_rate = gr.Number(value=100 * BASE_COSTS.base_rate, label="Base rate: % of snapshots with a bare head")
                c_fn = gr.Number(value=BASE_COSTS.fn, label="Missed bare head (FN), EUR")
                c_fp = gr.Number(value=BASE_COSTS.fp, label="Supervisor sent for nothing (FP), EUR")
                c_tp = gr.Number(value=BASE_COSTS.tp, label="Supervisor stops a real violation (TP), EUR")
            with gr.Row():
                hourly = gr.Number(value=BASE_COSTS.hourly_cost, label="Monitoring operator cost per hour, EUR")
                cases = gr.Number(value=BASE_COSTS.cases_per_day, label="Snapshots per day")
                review_s = gr.Number(value=BASE_COSTS.review_seconds, label="Seconds per review")
        with gr.Row():
            with gr.Column(scale=1):
                kpi_md = gr.Markdown()
            with gr.Column(scale=1):
                cm_plot = gr.Plot(label="Three-outcome confusion matrix")
        alt_table = gr.Dataframe(label="Against the alternatives (test split, current costs)", interactive=False)
        gr.Markdown(break_even_text())
        curve_plot = gr.Plot(label="Cost against threshold")

    with gr.Tab("Inspect a snapshot"):
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Site photo")
            image_out = gr.Image(type="numpy", label="Detections at the current lower threshold")
        decision_md = gr.Markdown()
        if EXAMPLES:
            gr.Examples(examples=EXAMPLES, inputs=image_in, label="Test-split snapshots")

    with gr.Tab("Model results"):
        gr.Markdown("### Finetuned model against the two baselines (same test split, same evaluator)")
        gr.Dataframe(value=baseline_table(), interactive=False)
        gr.Markdown(f"### Per class (test split). Precision and recall at the best-F1 confidence, and at the "
                    f"operating threshold {OP['t_single']:.3g}")
        gr.Dataframe(value=per_class_table(), interactive=False)
        with gr.Row():
            cm_choice = gr.Radio([("confidence 0.25", "at_conf_0.25"),
                                  (f"operating threshold {OP['t_single']:.3g}", "at_operating_threshold")],
                                 value="at_conf_0.25", label="Box confusion matrix: keep boxes at")
        det_cm = gr.Plot(value=detection_cm_figure("at_conf_0.25"))
        cm_choice.change(detection_cm_figure, cm_choice, det_cm)
        gr.Markdown("### Where it fails: share of labelled bare heads found (IoU 0.5), by head size and crowd size")
        gr.Dataframe(value=breakdown_table(), interactive=False)
        if (ARTIFACTS / "failures.jpg").exists():
            gr.Image(value=str(ARTIFACTS / "failures.jpg"), interactive=False,
                     label="Top row: the 6 violation snapshots the model scores lowest. Bottom row: the 6 compliant "
                           "ones it scores highest. Captions say what the chosen thresholds do with each. "
                           "Red = labelled bare head, yellow = predicted bare head (conf 0.25+)")
        gr.Markdown("### Dataset composition")
        gr.Dataframe(value=dataset_table(), interactive=False)
        gr.Plot(value=history_figure())
        gr.Markdown("### Sensitivity: thresholds re-chosen on validation under other assumptions, cost on test")
        gr.Dataframe(value=sensitivity_table(), interactive=False)

    cost_inputs = [t_low, t_high, base_rate, c_fn, c_fp, c_tp, hourly, cases, review_s]
    cost_outputs = [cm_plot, kpi_md, alt_table, curve_plot]
    for control in cost_inputs:
        control.change(operating_view, cost_inputs, cost_outputs)
    for control in (t_low, t_high):
        control.change(inspect, [image_in, t_low, t_high], [image_out, decision_md])
    image_in.change(inspect, [image_in, t_low, t_high], [image_out, decision_md])
    reset.click(lambda: (OP["t_low"], OP["t_high"]), None, [t_low, t_high])
    demo.load(operating_view, cost_inputs, cost_outputs)


if __name__ == "__main__":
    demo.launch()
