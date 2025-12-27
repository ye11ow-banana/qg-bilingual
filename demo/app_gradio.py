from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

from demo.client import QGClient

HERE = Path(__file__).parent
EXAMPLES_PATH = HERE / "examples.jsonl"


def load_examples() -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    if not EXAMPLES_PATH.exists():
        return examples
    with EXAMPLES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_badge(passed: bool) -> str:
    color = "#16a34a" if passed else "#dc2626"
    label = "PASSED" if passed else "BLOCKED"
    return f"<div style='padding:8px;text-align:center;background:{color};color:white;font-weight:700;border-radius:6px'>{label}</div>"


def format_reasons(reasons: List[str]) -> str:
    if not reasons:
        return "<div style='color:#16a34a;font-weight:600'>No blocking reasons</div>"
    chips = " ".join(
        f"<span style='background:#fee2e2;color:#991b1b;padding:4px 8px;border-radius:999px;font-size:12px'>{r}</span>"
        for r in reasons
    )
    return f"<div style='display:flex;gap:6px;flex-wrap:wrap'>{chips}</div>"


def metrics_to_table(metrics: Dict[str, Any]) -> List[List[Any]]:
    ordered = ["qa_em", "qa_f1", "qa_conf", "nli", "tox_prob", "rougeL", "lex_hits"]
    rows: List[List[Any]] = []
    for key in ordered:
        if key in metrics:
            rows.append([key, metrics.get(key)])
    for key, value in metrics.items():
        if all(key != row[0] for row in rows):
            rows.append([key, value])
    return rows


def normalize_question_display(question: Optional[str], passed: bool, show_unsafe: bool) -> str:
    if not question:
        return "—" if not passed else ""
    cleaned = question.strip()
    if not cleaned.endswith("?"):
        cleaned += "?"
    if passed or show_unsafe:
        return cleaned
    return "—"


def make_app(client: QGClient, dev_mode: bool) -> gr.Blocks:
    examples = load_examples()
    example_names = [ex["name"] for ex in examples]

    with gr.Blocks(title="QG Bilingual Demo", css=".compact-json textarea{font-size:12px}") as demo:
        gr.Markdown("# Safety-aware Question Generation Demo")
        with gr.Row():
            with gr.Column():
                context_in = gr.Textbox(
                    label="Context",
                    placeholder="Paste passage here",
                    lines=8,
                    min_width=600,
                    value="",
                )
                answer_in = gr.Textbox(
                    label="Answer (optional)",
                    placeholder="Provide answer for aware mode",
                    lines=3,
                    value="",
                )
                with gr.Row():
                    lang_in = gr.Dropdown(
                        choices=["en", "ua"],
                        value="en",
                        label="Language",
                    )
                    mode_in = gr.Dropdown(
                        choices=["aware", "agnostic"],
                        value="aware",
                        label="Mode",
                    )
                    wh_in = gr.Dropdown(
                        choices=["auto", "who", "when", "where", "what", "why", "how"],
                        value="auto",
                        label="WH type",
                    )
                with gr.Row():
                    gen_btn = gr.Button("Generate", variant="primary")
                    example_dd = gr.Dropdown(
                        choices=example_names,
                        label="Try example",
                        value=example_names[0] if example_names else None,
                    )
                    load_example_btn = gr.Button("Load example")
            with gr.Column():
                badge_md = gr.HTML("", label="Result")
                question_out = gr.Textbox(
                    label="Question",
                    lines=2,
                    interactive=False,
                    show_copy_button=True,
                )
                reasons_md = gr.HTML(label="Reasons")
                metrics_table = gr.Dataframe(
                    headers=["Metric", "Value"],
                    datatype=["str", "str"],
                    interactive=False,
                    label="Metrics",
                )
                debug_json = gr.JSON(label="Debug", elem_classes=["compact-json"])
        with gr.Row():
            show_adv = gr.Checkbox(label="Show advanced", value=False)
            json_box = gr.Textbox(
                label="Full response JSON",
                lines=6,
                interactive=False,
                show_copy_button=True,
            )
            copy_btn = gr.Button("Copy JSON")
            if dev_mode:
                show_unsafe = gr.Checkbox(label="Show unsafe question (dev)", value=False)
            else:
                show_unsafe = gr.State(False)

        with gr.Accordion("Advanced parameters", visible=False) as adv_section:
            decoding_json = gr.JSON(value=client.config.decoding, label="Decoding config")
            thresholds_json = gr.JSON(value=client.config.thresholds, label="Thresholds")

        def update_adv_visibility(show: bool):
            return gr.Accordion.update(visible=show)

        show_adv.change(update_adv_visibility, inputs=show_adv, outputs=adv_section)

        def apply_example(name: str):
            for ex in examples:
                if ex["name"] == name:
                    return (
                        ex.get("context", ""),
                        ex.get("answer", ""),
                        ex.get("lang", "en"),
                        ex.get("mode", "aware"),
                        ex.get("wh_type", "auto"),
                    )
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        load_example_btn.click(
            apply_example,
            inputs=example_dd,
            outputs=[context_in, answer_in, lang_in, mode_in, wh_in],
        )

        def run_generate(context: str, answer: str, lang: str, mode: str, wh_type: str, show_unsafe_flag: bool):
            if not context.strip():
                raise gr.Error("Context is required")
            payload_wh = None if wh_type == "auto" else wh_type
            try:
                response = client.generate_safe(
                    context=context,
                    answer=answer or None,
                    lang=lang,
                    mode=mode,
                    wh_type=payload_wh,
                )
            except Exception as exc:  # pragma: no cover - runtime error path
                raise gr.Error(str(exc))

            question_display = normalize_question_display(
                response.get("question"), response.get("passed", False), show_unsafe_flag
            )
            badge_html = format_badge(bool(response.get("passed")))
            reasons_html = format_reasons(response.get("reasons", []))
            metrics_rows = metrics_to_table(response.get("metrics", {}))
            debug_payload = response.get("debug", {})
            json_payload = json.dumps(response, ensure_ascii=False, indent=2)
            return (
                question_display,
                badge_html,
                reasons_html,
                metrics_rows,
                debug_payload,
                json_payload,
                response,
            )

        result_state = gr.State()
        gen_btn.click(
            run_generate,
            inputs=[context_in, answer_in, lang_in, mode_in, wh_in, show_unsafe],
            outputs=[question_out, badge_md, reasons_md, metrics_table, debug_json, json_box, result_state],
        )

        def copy_json(resp: Optional[Dict]):
            if not resp:
                raise gr.Error("Nothing to copy yet")
            return json.dumps(resp, ensure_ascii=False, indent=2)

        copy_btn.click(copy_json, inputs=result_state, outputs=json_box)

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gradio demo for safety-aware QG")
    parser.add_argument("--mode", choices=["local", "http"], default="local")
    parser.add_argument("--server-url", dest="server_url", default=None)
    parser.add_argument("--config", dest="config", default=None)
    parser.add_argument("--dev", action="store_true", help="Enable dev-only controls")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = QGClient(mode=args.mode, server_url=args.server_url, config=args.config)
    app = make_app(client, dev_mode=args.dev)
    app.launch(show_error=True, share=args.share)


if __name__ == "__main__":
    main()
