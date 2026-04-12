"""CrossKEY HuggingFace Space -- Interactive 3D Keypoint Matching Demo.

Two-tab Gradio app:
  Tab 1 (Explore): Pre-computed results with adjustable matching parameters.
  Tab 2 (Your Data): Upload volumes + checkpoint for live inference.
"""

import logging
import os
import sys

# Add space/ to path so local imports work both locally and on HF
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import numpy as np

from inference import load_precomputed, run_inference, run_matching
from visualization import build_matching_figure

# ZeroGPU decorator -- no-op when running locally
try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    gpu_decorator = lambda fn: fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crosskey.app")

# -- Load pre-computed data on startup --
logger.info("Loading pre-computed demo data...")
DEMO_DATA = load_precomputed(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed")
)
logger.info(
    "Loaded: %d MR descriptors, %d US descriptors",
    len(DEMO_DATA["descriptors_mr"]),
    len(DEMO_DATA["descriptors_us"]),
)


def update_demo(
    ratio_threshold: float,
    evaluation_threshold: float,
    mutual: bool,
    metric: str,
) -> tuple:
    """Re-run matching with new parameters and rebuild the figure."""
    match_pairs, metrics = run_matching(
        DEMO_DATA["descriptors_mr"],
        DEMO_DATA["descriptors_us"],
        DEMO_DATA["points_mr"],
        DEMO_DATA["points_us"],
        ratio_threshold=ratio_threshold,
        mutual=mutual,
        metric=metric,
        evaluation_threshold=evaluation_threshold,
    )

    fig = build_matching_figure(
        volume_mr=DEMO_DATA["volume_mr"],
        volume_us=DEMO_DATA["volume_us"],
        points_mr=DEMO_DATA["points_mr"],
        points_us=DEMO_DATA["points_us"],
        padded_shape_mr=tuple(DEMO_DATA["metadata"]["padded_shape_mr"]),
        padded_shape_us=tuple(DEMO_DATA["metadata"]["padded_shape_us"]),
        match_pairs=match_pairs,
        metrics=metrics,
        evaluation_threshold=evaluation_threshold,
    )

    return (
        fig,
        metrics['num_matches'],
        metrics['num_correct'],
        round(metrics['precision'], 1),
        round(metrics['matching_score'], 4),
    )


@gpu_decorator
def run_custom_inference(mr_file, us_file, heatmap_file, ckpt_file):
    """Run inference on uploaded data. Uses ZeroGPU on HF Spaces."""
    if any(f is None for f in [mr_file, us_file, heatmap_file, ckpt_file]):
        raise gr.Error("Please upload all four files: MR volume, US volume, heatmap, and checkpoint.")

    logger.info("Running inference on uploaded data...")
    data = run_inference(
        mr_path=mr_file.name,
        us_path=us_file.name,
        heatmap_path=heatmap_file.name,
        checkpoint_path=ckpt_file.name,
    )
    return data


def update_custom(
    data: dict,
    ratio_threshold: float,
    evaluation_threshold: float,
    mutual: bool,
    metric: str,
) -> tuple:
    """Re-run matching on custom data with new parameters."""
    if data is None:
        raise gr.Error("Run inference first.")

    match_pairs, metrics = run_matching(
        data["descriptors_mr"],
        data["descriptors_us"],
        data["points_mr"],
        data["points_us"],
        ratio_threshold=ratio_threshold,
        mutual=mutual,
        metric=metric,
        evaluation_threshold=evaluation_threshold,
    )

    fig = build_matching_figure(
        volume_mr=data["volume_mr"],
        volume_us=data["volume_us"],
        points_mr=data["points_mr"],
        points_us=data["points_us"],
        padded_shape_mr=tuple(data["metadata"]["padded_shape_mr"]),
        padded_shape_us=tuple(data["metadata"]["padded_shape_us"]),
        match_pairs=match_pairs,
        metrics=metrics,
        evaluation_threshold=evaluation_threshold,
    )

    return (
        fig,
        metrics['num_matches'],
        metrics['num_correct'],
        round(metrics['precision'], 1),
        round(metrics['matching_score'], 4),
    )


# -- Build Gradio UI --

with gr.Blocks(
    title="CrossKEY -- 3D Cross-modal Keypoint Matching",
) as demo:
    gr.Markdown(
        "# CrossKEY\n"
        "**3D Cross-modal Keypoint Descriptor for MR-US Matching and Registration**"
    )

    with gr.Tabs():
        # ---- Tab 1: Explore ----
        with gr.Tab("Explore"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("### Matching Parameters")
                    demo_ratio = gr.Slider(
                        minimum=0.5, maximum=1.0, value=0.75, step=0.05,
                        label="Ratio Threshold",
                    )
                    demo_eval_thresh = gr.Slider(
                        minimum=1.0, maximum=10.0, value=5.0, step=0.5,
                        label="Evaluation Threshold (mm)",
                    )
                    demo_mutual = gr.Checkbox(value=True, label="Mutual Nearest Neighbor")
                    demo_metric = gr.Dropdown(
                        choices=["euclidean", "cosine"], value="euclidean",
                        label="Distance Metric",
                    )

                    gr.Markdown("### Results")
                    with gr.Group():
                        with gr.Row():
                            demo_n_matches = gr.Number(label="Matches", interactive=False)
                            demo_n_correct = gr.Number(label="Correct", interactive=False)
                        with gr.Row():
                            demo_precision = gr.Number(label="Precision (%)", interactive=False)
                            demo_match_score = gr.Number(label="Match Score", interactive=False)

                with gr.Column(scale=3):
                    demo_plot = gr.Plot(label="3D Matching Visualization")

            demo_inputs = [demo_ratio, demo_eval_thresh, demo_mutual, demo_metric]
            demo_outputs = [demo_plot, demo_n_matches, demo_n_correct, demo_precision, demo_match_score]

            # Update on any parameter change
            for inp in demo_inputs:
                inp.change(fn=update_demo, inputs=demo_inputs, outputs=demo_outputs)

            # Load initial results
            demo.load(fn=update_demo, inputs=demo_inputs, outputs=demo_outputs)

        # ---- Tab 2: Your Data ----
        with gr.Tab("Your Data"):
            gr.Markdown(
                "Upload your own MR volume, US volume, heatmap, and a trained CrossKEY checkpoint.\n\n"
                "Inference runs on GPU and may take 30-60 seconds."
            )

            with gr.Row():
                custom_mr = gr.File(label="MR Volume (.nii.gz)", file_types=[".nii.gz"])
                custom_us = gr.File(label="US Volume (.nii.gz)", file_types=[".nii.gz"])
            with gr.Row():
                custom_heatmap = gr.File(label="Heatmap (.nii.gz)", file_types=[".nii.gz"])
                custom_ckpt = gr.File(label="Checkpoint (.ckpt)", file_types=[".ckpt"])

            custom_run_btn = gr.Button("Run Inference (GPU)", variant="primary")

            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("### Matching Parameters")
                    custom_ratio = gr.Slider(
                        minimum=0.5, maximum=1.0, value=0.75, step=0.05,
                        label="Ratio Threshold",
                    )
                    custom_eval_thresh = gr.Slider(
                        minimum=1.0, maximum=10.0, value=5.0, step=0.5,
                        label="Evaluation Threshold (mm)",
                    )
                    custom_mutual = gr.Checkbox(value=True, label="Mutual Nearest Neighbor")
                    custom_metric = gr.Dropdown(
                        choices=["euclidean", "cosine"], value="euclidean",
                        label="Distance Metric",
                    )

                    gr.Markdown("### Results")
                    with gr.Group():
                        with gr.Row():
                            custom_n_matches = gr.Number(label="Matches", interactive=False)
                            custom_n_correct = gr.Number(label="Correct", interactive=False)
                        with gr.Row():
                            custom_precision = gr.Number(label="Precision (%)", interactive=False)
                            custom_match_score = gr.Number(label="Match Score", interactive=False)

                with gr.Column(scale=3):
                    custom_plot = gr.Plot(label="3D Matching Visualization")

            # State to hold inference results
            custom_data_state = gr.State(value=None)

            custom_param_inputs = [custom_ratio, custom_eval_thresh, custom_mutual, custom_metric]
            custom_outputs = [custom_plot, custom_n_matches, custom_n_correct, custom_precision, custom_match_score]

            # Inference button: run model, then update visualization
            def infer_and_display(mr_file, us_file, heatmap_file, ckpt_file, ratio, eval_thresh, mutual, metric):
                data = run_custom_inference(mr_file, us_file, heatmap_file, ckpt_file)
                fig, n_m, n_c, prec, ms = update_custom(data, ratio, eval_thresh, mutual, metric)
                return data, fig, n_m, n_c, prec, ms

            custom_run_btn.click(
                fn=infer_and_display,
                inputs=[custom_mr, custom_us, custom_heatmap, custom_ckpt] + custom_param_inputs,
                outputs=[custom_data_state] + custom_outputs,
            )

            # Re-match on parameter change (no re-inference)
            for inp in custom_param_inputs:
                inp.change(
                    fn=update_custom,
                    inputs=[custom_data_state] + custom_param_inputs,
                    outputs=custom_outputs,
                )


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css="footer {display: none !important;}",
    )
