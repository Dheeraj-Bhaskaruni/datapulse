"""DataPulse Gradio interface for Hugging Face Spaces deployment."""

import gradio as gr
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ingestion import DataLoader
from src.pipeline.inference_pipeline import InferencePipeline, ModelNotFoundError

inference = InferencePipeline(model_path=str(ROOT / "models"))
data_loader = DataLoader(str(ROOT / "data"))


def predict_performance(points_avg, assists_avg, rebounds_avg, salary, consistency):
    features = {
        "points_avg": points_avg, "assists_avg": assists_avg,
        "rebounds_avg": rebounds_avg, "salary": salary,
        "consistency_score": consistency,
    }
    try:
        result = inference.predict_player(features)
        predicted = result['prediction']
        value_per_k = predicted / (salary / 1000) if salary > 0 else 0
        return {
            "Predicted Fantasy Points": round(predicted, 1),
            "Model Version": result['model_version'],
            "Value Per $1K Salary": round(value_per_k, 2),
            "Value Rating": "Excellent" if value_per_k > 5 else "Good" if value_per_k > 3 else "Fair",
        }
    except ModelNotFoundError:
        return {"Error": "Player model not loaded. Run 'make build' to train models."}


def score_risk(win_rate, total_wagered, roi, total_entries):
    features = {
        "total_entries": float(total_entries), "win_rate": win_rate,
        "avg_entry_fee": total_wagered / max(total_entries, 1),
        "total_wagered": total_wagered,
        "total_won": total_wagered * (1 + roi),
        "net_profit": total_wagered * roi,
    }
    try:
        result = inference.score_risk(features)
        labels = {0: "Low", 1: "Medium", 2: "High"}
        return {
            "Risk Category": labels.get(result['risk_category'], str(result['risk_category'])),
            "Confidence": f"{max(result['probabilities']):.1%}",
            "Model Version": result['model_version'],
            "Class Probabilities": {labels.get(i, str(i)): round(p, 4) for i, p in enumerate(result['probabilities'])},
        }
    except ModelNotFoundError:
        return {"Error": "Risk model not loaded. Run 'make build' to train models."}


def evaluate_market(odds, estimated_probability):
    implied = 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)
    payout = odds / 100 if odds > 0 else 100 / abs(odds)
    ev = (estimated_probability * payout) - (1 - estimated_probability)
    edge = estimated_probability - implied
    return {
        "Implied Probability": f"{implied:.1%}",
        "Expected Value": round(ev, 4),
        "Edge": f"{edge:.1%}",
        "Recommendation": "BET" if ev > 0.02 else "MARGINAL" if ev > 0 else "PASS",
    }


def explore_dataset(dataset_name):
    try:
        df = data_loader.get_sample_data(dataset_name)
    except FileNotFoundError:
        return "Dataset not found."
    summary = f"**Shape:** {df.shape[0]:,} rows x {df.shape[1]} columns\n\n"
    summary += f"**Columns:** {', '.join(df.columns.tolist())}\n\n"
    summary += "**Sample Data:**\n\n"
    summary += df.head(5).to_markdown()
    summary += "\n\n**Statistics:**\n\n"
    summary += df.describe().round(2).to_markdown()
    return summary


with gr.Blocks(title="DataPulse Analytics", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# DataPulse Analytics\n*Analytics & ML Platform*")
    gr.Markdown(f"Models loaded: **{', '.join(inference.available_models()) or 'none'}**")

    with gr.Tab("Player Performance"):
        gr.Markdown("### Predict Player Fantasy Points")
        with gr.Row():
            pts = gr.Number(label="Points Avg", value=20.0, minimum=0, maximum=50)
            ast = gr.Number(label="Assists Avg", value=5.0, minimum=0, maximum=15)
            reb = gr.Number(label="Rebounds Avg", value=7.0, minimum=0, maximum=15)
        with gr.Row():
            sal = gr.Number(label="Salary", value=7000, minimum=3000, maximum=12000)
            con = gr.Slider(0, 1, value=0.7, label="Consistency Score")
        predict_btn = gr.Button("Predict Performance", variant="primary")
        predict_output = gr.JSON(label="Prediction Results")
        predict_btn.click(predict_performance, inputs=[pts, ast, reb, sal, con], outputs=predict_output)

    with gr.Tab("Risk Scoring"):
        gr.Markdown("### Evaluate User Risk Level")
        with gr.Row():
            wr = gr.Slider(0, 1, value=0.5, label="Win Rate")
            tw = gr.Number(label="Total Wagered ($)", value=10000, minimum=0)
        with gr.Row():
            roi_input = gr.Slider(-0.5, 1.0, value=0.05, label="ROI")
            te = gr.Number(label="Total Entries", value=500, minimum=0)
        risk_btn = gr.Button("Score Risk", variant="primary")
        risk_output = gr.JSON(label="Risk Assessment")
        risk_btn.click(score_risk, inputs=[wr, tw, roi_input, te], outputs=risk_output)

    with gr.Tab("Market Evaluation"):
        gr.Markdown("### Evaluate Market Expected Value")
        with gr.Row():
            odds_input = gr.Number(label="American Odds (e.g., -110, +150)", value=-110)
            prob_input = gr.Slider(0, 1, value=0.55, label="Your Estimated True Probability")
        market_btn = gr.Button("Evaluate Market", variant="primary")
        market_output = gr.JSON(label="Market Analysis")
        market_btn.click(evaluate_market, inputs=[odds_input, prob_input], outputs=market_output)

    with gr.Tab("Data Explorer"):
        gr.Markdown("### Explore Sample Datasets")
        dataset_select = gr.Dropdown(
            choices=["players", "contests", "user_profiles", "market_odds", "user_entries"],
            value="players", label="Select Dataset",
        )
        explore_btn = gr.Button("Load & Explore", variant="primary")
        explore_output = gr.Markdown(label="Dataset Summary")
        explore_btn.click(explore_dataset, inputs=[dataset_select], outputs=explore_output)

if __name__ == "__main__":
    demo.launch()
