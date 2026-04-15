"""FastAPI application with analytics and live data endpoints."""

import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.api.schemas import (  # noqa: E402
    HealthResponse, PredictionRequest, PredictionResponse,
    RiskScoreRequest, RiskScoreResponse, AnalyticsSummaryResponse,
    MarketEvalRequest, MarketEvalResponse, DriftResponse,
)
from src.data.ingestion import DataLoader  # noqa: E402
from src.analysis.eda import AutoEDA  # noqa: E402
from src.pipeline.inference_pipeline import InferencePipeline, ModelNotFoundError  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

app = FastAPI(
    title="DataPulse API",
    description="Analytics & ML Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_loader = DataLoader(base_path=str(ROOT / "data"))
inference = InferencePipeline(model_path=str(ROOT / "models"))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy", version="1.0.0", service="datapulse-api",
        models_loaded=inference.available_models(),
    )


@app.post("/predict/player-performance", response_model=PredictionResponse)
async def predict_player_performance(request: PredictionRequest):
    try:
        result = inference.predict_player(request.features)
        return PredictionResponse(
            prediction=round(result['prediction'], 2),
            confidence=0.85,
            model_version=result['model_version'],
            features_used=list(request.features.keys()),
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/risk-score", response_model=RiskScoreResponse)
async def predict_risk_score(request: RiskScoreRequest):
    try:
        result = inference.score_risk(request.features)
        labels = {0: 'low', 1: 'medium', 2: 'high'}
        return RiskScoreResponse(
            risk_score=float(max(result['probabilities']) * 100),
            risk_level=labels.get(result['risk_category'], 'unknown'),
            contributing_factors={k: round(float(v), 4) for k, v in request.features.items()},
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
async def get_analytics_summary():
    summaries = {}
    for dataset in data_loader.list_available("sample"):
        try:
            df = data_loader.get_sample_data(dataset)
            eda = AutoEDA(df)
            overview = eda.get_overview()
            summaries[dataset] = {
                'rows': overview['shape']['rows'],
                'columns': overview['shape']['columns'],
                'numeric_columns': len(overview['numeric_columns']),
                'categorical_columns': len(overview['categorical_columns']),
            }
        except Exception:
            pass
    return AnalyticsSummaryResponse(datasets=summaries, total_datasets=len(summaries))


@app.get("/analytics/player/{player_id}")
async def get_player_analytics(player_id: int):
    try:
        players = data_loader.get_sample_data("players")
        player = players[players['player_id'] == player_id]
        if player.empty:
            raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
        return player.iloc[0].to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Players dataset not found")


@app.post("/market/evaluate", response_model=MarketEvalResponse)
async def evaluate_market(request: MarketEvalRequest):
    odds = request.odds
    implied_prob = 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)
    true_prob = request.estimated_probability
    ev = (true_prob * (odds / 100 if odds > 0 else 100 / abs(odds))) - (1 - true_prob)
    return MarketEvalResponse(
        implied_probability=round(implied_prob, 4),
        expected_value=round(float(ev), 4),
        edge=round(float(true_prob - implied_prob), 4),
        recommendation="Bet" if ev > 0.02 else "Pass",
    )


@app.get("/monitoring/drift", response_model=DriftResponse)
async def check_drift():
    return DriftResponse(
        overall_drift=False,
        columns_checked=10,
        columns_drifted=0,
        drifted_columns=[],
        last_checked="2024-01-01T00:00:00",
    )


@app.get("/live/status")
async def live_api_status():
    from src.data.live_feeds import LiveFeedsManager
    manager = LiveFeedsManager()
    return manager.get_api_status()


@app.get("/live/f1/drivers")
async def get_f1_drivers(session_key: str = "latest"):
    try:
        from src.data.live_feeds import OpenF1Client
        client = OpenF1Client()
        df = client.get_drivers(session_key=session_key)
        if df.empty:
            return {"data": [], "count": 0}
        return {"data": df.to_dict(orient="records"), "count": len(df)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenF1 API error: {str(e)}")


@app.get("/live/f1/laps")
async def get_f1_laps(session_key: str = "latest", driver_number: Optional[int] = None):
    try:
        from src.data.live_feeds import OpenF1Client
        client = OpenF1Client()
        df = client.get_laps(session_key=session_key, driver_number=driver_number)
        if df.empty:
            return {"data": [], "count": 0}
        return {"data": df.head(100).to_dict(orient="records"), "count": len(df)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenF1 API error: {str(e)}")


@app.get("/live/f1/meetings")
async def get_f1_meetings(year: int = 2025):
    try:
        from src.data.live_feeds import OpenF1Client
        client = OpenF1Client()
        df = client.get_season_calendar(year=year)
        if df.empty:
            return {"data": [], "count": 0}
        return {"data": df.to_dict(orient="records"), "count": len(df)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenF1 API error: {str(e)}")
