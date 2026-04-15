"""Pydantic request/response schemas for the API."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str
    models_loaded: List[str] = []


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        json_schema_extra={"example": {"points_avg": 25.5, "assists_avg": 7.2, "salary": 8500}},
    )
    model_version: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    features_used: List[str]


class RiskScoreRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        json_schema_extra={"example": {"win_rate": 0.65, "total_wagered": 50000, "roi": 0.12}},
    )
    user_id: Optional[int] = None


class RiskScoreResponse(BaseModel):
    risk_score: float
    risk_level: str
    contributing_factors: Dict[str, float]


class AnalyticsSummaryResponse(BaseModel):
    datasets: Dict[str, Any]
    total_datasets: int


class MarketEvalRequest(BaseModel):
    odds: float = Field(..., json_schema_extra={"example": -110})
    estimated_probability: float = Field(
        ..., ge=0, le=1, json_schema_extra={"example": 0.55}
    )
    market_type: Optional[str] = "moneyline"


class MarketEvalResponse(BaseModel):
    implied_probability: float
    expected_value: float
    edge: float
    recommendation: str


class DriftResponse(BaseModel):
    overall_drift: bool
    columns_checked: int
    columns_drifted: int
    drifted_columns: List[str]
    last_checked: str
