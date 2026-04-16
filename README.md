# DataPulse Analytics

```
  ____        __        ____        __
 / __ \____ _/ /_____ _/ __ \__  __/ /______
/ / / / __ `/ __/ __ `/ /_/ / / / / / ___/ _ \
/ /_/ / /_/ / /_/ /_/ / ____/ /_/ / (__  )  __/
\____/\__,_/\__/\__,_/_/    \__,_/_/____/\___/
```

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live-datapulse.lifedblocks.com-2563eb)](https://datapulse.lifedblocks.com)

> **Live at [datapulse.lifedblocks.com](https://datapulse.lifedblocks.com)**

An enterprise-grade, domain-agnostic analytics platform showcasing the complete data science stack — from raw data ingestion through feature engineering, model training, evaluation, monitoring, and multi-deployment production interfaces.

---

## Architecture

```
datapulse/
├── src/
│   ├── data/          ingestion, preprocessing, validation, live feeds
│   ├── features/      player, market, user feature generators + feature store
│   ├── models/        player_performance, risk_scoring, market_predictor,
│   │                  anomaly_detection, ensemble, evaluation
│   ├── analysis/      EDA, statistical tests, time series, segmentation
│   ├── visualization/ plots, dashboards
│   ├── monitoring/    drift detection, alerting, structured logging
│   ├── api/           FastAPI app, schemas, middleware
│   └── pipeline/      training, inference, scheduler
├── app/
│   ├── streamlit_app.py   — Interactive dashboard (primary UI)
│   ├── gradio_app.py      — Hugging Face Spaces interface
│   └── flask_app.py       — cPanel/traditional hosting
├── huggingface/           — HF Spaces deployment entry point
├── data/sample/           — Realistic synthetic datasets
├── models/                — Pre-trained model artifacts (.joblib)
├── notebooks/             — Jupyter notebooks
├── tests/                 — pytest suite
└── config/                — YAML configuration and logging
```

## Features

### Data Stack
- Universal data loader (CSV, JSON, Parquet, Excel, TSV, URL)
- Schema-based validation with severity levels
- Automated data cleaning (dedup, imputation, outlier removal, dtype fixing)
- File-based feature store with versioning and checksums
- Live sports data feeds (F1, Cricket, NBA, Betting Odds)

### ML Models
- **PlayerPerformanceModel** — Gradient Boosting regression for fantasy point prediction
- **RiskScoringModel** — Random Forest classification for user risk categorization
- **MarketPredictorModel** — Calibrated GBM classifier with isotonic calibration
- **AnomalyDetectionModel** — Isolation Forest + z-score statistical detection
- **EnsembleModel** — Weighted ensemble wrapper for any combination of models

### Analysis
- AutoEDA with correlation, distribution, outlier, and missing-value analysis
- Statistical testing: t-test, chi-square, ANOVA, Shapiro-Wilk, bootstrap CI, A/B test
- Time series decomposition, stationarity testing, and exponential smoothing forecasts
- K-Means and DBSCAN segmentation with optimal-k selection

### Monitoring
- PSI (Population Stability Index) and KS-test drift detection
- Performance degradation detection with z-score thresholds
- Structured alert system with pluggable handlers

### Interfaces
- **Streamlit Dashboard** — Full interactive UI with 8 pages
- **FastAPI** — REST API with auto docs at `/docs`
- **Gradio** — Hugging Face Spaces compatible interface
- **Flask** — cPanel/traditional hosting compatible

---

## Quick Start

```bash
python >= 3.9
pip install -r requirements.txt

# Generate data + train models
make build

# Launch dashboard
make serve        # http://localhost:8501

# Launch API
make api          # http://localhost:8000/docs

# Run tests
make test
```

---

## Deployment

### Docker
```bash
docker build -t datapulse .
docker run -p 8501:8501 -p 8000:8000 datapulse
```

### Hugging Face Spaces
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/datapulse
git push hf main
```

### cPanel (Currently deployed)
Deployed and live at **[datapulse.lifedblocks.com](https://datapulse.lifedblocks.com)** via Passenger WSGI.

1. Upload files, set Python 3.9+, install requirements
2. Run `make build` to generate data + train models
3. Point `wsgi.py` as the WSGI entry point

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/predict/player-performance` | Predict fantasy points |
| POST | `/predict/risk-score` | Score user risk |
| GET | `/analytics/summary` | Dataset analytics summary |
| POST | `/market/evaluate` | Market EV evaluation |
| GET | `/monitoring/drift` | Drift detection status |
| GET | `/live/f1/drivers` | F1 driver data |
| GET | `/live/f1/laps` | F1 lap time data |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | pandas, numpy, pyarrow |
| ML | scikit-learn, XGBoost, LightGBM |
| Stats | scipy |
| Visualization | Plotly, Streamlit |
| API | FastAPI, Uvicorn, Flask |
| Interface | Gradio |
| Testing | pytest |
| Containers | Docker |
| CI/CD | GitHub Actions |

---

## License

MIT
