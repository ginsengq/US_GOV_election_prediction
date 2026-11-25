"""FastAPI

Запуск:        uvicorn main:app --reload
Документация:  http://127.0.0.1:8000/docs
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent

model = joblib.load(ROOT / "model.joblib")
history = pd.read_csv(ROOT / "data" / "processed" / "state_features.csv")
FEATURES = list(model.feature_names_in_) 

app = FastAPI(
    title="US Presidential Election Predictor",
    description="Прогноз победителя президентских выборов США в штате "
                "(логистическая регрессия, обучена на выборах 1980–2024).",
    version="1.0.0",
)


class ElectionFeatures(BaseModel):
    """Признаки — с точки зрения партии действующего президента"""

    incumbent_party: Literal["D", "R"] = Field(description="Партия действующего президента")
    inc_prev_share: float = Field(ge=0, le=1, description="Доля партии власти в штате на прошлых выборах (0–1)")
    inc_prev_lean: float = Field(ge=-1, le=1, description="Насколько штат был за партию власти сильнее (+) или слабее (−) страны")
    inc_running: Literal[0, 1] = Field(description="1 — сам президент идёт на второй срок")
    inc_long_tenure: Literal[0, 1] = Field(description="1 — партия у власти 2+ срока подряд")
    inc_home_state: Literal[-1, 0, 1] = Field(description="+1 родной штат кандидата партии власти, −1 — соперника")

    model_config = {
        "json_schema_extra": {
            "examples": [{  # Пенсильвания, 2028 год
                "incumbent_party": "R",
                "inc_prev_share": 0.509,
                "inc_prev_lean": 0.001,
                "inc_running": 0,
                "inc_long_tenure": 0,
                "inc_home_state": 0,
            }]
        }
    }


class Prediction(BaseModel):
    winner: Literal["Democrat", "Republican"]
    probability_democrat: float
    probability_republican: float


class StatePrediction(Prediction):
    state: str
    year: int
    actual_winner: Literal["Democrat", "Republican"] | None = Field(
        default=None, description="Реальный результат (для прошедших выборов)")


def make_prediction(features: dict, incumbent_party: str) -> dict:
    x = pd.DataFrame([features])[FEATURES]
    p_incumbent = float(model.predict_proba(x)[0, 1])  # вероятность, что штат удержит партия власти
    p_dem = p_incumbent if incumbent_party == "D" else 1 - p_incumbent
    return {
        "winner": "Democrat" if p_dem > 0.5 else "Republican",
        "probability_democrat": round(p_dem, 3),
        "probability_republican": round(1 - p_dem, 3),
    }


@app.get("/")
def root():
    """Проверка, что сервис работает."""
    return {"message": "ML API is running"}


@app.post("/predict", response_model=Prediction)
def predict(data: ElectionFeatures):
    """Прогноз по признакам, которые передаёт пользователь."""
    features = data.model_dump(exclude={"incumbent_party"})
    return make_prediction(features, data.incumbent_party)


@app.get("/predict/{state}", response_model=StatePrediction)
def predict_state(state: str, year: int = 2028):
    """Прогноз для штата (код: PA, TX, CA…) — признаки берутся из исторических данных.
    """
    rows = history[(history.state == state.upper()) & (history.year == year)]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Нет данных для штата '{state}' в {year} году")
    row = rows.iloc[0]

    result = make_prediction(row[FEATURES].to_dict(), row.incumbent)
    actual = None if pd.isna(row.dem_win) else ("Democrat" if row.dem_win == 1 else "Republican")
    return {**result, "state": row.state, "year": int(row.year), "actual_winner": actual}
