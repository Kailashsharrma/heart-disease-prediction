from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
model = joblib.load(BASE / "knn_heart_model.pkl")
scaler = joblib.load(BASE / "heart_scaler.pkl")
COLS = list(joblib.load(BASE / "heart_columns.pkl"))

app = FastAPI(title="Heart Disease Prediction API")


class Patient(BaseModel):
    Age: int = Field(ge=18, le=100)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["ATA", "NAP", "TA", "ASY"]
    RestingBP: int = Field(ge=80, le=200)
    Cholesterol: int = Field(ge=100, le=600)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(ge=60, le=220)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(ge=0, le=6)
    ST_Slope: Literal["Up", "Flat", "Down"]


def probability(p: dict) -> float:
    row = {c: 0 for c in COLS}
    row.update(Age=p["Age"], RestingBP=p["RestingBP"], Cholesterol=p["Cholesterol"],
               FastingBS=p["FastingBS"], MaxHR=p["MaxHR"],
               Oldpeak=int(p["Oldpeak"]))  # notebook cast all columns to int before training
    for prefix, key in [("Sex", "Sex"), ("ChestPainType", "ChestPainType"),
                        ("RestingECG", "RestingECG"), ("ExerciseAngina", "ExerciseAngina"),
                        ("ST_Slope", "ST_Slope")]:
        col = f"{prefix}_{p[key]}"
        if col in row:  # reference categories (F, ASY, LVH, N, Down) have no column
            row[col] = 1
    X = scaler.transform(pd.DataFrame([row])[COLS])
    return float(model.predict_proba(X)[0][1])


# "What if" scenarios: change one thing to a healthier value, see how the score moves.
SCENARIOS = [
    ("Bring cholesterol down to 200", {"Cholesterol": 200}),
    ("Bring blood pressure down to 120", {"RestingBP": 120}),
    ("Keep fasting blood sugar normal", {"FastingBS": 0}),
    ("No exercise-induced angina", {"ExerciseAngina": "N"}),
    ("ST depression (Oldpeak) at 0", {"Oldpeak": 0}),
    ("Healthy ST slope (Up)", {"ST_Slope": "Up"}),
    ("Raise max heart rate to 160", {"MaxHR": 160}),
]


@app.post("/predict")
def predict(patient: Patient):
    p = patient.model_dump()
    base = probability(p)
    levers = []
    for label, change in SCENARIOS:
        if all(p[k] == v for k, v in change.items()):
            continue
        drop = base - probability({**p, **change})
        if drop > 0:
            levers.append({"label": label, "drop": round(drop, 3)})
    levers.sort(key=lambda x: -x["drop"])
    levers = levers[:3]
    everything = {k: v for _, c in SCENARIOS for k, v in c.items()}
    combo = base - probability({**p, **everything})
    if combo > 0:
        levers.insert(0, {"label": "All of the improvements together", "drop": round(combo, 3)})
    return {"probability": round(base, 3), "prediction": int(base >= 0.5), "levers": levers}


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")
