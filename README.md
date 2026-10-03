# Heart Disease Prediction

A web app that estimates heart disease risk from clinical readings, using a trained **KNN machine-learning model** with a **FastAPI** backend and an interactive HTML frontend.

## Features

- Predicts heart disease risk from 11 clinical inputs
- Live result: the score updates as you move the sliders
- Risk gauge with Low, Moderate and High levels
- "What would help most" list, showing how much the risk could drop if values improve
- Downloadable report (print it or save it as PDF)
- Light and dark mode, works on mobile

## Tech Stack

- **Model:** scikit-learn (KNN, StandardScaler)
- **Backend:** FastAPI, Uvicorn, pandas, joblib
- **Frontend:** HTML, CSS and JavaScript

## Inputs

Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced Angina, Oldpeak (ST depression), ST Slope.

## Project Structure

```
main.py                  FastAPI backend
index.html               Frontend
knn_heart_model.pkl      Trained KNN model
heart_scaler.pkl         Fitted StandardScaler
heart_columns.pkl        Expected feature columns
HeartdiseaseFinal.ipynb  Model training notebook
requirements.txt         Python dependencies
```

## How to Run

1. Clone the repository

```bash
git clone https://github.com/Kailashsharrma/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Start the server

```bash
python -m uvicorn main:app --reload
```

4. Open `http://127.0.0.1:8000` in your browser.

## API

`POST /predict` takes the 11 inputs as JSON and returns the risk probability and a list of improvements that would lower it.

## Notes

- The model was trained with scikit-learn 1.6.1. Other versions may show an `InconsistentVersionWarning`, which is harmless in most cases.
- KNN uses 5 neighbours, so the score takes values like 0%, 20%, 40%, 60%, 80% or 100%.

## Disclaimer

This project is for learning purposes only. It is **not** a medical diagnosis. Please consult a doctor for medical advice.

## Author

Made by **Kailash Kumar**
