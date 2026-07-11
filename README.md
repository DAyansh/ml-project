# 🎬 Box Office Revenue Predictor

A machine learning web app that predicts a movie's box office revenue based on its budget, popularity, runtime, and audience ratings — built with Python, Scikit-learn, and Streamlit.

## 📊 Project Overview

- **Dataset:** TMDB 5000 Movie Dataset (4,803 films)
- **Cleaning:** Removed 890 records with missing budget/revenue → 3,913 clean records
- **Model:** Linear Regression
- **Performance:** R² = 0.68, MAE ≈ $59.8M

## 🚀 Features

- 📊 **Dataset Explorer** — filter and browse the cleaned dataset
- 📈 **Visualizations** — budget vs revenue and popularity vs revenue trends
- 🤖 **Revenue Prediction** — enter a hypothetical movie's details, get an instant estimate
- 🏆 **Model Performance** — evaluation metrics and analysis

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Streamlit

## ⚙️ Setup

\`\`\`bash
git clone https://github.com/DAyansh/ml-project.git
cd ml-project
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## 📁 Project Structure

\`\`\`
BoxOfficePrediction/
├── data/               # Raw dataset (TMDB 5000)
├── notebooks/          # EDA and model training notebook
├── models/             # Saved trained model (.pkl)
├── app.py              # Streamlit dashboard
├── requirements.txt
└── README.md
\`\`\`

## 📈 Future Improvements

- Add genre as a categorical feature
- Try non-linear models (Random Forest, XGBoost)
- Handle revenue outliers for better blockbuster predictions