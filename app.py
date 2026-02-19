import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="IBM Hybrid Predictor", layout="wide")
st.title("🚀 IBM Stock Predictor: Prophet + LSTM Hybrid")

# --- ARCHITECTURE DU MODÈLE LSTM ---
class IBM_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(IBM_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- CHARGEMENT DES COMPOSANTS ---
@st.cache_resource
def load_assets():
    scaler = joblib.load("res_scaler.pkl") # Assurez-vous que le fichier est présent
    model_lstm = IBM_LSTM()
    # model_lstm.load_state_dict(torch.load("ibm_lstm_model.pth")) # Optionnel si vous l'utilisez
    model_lstm.eval()
    return model_lstm, scaler

def run_analysis():
    # 1. Téléchargement des données (Fin au 19-02 pour prédire le 20-02)
    # Note: end="2026-02-20" car la borne supérieure est exclusive dans yfinance
    df = yf.download("IBM", start="2018-01-01", end="2026-02-20")
    df = df.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    
    last_price = df['y'].iloc[-1]

    # 2. Configuration NeuralProphet (Fréquence B)
    m = NeuralProphet(
        n_changepoints=10,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    # Entraînement rapide pour la démo (ou chargement de modèle)
    m.fit(df, freq="B") 

    # 3. Prévisions Futures (Hybride)
    future = m.make_future_dataframe(df, periods=5, n_historic_predictions=False)
    forecast = m.predict(future)
    
    # On récupère les valeurs brutes (yhat1)
    prediction_totale = forecast['yhat1'].values
    f_dates = forecast['ds'].values

    # 4. NOUVEAU CALCUL DU GAP (Basé sur la variable prediction_totale)
    # On compare le dernier prix réel avec la première prédiction hybride (J+1)
    pred_brute_demain = prediction_totale[0]
    gap_correct = last_price - pred_brute_demain
    
    # Application du gap
    prediction_ajustee = prediction_totale + gap_correct

    return df, f_dates, prediction_ajustee, last_price, gap_correct, pred_brute_demain

# --- INTERFACE UTILISATEUR ---
if st.sidebar.button("Calculer les prévisions"):
    with st.spinner("Analyse du marché en cours..."):
        df_hist, dates, preds, last_price, gap, pred_brute = run_analysis()

    # Affichage des métriques
    c1, c2, c3 = st.columns(3)
    c1.metric("Dernière Clôture (19-02)", f"{last_price:.2f} $")
    c2.metric("Prédiction Demain (Ajustée)", f"{preds[0]:.2f} $")
    c3.metric("Correction Gap", f"{gap:+.2f} $", delta_color="inverse")

    # Explication technique du Gap
    with st.expander("Détails du recalage (Debug)"):
        st.write(f"Prix réel au 19-02 : **{last_price:.2f} $**")
        st.write(f"Prédiction brute modèle (sans gap) : **{pred_brute:.2f} $**")
        st.write(f"L'écart de **{gap:.2f} $** a été appliqué pour synchroniser le modèle avec le marché.")

    # Graphique Plotly (pour éviter l'erreur Plotly failed)
    import plotly.graph_objects as go
    
    fig = go.Figure()
    # Historique récent
    fig.add_trace(go.Scatter(x=df_hist['ds'].tail(30), y=df_hist['y'].tail(30), name="Historique"))
    # Prédictions
    fig.add_trace(go.Scatter(x=dates, y=preds, name="Prédiction (Gap corrigé)", line=dict(color='red', dash='dash')))
    
    fig.update_layout(title="Prévisions IBM (Post-Chute Février)", xaxis_title="Date", yaxis_title="Prix ($)")
    st.plotly_chart(fig, use_container_width=True)

