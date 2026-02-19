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
st.write("Ce modèle utilise NeuralProphet pour la tendance long-terme et un LSTM pré-entraîné pour la correction de volatilité.")

# --- ARCHITECTURE DU MODÈLE LSTM (Doit être identique à l'entraînement) ---
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
    # Charger le scaler sauvegardé
    scaler = joblib.load("residus_scaler.gz")
    # Charger le modèle LSTM
    model = IBM_LSTM()
    model.load_state_dict(torch.load("ibm_lstm_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model, scaler

# --- LOGIQUE DE PRÉDICTION ---
def run_analysis(use_gap):
    # 1. Récupération des données fraîches
    with st.spinner("Téléchargement des données IBM en direct..."):
        df_raw = yf.download("IBM", start="2018-01-01").reset_index()
        df = df_raw[['Date', 'Close']].copy()
        df.columns = ['ds', 'y']

    # 2. Fit rapide de NeuralProphet pour la tendance actuelle
    with st.spinner("Ajustement de la tendance (NeuralProphet)..."):
        m_np = NeuralProphet(learning_rate=0.01)
        # On désactive les logs pour Streamlit
        m_np.fit(df, freq="D", progress=None)
        forecast = m_np.predict(df)
        
        # Calcul des résidus actuels
        residus_actuels = df['y'].values - forecast['yhat1'].values
        res_scaled = scaler_res.transform(residus_actuels.reshape(-1, 1))

    # 3. Prédiction Future (5 jours)
    future = m_np.make_future_dataframe(df, periods=7, n_historic_predictions=False)
    forecast_future = m_np.predict(future)
    
    # Filtrage jours ouvrés (Business Days)
    forecast_future['day'] = forecast_future['ds'].dt.dayofweek
    biz_days = forecast_future[forecast_future['day'] < 5].head(5)
    
    f_trend = biz_days['yhat1'].values
    f_dates = biz_days['ds'].values

    # 4. Boucle LSTM pour les résidus futurs
    curr_batch = torch.FloatTensor(res_scaled[-30:].reshape(1, 30, 1))
    preds_res_scaled = []
    
    for i in range(len(f_trend)):
        with torch.no_grad():
            pred = model_lstm(curr_batch).item()
            # Clipping de sécurité
            pred = max(min(pred, 0.2), -0.2)
            preds_res_scaled.append(pred)
            # Update batch
            new_val = torch.FloatTensor([[[pred]]])
            curr_batch = torch.cat((curr_batch[:, 1:, :], new_val), dim=1)

    # Re-transformation des résidus
    res_final_dollars = scaler_res.inverse_transform(np.array(preds_res_scaled).reshape(-1, 1)).flatten()
    
    # 5. Application du Gap et Somme Finale
    base_pred = f_trend + res_final_dollars
    
    if use_gap:
        gap = df['y'].iloc[-1] - forecast['yhat1'].iloc[-1]
        final_pred = base_pred + gap
        st.success(f"Signal recalé sur le prix actuel (Gap : {gap:.2f} $)")
    else:
        final_pred = base_pred
        st.warning("Mode sans correction : Affichage de la valeur intrinsèque théorique.")

    return df, f_dates, final_pred, df['y'].iloc[-1]

# --- INTERFACE UTILISATEUR ---
model_lstm, scaler_res = load_assets()

st.sidebar.header("Configuration")
gap_option = st.sidebar.toggle("Appliquer correction du Gap", value=True)

if st.sidebar.button("Calculer les prévisions"):
    df_hist, dates, preds, last_price = run_analysis(gap_option)

    # Affichage des métriques
    c1, c2, c3 = st.columns(3)
    c1.metric("Dernière Clôture", f"{last_price:.2f} $")
    c2.metric("Prédiction J+5", f"{preds[-1]:.2f} $")
    c3.metric("Variation Attendue", f"{((preds[-1]-last_price)/last_price)*100:+.2f} %")

    # Graphique
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_hist['ds'].tail(40), df_hist['y'].tail(40), label="Réel", color="black", alpha=0.6)
    ax.plot(dates, preds, 'r--o', label="Hybride (NP + LSTM)")
    ax.set_ylabel("Prix IBM ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # Tableau de données
    st.write("### Prévisions détaillées")
    df_res = pd.DataFrame({"Date": dates.astype(str), "Prix Prédit": preds})
    st.dataframe(df_res.style.format({"Prix Prédit": "{:.2f} $"}))