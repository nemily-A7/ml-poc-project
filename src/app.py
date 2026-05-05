from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import MODEL_METRICS_FILE, MODELS_DIR, PLOTS_DIR


CSS = """
<style>
    .stApp {
        background: linear-gradient(160deg, #0d0d1a 0%, #1a1a35 50%, #0d0d1a 100%);
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #f72585, #7209b7, #3a86ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }

    .hero-sub {
        font-size: 1.15rem;
        color: rgba(255,255,255,0.6);
        text-align: center;
        margin-bottom: 2rem;
    }

    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }

    #MainMenu, footer, header {visibility: hidden;}
</style>
"""


def _zone(puissance: int) -> tuple[str, str, str]:
    if puissance < 100:
        return "Zone 1 — Récupération", "#48cae4", "Effort très léger. Idéal pour récupérer entre deux séances."
    elif puissance < 180:
        return "Zone 2 — Endurance", "#80b918", "Allure confortable. La base de l'entraînement cycliste."
    elif puissance < 250:
        return "Zone 3 — Tempo", "#f9c74f", "Effort soutenu. On commence à bien souffler."
    elif puissance < 320:
        return "Zone 4 — Seuil lactique", "#f77f00", "Effort intense. La limite de ce qu'on peut tenir longtemps."
    else:
        return "Zone 5+ — VO₂max", "#e63946", "Effort maximal. Sprints et montées à bloc."


def page_resultats() -> None:
    st.markdown('<p class="hero-title">⚡ Capteur Virtuel de Puissance</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Un algorithme qui estime la puissance d\'un cycliste — sans capteur à 1 000 €</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Précision du meilleur modèle", "50 %", "XGBoost")
    col2.metric("⚡ Erreur moyenne", "58 W", "-10 W vs Random Forest", delta_color="inverse")
    col3.metric("📊 Données utilisées", "1.62 M lignes", "163 sorties · 1 athlète")

    st.markdown("### 💡 Ce que ça veut dire concrètement")

    col_a, col_b = st.columns(2)
    with col_a:
        st.info(
            "**Exemple :** un cycliste qui pédale à 250 W réels — "
            "notre algorithme estimera entre **~192 W et ~308 W** en moyenne. "
            "C'est imparfait, mais c'est entièrement gratuit, sans aucun capteur matériel."
        )
    with col_b:
        st.warning(
            "**Pourquoi pas 100 % précis ?** "
            "Le vent et l'aérodynamisme influencent énormément la puissance réelle, "
            "mais ces facteurs sont invisibles pour nos capteurs bas coût."
        )

    if MODEL_METRICS_FILE.exists():
        st.markdown("### 🏅 Comparaison des 3 algorithmes")
        df = pd.read_csv(MODEL_METRICS_FILE)

        noms = df["model_name"].tolist()
        colors = ["#3a86ff", "#8338ec", "#ff006e"]

        col_r2, col_mae, col_rmse = st.columns(3)

        with col_r2:
            fig = go.Figure(go.Bar(
                x=noms, y=df["r2"].tolist(),
                marker_color=colors, text=[f"{v:.0%}" for v in df["r2"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="R² — Précision<br><sup>Plus haut = mieux (max 1.0)</sup>",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), yaxis=dict(range=[0, 0.7], showgrid=False),
                xaxis=dict(showgrid=False), margin=dict(t=60, b=10), height=300,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_mae:
            fig = go.Figure(go.Bar(
                x=noms, y=df["mae"].tolist(),
                marker_color=colors, text=[f"{v:.0f} W" for v in df["mae"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="MAE — Erreur moyenne<br><sup>Plus bas = mieux</sup>",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), yaxis=dict(range=[0, 90], showgrid=False),
                xaxis=dict(showgrid=False), margin=dict(t=60, b=10), height=300,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_rmse:
            fig = go.Figure(go.Bar(
                x=noms, y=df["rmse"].tolist(),
                marker_color=colors, text=[f"{v:.0f} W" for v in df["rmse"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="RMSE — Erreur sur les pics<br><sup>Plus bas = mieux</sup>",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), yaxis=dict(range=[0, 120], showgrid=False),
                xaxis=dict(showgrid=False), margin=dict(t=60, b=10), height=300,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Puissance réelle vs. estimée par XGBoost")
    chemin_pred = PLOTS_DIR / "xgboost_reel_vs_predit.png"
    if chemin_pred.exists():
        st.image(str(chemin_pred), use_container_width=True)
        st.caption(
            "Chaque point = 1 seconde de pédalage. "
            "La ligne rouge = prédiction parfaite. "
            "Plus les points sont proches de la ligne, meilleure est l'estimation."
        )

    st.markdown("### 🔑 Quels capteurs comptent le plus ?")
    chemin_imp = PLOTS_DIR / "feature_importance.png"
    if chemin_imp.exists():
        st.image(str(chemin_imp), use_container_width=True)
        st.caption("La vitesse et la pente sont de loin les indicateurs les plus importants pour prédire la puissance.")


def page_demo() -> None:
    st.markdown('<p class="hero-title">🎮 Essaie toi-même</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Règle les curseurs comme si tu étais sur ton vélo — et vois la puissance estimée en temps réel</p>',
        unsafe_allow_html=True,
    )

    chemin_modele = MODELS_DIR / "xgboost.joblib"
    if not chemin_modele.exists():
        st.error("⚠️ Modèle introuvable. Lance le notebook training.ipynb d'abord.")
        return

    modele = joblib.load(chemin_modele)

    col1, col2, col3 = st.columns(3)
    with col1:
        hr  = st.slider("💓 Fréquence cardiaque", 60, 200, 150, format="%d bpm")
        cad = st.slider("🔄 Cadence de pédalage", 0, 120, 85, format="%d rpm")
    with col2:
        vitesse = st.slider("🚴 Vitesse", 0, 80, 30, format="%d km/h")
        pente   = st.slider("⛰️ Pente", -15, 20, 0, format="%d%%")
    with col3:
        alt = st.slider("📍 Altitude", 0, 2500, 200, format="%d m")

    X = np.array([[
        hr, cad, alt, vitesse, pente,
        0.0, 0.0,
        float(vitesse), float(hr), float(cad), float(pente),
    ]])

    puissance = max(0, round(modele.predict(X)[0]))
    zone_label, zone_color, zone_desc = _zone(puissance)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=puissance,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Puissance estimée (watts)", "font": {"color": "white", "size": 18}},
        number={"suffix": " W", "font": {"color": zone_color, "size": 56}},
        gauge={
            "axis": {
                "range": [0, 600],
                "tickcolor": "rgba(255,255,255,0.4)",
                "tickfont": {"color": "rgba(255,255,255,0.4)"},
            },
            "bar": {"color": zone_color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   100], "color": "rgba(72,202,228,0.12)"},
                {"range": [100, 180], "color": "rgba(128,185,24,0.12)"},
                {"range": [180, 250], "color": "rgba(249,199,79,0.12)"},
                {"range": [250, 320], "color": "rgba(247,127,0,0.12)"},
                {"range": [320, 600], "color": "rgba(230,57,70,0.12)"},
            ],
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=320,
        margin=dict(t=30, b=10, l=30, r=30),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown(
        f"<h3 style='text-align:center; color:{zone_color}'>{zone_label}</h3>"
        f"<p style='text-align:center; color:rgba(255,255,255,0.7); font-size:1.05rem'>{zone_desc}</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### 🏆 Pour situer par rapport aux pros")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tour de France (étape)", "350–400 W", "moyenne sur plusieurs heures")
    c2.metric("Cycliste amateur (1 h)", "150–250 W", "allure soutenue")
    c3.metric("Sprint maximal (2 s)", "800–1 200 W", "effort court et explosif")


def page_donnees() -> None:
    st.markdown('<p class="hero-title">🔍 Les données</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">163 sorties vélo · 1.62 million de secondes de pédalage · 1 seul athlète</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        p = PLOTS_DIR / "distribution_puissance.png"
        if p.exists():
            st.image(str(p), caption="La plupart du temps, le cycliste pédale entre 100 et 300 watts")
    with col2:
        p = PLOTS_DIR / "distribution_fc.png"
        if p.exists():
            st.image(str(p), caption="La fréquence cardiaque oscille surtout entre 120 et 175 bpm")

    p = PLOTS_DIR / "correlation_matrix.png"
    if p.exists():
        st.image(str(p), caption="Corrélations entre capteurs — plus la couleur est intense, plus la liaison avec la puissance est forte")

    col3, col4 = st.columns(2)
    with col3:
        p = PLOTS_DIR / "hr_vs_power.png"
        if p.exists():
            st.image(str(p), caption="Fréquence cardiaque vs Puissance")
    with col4:
        p = PLOTS_DIR / "cadence_vs_power.png"
        if p.exists():
            st.image(str(p), caption="Cadence vs Puissance")

    p = PLOTS_DIR / "exemple_sortie.png"
    if p.exists():
        st.image(str(p), caption="Exemple d'une sortie complète — vitesse, fréquence cardiaque et puissance seconde par seconde")


def build_app() -> None:
    st.set_page_config(
        page_title="Capteur Virtuel de Puissance ⚡",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🏆  Résultats",
        "🎮  Essaie toi-même",
        "🔍  Explorer les données",
    ])

    with tab1:
        page_resultats()
    with tab2:
        page_demo()
    with tab3:
        page_donnees()


if __name__ == "__main__":
    build_app()
