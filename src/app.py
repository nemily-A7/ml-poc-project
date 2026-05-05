"""Application Streamlit - Capteur Virtuel de Puissance Cycliste."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from config import DATA_DIR, MODEL_METRICS_FILE, MODELS_DIR, PLOTS_DIR


def build_app() -> None:
    """Construit et affiche l'application Streamlit du projet."""

    st.set_page_config(page_title="Capteur Virtuel de Puissance", layout="wide")

    # ------------------------------------------------------------------ #
    # Barre de navigation (sidebar)
    # ------------------------------------------------------------------ #
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à", [
        "Présentation du projet",
        "Exploration des données",
        "Résultats des modèles",
        "Démo interactive",
    ])

    # ------------------------------------------------------------------ #
    # PAGE 1 : Présentation
    # ------------------------------------------------------------------ #
    if page == "Présentation du projet":
        st.title("Capteur Virtuel de Puissance Cycliste")
        st.subheader("Estimer la puissance sans capteur matériel grâce au Machine Learning")

        st.markdown("""
        ### Le problème

        En cyclisme, la **puissance mécanique (watts)** est la métrique la plus fiable
        pour mesurer et prescrire l'entraînement. Contrairement à la fréquence cardiaque,
        elle reflète l'effort réel en temps réel, sans délai ni dérive.

        Le problème : un capteur de puissance coûte entre **300 et 1500€**,
        ce qui le rend inaccessible pour la majorité des cyclistes amateurs.

        ### Notre solution

        On entraîne un modèle de Machine Learning sur les données historiques d'un
        cycliste qui possédait un vrai capteur de puissance. Le modèle apprend à
        reproduire la puissance à partir de capteurs bas coût :

        - Cardiofréquencemètre
        - Capteur de vitesse et de cadence
        - Altimètre barométrique

        Une fois entraîné, ce modèle devient le **"jumeau numérique"** du cycliste :
        il peut estimer la puissance sur n'importe quelle sortie, sans capteur matériel.

        ### La physique derrière le modèle

        La puissance nécessaire pour avancer en vélo suit cette équation :
        """)

        st.latex(r"""
        P_{meca} = \left( 0.5 \cdot \rho \cdot S \cdot C_x \cdot V_a^2
                        + C_r \cdot m \cdot g \cdot \cos(\alpha)
                        + m \cdot g \cdot \sin(\alpha) \right) \cdot V_d
        """)

        st.markdown("""
        Le défi : des paramètres comme le vent ($V_a$) ou le coefficient aérodynamique
        ($C_x$) sont impossibles à mesurer sans instrumentation spécialisée.
        Le Machine Learning compense cette lacune en apprenant des patterns cachés.

        ### Données utilisées

        - **163 sorties vélo** d'un seul athlète (homme, né en 1998)
        - **1.62 million de lignes** — une ligne par seconde
        - Entraînement sur les **143 premières sorties**, test sur les **20 dernières**
        """)

    # ------------------------------------------------------------------ #
    # PAGE 2 : Exploration des données
    # ------------------------------------------------------------------ #
    elif page == "Exploration des données":
        st.title("Exploration des données")

        # Graphiques sauvegardés pendant l'EDA
        col1, col2 = st.columns(2)

        with col1:
            chemin_power = PLOTS_DIR / "distribution_puissance.png"
            if chemin_power.exists():
                st.image(str(chemin_power), caption="Distribution de la puissance")
            else:
                st.info("Exécute le notebook exploration.ipynb pour générer les graphiques.")

        with col2:
            chemin_fc = PLOTS_DIR / "distribution_fc.png"
            if chemin_fc.exists():
                st.image(str(chemin_fc), caption="Distribution de la fréquence cardiaque")

        chemin_corr = PLOTS_DIR / "correlation_matrix.png"
        if chemin_corr.exists():
            st.image(str(chemin_corr), caption="Matrice de corrélation")

        col3, col4 = st.columns(2)
        with col3:
            chemin_hr = PLOTS_DIR / "hr_vs_power.png"
            if chemin_hr.exists():
                st.image(str(chemin_hr), caption="FC vs Puissance")

        with col4:
            chemin_cad = PLOTS_DIR / "cadence_vs_power.png"
            if chemin_cad.exists():
                st.image(str(chemin_cad), caption="Cadence vs Puissance")

        chemin_sortie = PLOTS_DIR / "exemple_sortie.png"
        if chemin_sortie.exists():
            st.image(str(chemin_sortie), caption="Exemple d'une sortie complète")

        st.markdown("""
        ### Features créées (Feature Engineering)

        | Feature | Description |
        |---------|-------------|
        | `vitesse_kmh` | Vitesse calculée depuis la distance GPS |
        | `pente_pct` | Pente calculée depuis l'altitude et la distance |
        | `acceleration` | Variation de vitesse par seconde |
        | `delta_hr` | Variation de FC (capture l'inertie cardiaque) |
        | `vitesse_moy_5s` | Moyenne glissante sur 5 secondes |
        | `hr_moy_5s` | Moyenne glissante de la FC |
        | `cad_moy_5s` | Moyenne glissante de la cadence |
        | `pente_moy_5s` | Moyenne glissante de la pente |
        """)

    # ------------------------------------------------------------------ #
    # PAGE 3 : Résultats des modèles
    # ------------------------------------------------------------------ #
    elif page == "Résultats des modèles":
        st.title("Comparaison des modèles")

        if MODEL_METRICS_FILE.exists():
            df_metrics = pd.read_csv(MODEL_METRICS_FILE)
            st.dataframe(df_metrics, use_container_width=True)

            # Graphique comparatif en barres
            modeles = df_metrics['model_name'].tolist()

            fig, axes = plt.subplots(1, 3, figsize=(14, 5))

            for ax, metrique, couleur, titre in zip(
                axes,
                ['r2', 'mae', 'rmse'],
                ['steelblue', 'tomato', 'seagreen'],
                ['R² (plus haut = mieux)', 'MAE en watts (plus bas = mieux)', 'RMSE en watts (plus bas = mieux)']
            ):
                valeurs = df_metrics[metrique].tolist()
                ax.bar(modeles, valeurs, color=couleur)
                ax.set_title(titre)
                ax.set_ylabel(metrique.upper())
                ax.tick_params(axis='x', rotation=15)

            plt.tight_layout()
            st.pyplot(fig)

        else:
            st.warning("Lance `python scripts/main.py` pour générer les résultats.")

        # Graphique réel vs prédit
        chemin_pred = PLOTS_DIR / "xgboost_reel_vs_predit.png"
        if chemin_pred.exists():
            st.image(str(chemin_pred), caption="XGBoost : Puissance réelle vs Prédite")

        # Importance des features
        chemin_imp = PLOTS_DIR / "feature_importance.png"
        if chemin_imp.exists():
            st.image(str(chemin_imp), caption="Importance des features - XGBoost")

        st.markdown("""
        ### Interprétation

        - Le **R²** mesure la part de variance expliquée (1.0 = parfait, 0 = inutile)
        - Le **MAE** est l'erreur moyenne en watts — combien on se trompe en moyenne
        - Le **RMSE** pénalise plus les grosses erreurs (sprints, montées abruptes)

        XGBoost obtient le meilleur score avec R²≈0.50, soit une erreur moyenne
        d'environ 58 watts. Les limites principales sont l'absence de données
        de vent et la qualité du signal d'altitude GPS.
        """)

    # ------------------------------------------------------------------ #
    # PAGE 4 : Démo interactive
    # ------------------------------------------------------------------ #
    elif page == "Démo interactive":
        st.title("Démo : Estimer la puissance")
        st.write("Entre les valeurs de tes capteurs pour obtenir une estimation de la puissance.")

        # Chargement du meilleur modèle
        chemin_modele = MODELS_DIR / "xgboost.joblib"
        if not chemin_modele.exists():
            st.error("Modèle introuvable. Lance d'abord le notebook training.ipynb.")
            return

        modele = joblib.load(chemin_modele)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Capteurs")
            hr     = st.slider("Fréquence cardiaque (bpm)", 60, 200, 150)
            cad    = st.slider("Cadence de pédalage (rpm)", 0, 120, 85)
            alt    = st.slider("Altitude (m)", 0, 2500, 200)
            vitesse = st.slider("Vitesse (km/h)", 0, 80, 30)
            pente  = st.slider("Pente (%)", -15, 20, 0)

        with col2:
            st.subheader("Valeurs calculées automatiquement")
            # Valeurs dérivées simplifiées pour la démo
            acceleration  = 0.0
            delta_hr      = 0.0
            vitesse_moy   = vitesse
            hr_moy        = float(hr)
            cad_moy       = float(cad)
            pente_moy     = float(pente)

            st.write(f"Accélération : {acceleration} km/h/s")
            st.write(f"Variation FC : {delta_hr} bpm/s")
            st.write(f"Moy. vitesse 5s : {vitesse_moy:.1f} km/h")

        # Prédiction
        X = np.array([[hr, cad, alt, vitesse, pente,
                        acceleration, delta_hr,
                        vitesse_moy, hr_moy, cad_moy, pente_moy]])

        puissance_estimee = modele.predict(X)[0]
        puissance_estimee = max(0, round(puissance_estimee))

        st.markdown("---")
        st.metric(
            label="Puissance estimée",
            value=f"{puissance_estimee} W",
            help="Estimation du modèle XGBoost basée sur tes capteurs"
        )

        # Interprétation simple
        if puissance_estimee < 100:
            zone = "Zone 1 — Récupération active"
        elif puissance_estimee < 180:
            zone = "Zone 2 — Endurance de base"
        elif puissance_estimee < 250:
            zone = "Zone 3 — Tempo"
        elif puissance_estimee < 320:
            zone = "Zone 4 — Seuil lactique"
        else:
            zone = "Zone 5+ — Effort intense / VO2max"

        st.info(f"Zone d'effort estimée : **{zone}**")


if __name__ == "__main__":
    build_app()
