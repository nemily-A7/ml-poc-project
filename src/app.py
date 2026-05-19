from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import MODELS_DIR, PLOTS_DIR


STRAVA_ORANGE = "#FC4C02"
DARK_BG       = "#111111"
CARD_BG       = "#1C1C1E"
BORDER        = "rgba(255,255,255,0.08)"
TEXT_DIM      = "#8E8E93"

CSS = f"""
<style>
    .stApp {{
        background-color: {DARK_BG};
    }}

    /* Remove top padding */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: {CARD_BG};
        border-radius: 10px;
        border: 1px solid {BORDER};
        gap: 4px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        color: {TEXT_DIM};
        font-weight: 600;
        font-size: 0.875rem;
        padding: 8px 18px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {STRAVA_ORANGE} !important;
        color: white !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}

    /* Headings */
    h1, h2, h3, h4 {{ color: white; }}

    p, li {{ color: rgba(255,255,255,0.85); }}

    /* Streamlit info/warning overrides */
    .stAlert {{ border-radius: 10px; }}

    /* Custom stat card */
    .stat-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 22px 16px;
        text-align: center;
    }}
    .stat-number {{
        font-size: 2.4rem;
        font-weight: 800;
        color: {STRAVA_ORANGE};
        display: block;
        line-height: 1;
    }}
    .stat-label {{
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {TEXT_DIM};
        display: block;
        margin-top: 6px;
    }}
    .stat-sub {{
        font-size: 0.78rem;
        color: rgba(255,255,255,0.4);
        display: block;
        margin-top: 2px;
    }}

    /* Section header */
    .section-header {{
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: {TEXT_DIM};
        border-bottom: 1px solid {BORDER};
        padding-bottom: 8px;
        margin-bottom: 16px;
        margin-top: 28px;
    }}

    /* Activity header */
    .activity-title {{
        font-size: 2rem;
        font-weight: 800;
        color: white;
        line-height: 1.1;
    }}
    .activity-sub {{
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {TEXT_DIM};
        margin-top: 4px;
    }}

    /* Orange accent bar */
    .accent-bar {{
        height: 3px;
        background: {STRAVA_ORANGE};
        border-radius: 2px;
        margin: 20px 0;
    }}

    /* Zone badge */
    .zone-badge {{
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        text-align: center;
        width: 100%;
    }}

    /* Live dot */
    @keyframes blink {{
        0%, 100% {{ opacity: 1; }}
        50%       {{ opacity: 0.2; }}
    }}
    .live-dot {{
        display: inline-block;
        width: 8px; height: 8px;
        background: {STRAVA_ORANGE};
        border-radius: 50%;
        animation: blink 1.4s infinite;
        margin-right: 6px;
        vertical-align: middle;
    }}

    /* Caption text */
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {TEXT_DIM} !important; }}

    /* Hide streamlit chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}
</style>
"""


def stat_card(number: str, label: str, sub: str = "") -> str:
    sub_html = f'<span class="stat-sub">{sub}</span>' if sub else ""
    return f"""
    <div class="stat-card">
        <span class="stat-number">{number}</span>
        <span class="stat-label">{label}</span>
        {sub_html}
    </div>
    """


def section_header(text: str) -> None:
    st.markdown(f'<p class="section-header">{text}</p>', unsafe_allow_html=True)


def _zone(w: int) -> tuple[str, str, str]:
    if w < 100:
        return "Z1 — Récupération", "#48cae4", "Effort très léger · idéal pour récupérer entre deux séances"
    elif w < 180:
        return "Z2 — Endurance", "#52b788", "Allure confortable · la base de l'entraînement cycliste"
    elif w < 250:
        return "Z3 — Tempo", "#f9c74f", "Effort soutenu · on commence à bien souffler"
    elif w < 320:
        return "Z4 — Seuil lactique", "#f77f00", "Effort intense · la limite de ce qu'on peut tenir longtemps"
    else:
        return "Z5 — VO₂max", "#e63946", "Effort maximal · sprints et montées à bloc"


def _bar_chart(df: pd.DataFrame, col: str, title: str, suffix: str, y_max: float) -> go.Figure:
    shades = ["#FC4C02", "#E84400", "#C93A00"]
    fig = go.Figure(go.Bar(
        x=df["model_name"].tolist(),
        y=df[col].tolist(),
        marker_color=shades,
        marker_line_width=0,
        text=[f"{v}{suffix}" for v in df[col].tolist()],
        textposition="outside",
        textfont=dict(color="white", size=13, family="Arial Black"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_DIM, size=11), x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(range=[0, y_max], showgrid=True, gridcolor=BORDER, showticklabels=False),
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        margin=dict(t=40, b=10, l=10, r=10),
        height=260,
        showlegend=False,
    )
    return fig


def page_resultats() -> None:
    st.markdown(
        '<p class="activity-title">Capteur Virtuel de Puissance</p>'
        '<p class="activity-sub">Estimation algorithmique · XGBoost · 163 sorties · 1 athlète</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(stat_card("50%", "Précision R²", "meilleur modèle"), unsafe_allow_html=True)
    col2.markdown(stat_card("58 W", "Erreur moyenne", "MAE · XGBoost"), unsafe_allow_html=True)
    col3.markdown(stat_card("89 W", "Erreur RMSE", "penalise les pics"), unsafe_allow_html=True)
    col4.markdown(stat_card("1.62M", "Points d'entraînement", "1 ligne = 1 seconde"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f'<div class="stat-card" style="text-align:left">'
            f'<span style="color:{STRAVA_ORANGE}; font-weight:700; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em">Ce que ça veut dire</span><br><br>'
            f'<span style="color:rgba(255,255,255,0.85); font-size:0.95rem">'
            "Avec un cycliste pédalant réellement à <b>250 W</b>, l'algorithme estime entre "
            "<b>~192 W et ~308 W</b>. C'est imparfait — mais c'est entièrement gratuit, "
            "sans aucun capteur à 1&nbsp;000 €.</span></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div class="stat-card" style="text-align:left">'
            f'<span style="color:{TEXT_DIM}; font-weight:700; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em">Limite du modèle</span><br><br>'
            f'<span style="color:rgba(255,255,255,0.85); font-size:0.95rem">'
            "Le vent et l'aérodynamisme influencent énormément la puissance réelle, "
            "mais restent <b>invisibles</b> pour nos capteurs bas coût. "
            "C'est la principale source d'erreur.</span></div>",
            unsafe_allow_html=True,
        )

    section_header("Features · Importance des capteurs")
    chemin_imp = PLOTS_DIR / "feature_importance.png"
    if chemin_imp.exists():
        st.image(str(chemin_imp), width="stretch")
        st.caption("La vitesse et la pente sont de loin les indicateurs les plus prédictifs")

    section_header("Performance · Réel vs. Estimé — XGBoost")
    chemin_pred = PLOTS_DIR / "xgboost_reel_vs_predit.png"
    if chemin_pred.exists():
        st.image(str(chemin_pred), width="stretch")
        st.caption(
            "Chaque point = 1 seconde de pédalage · "
            "Ligne rouge = prédiction parfaite · "
            "Plus les points sont proches de la ligne, meilleure est l'estimation"
        )


def _appliquer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    delta_km   = d['km'].diff().fillna(0)
    delta_secs = d['secs'].diff().fillna(1).replace(0, 1)
    d['vitesse_kmh'] = (delta_km / delta_secs * 3600).clip(0, 120)

    delta_alt = d['alt'].diff().fillna(0)
    delta_m   = delta_km * 1000
    d['pente_pct'] = np.where(delta_m > 0, (delta_alt / delta_m) * 100, 0)
    d['pente_pct'] = d['pente_pct'].clip(-30, 30)

    d['acceleration'] = d['vitesse_kmh'].diff().fillna(0).clip(-10, 10)
    d['delta_hr']     = d['hr'].diff().fillna(0).clip(-10, 10)

    d['vitesse_moy_5s'] = d['vitesse_kmh'].rolling(window=5, min_periods=1).mean()
    d['hr_moy_5s']      = d['hr'].rolling(window=5, min_periods=1).mean()
    d['cad_moy_5s']     = d['cad'].rolling(window=5, min_periods=1).mean()
    d['pente_moy_5s']   = d['pente_pct'].rolling(window=5, min_periods=1).mean()
    return d


def _graphique_sortie(df: pd.DataFrame) -> go.Figure:
    t = df['secs'] / 60
    has_power = 'power' in df.columns

    rows = 4 if has_power else 3
    titles = (["Puissance (W)", "Fréquence cardiaque (bpm)", "Cadence (rpm)", "Altitude (m)"] if has_power
              else ["Fréquence cardiaque (bpm)", "Cadence (rpm)", "Altitude (m)"])

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, subplot_titles=titles)

    row = 1
    if has_power:
        fig.add_trace(go.Scatter(
            x=t, y=df['power'],
            line=dict(color=STRAVA_ORANGE, width=1), name="Puissance",
        ), row=row, col=1)
        row += 1

    fig.add_trace(go.Scatter(
        x=t, y=df['hr'],
        line=dict(color="#e63946", width=1), name="FC",
    ), row=row, col=1)
    row += 1

    fig.add_trace(go.Scatter(
        x=t, y=df['cad'],
        line=dict(color="#f9c74f", width=1), name="Cadence",
    ), row=row, col=1)
    row += 1

    fig.add_trace(go.Scatter(
        x=t, y=df['alt'],
        line=dict(color="#52b788", width=1), name="Altitude",
        fill="tozeroy", fillcolor="rgba(82,183,136,0.08)",
    ), row=row, col=1)

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=550, showlegend=False,
        margin=dict(t=40, b=20, l=60, r=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, title_text="Temps (minutes)", row=row, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER)
    return fig


def _afficher_resultat(puissance: int) -> None:
    zone_label, zone_color, zone_desc = _zone(puissance)

    section_header("Puissance moyenne estimée")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=puissance,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "WATTS — MOYENNE DE LA SORTIE",
               "font": {"color": TEXT_DIM, "size": 12, "family": "Arial"}},
        number={"font": {"color": zone_color, "size": 72, "family": "Arial Black"}},
        gauge={
            "axis": {
                "range": [0, 600],
                "tickvals": [0, 100, 180, 250, 320, 600],
                "ticktext": ["0", "100", "180", "250", "320", "600"],
                "tickcolor": TEXT_DIM,
                "tickfont": {"color": TEXT_DIM, "size": 10},
            },
            "bar": {"color": zone_color, "thickness": 0.2},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0,   100], "color": "rgba(72,202,228,0.10)"},
                {"range": [100, 180], "color": "rgba(82,183,136,0.10)"},
                {"range": [180, 250], "color": "rgba(249,199,79,0.10)"},
                {"range": [250, 320], "color": "rgba(247,127,0,0.10)"},
                {"range": [320, 600], "color": "rgba(230,57,70,0.10)"},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280,
                      margin=dict(t=20, b=0, l=40, r=40))
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f'<div style="background:{zone_color}18; border:2px solid {zone_color};'
        f' border-radius:14px; padding:20px 28px; margin:12px 0 20px 0">'
        f'<span style="color:{zone_color}; font-weight:800; font-size:1.2rem;'
        f' display:block">{zone_label}</span>'
        f'<span style="color:rgba(255,255,255,0.75); font-size:0.95rem;'
        f' margin-top:4px; display:block">{zone_desc}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    zones = [
        (100,  "Z1 · Récupération", "#48cae4"),
        (180,  "Z2 · Endurance",    "#52b788"),
        (250,  "Z3 · Tempo",        "#f9c74f"),
        (320,  "Z4 · Seuil",        "#f77f00"),
        (9999, "Z5 · VO₂max",       "#e63946"),
    ]
    cols = st.columns(5)
    for i, (seuil, label, color) in enumerate(zones):
        active = puissance < seuil and (i == 0 or puissance >= zones[i - 1][0])
        border = f"2px solid {color}" if active else f"1px solid {BORDER}"
        bg     = f"{color}30" if active else "rgba(0,0,0,0)"
        weight = "800" if active else "600"
        cols[i].markdown(
            f'<div style="background:{bg}; border:{border}; border-radius:10px;'
            f' padding:10px 6px; text-align:center">'
            f'<span style="color:{color}; font-weight:{weight};'
            f' font-size:0.78rem">{label}</span></div>',
            unsafe_allow_html=True,
        )

    section_header("Références · Profil du cycliste")
    refs = [
        ("150 W", "Cycliste débutant",  "balade tranquille",      0,   160),
        ("220 W", "Cycliste amateur",   "sortie entraînement",   160,   290),
        ("380 W", "Tour de France",     "moyenne sur une étape", 290,  9999),
        ("1 100 W", "Sprint max pro",   "effort de 2 secondes",   -1,    -1),
    ]
    rcols = st.columns(4)
    for i, (val, label, sub, smin, smax) in enumerate(refs):
        active = smin <= puissance < smax
        border = f"2px solid {zone_color}" if active else f"1px solid {BORDER}"
        bg     = f"{zone_color}18" if active else CARD_BG
        badge  = (f'<span style="display:inline-block; background:{zone_color};'
                  f' color:white; font-size:0.6rem; font-weight:700;'
                  f' padding:2px 8px; border-radius:20px;'
                  f' text-transform:uppercase; margin-top:6px">Votre niveau</span>'
                  if active else "")
        rcols[i].markdown(
            f'<div style="background:{bg}; border:{border}; border-radius:14px;'
            f' padding:22px 16px; text-align:center">'
            f'<span style="font-size:2.4rem; font-weight:800; color:{STRAVA_ORANGE};'
            f' display:block; line-height:1">{val}</span>'
            f'<span style="font-size:0.65rem; text-transform:uppercase;'
            f' letter-spacing:0.12em; color:{TEXT_DIM}; display:block;'
            f' margin-top:6px">{label}</span>'
            f'<span style="font-size:0.78rem; color:rgba(255,255,255,0.4);'
            f' display:block; margin-top:2px">{sub}</span>'
            f'{badge}</div>',
            unsafe_allow_html=True,
        )


def page_demo() -> None:
    st.markdown(
        '<p class="activity-title">'
        '<span class="live-dot"></span>Analyse de sortie</p>'
        '<p class="activity-sub">Charge un fichier CSV · estimation de la puissance · XGBoost</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    chemin_modele = MODELS_DIR / "xgboost.joblib"
    if not chemin_modele.exists():
        st.error("Modèle introuvable. Lance le notebook training.ipynb d'abord.")
        return

    modele = joblib.load(chemin_modele)

    section_header("Charger une sortie")
    fichier = st.file_uploader(
        "Fichier CSV (même format que les données du projet)",
        type="csv",
        label_visibility="collapsed",
    )

    if fichier is None:
        st.markdown(
            f'<div class="stat-card" style="text-align:center; padding:40px">'
            f'<span style="color:{TEXT_DIM}; font-size:1rem">'
            f'Dépose un fichier CSV pour commencer l\'analyse</span></div>',
            unsafe_allow_html=True,
        )
        st.session_state.pop("puissance_demo", None)
        return

    if st.session_state.get("demo_fichier") != fichier.name:
        st.session_state.pop("puissance_demo", None)
        st.session_state["demo_fichier"] = fichier.name

    df = pd.read_csv(fichier)
    cols_requises = ['hr', 'cad', 'alt', 'km', 'secs']
    manquantes = [c for c in cols_requises if c not in df.columns]
    if manquantes:
        st.error(f"Colonnes manquantes dans le fichier : {', '.join(manquantes)}")
        return

    section_header("Aperçu de la sortie")
    st.plotly_chart(_graphique_sortie(df), width="stretch")

    if st.button("Calculer la puissance moyenne", type="primary"):
        features = ['hr', 'cad', 'alt', 'vitesse_kmh', 'pente_pct', 'acceleration',
                    'delta_hr', 'vitesse_moy_5s', 'hr_moy_5s', 'cad_moy_5s', 'pente_moy_5s']
        df_enrichi = _appliquer_features(df).dropna(subset=features)
        preds = np.maximum(modele.predict(df_enrichi[features].values), 0)
        st.session_state["puissance_demo"] = max(0, round(float(preds.mean())))

    if "puissance_demo" in st.session_state:
        _afficher_resultat(st.session_state["puissance_demo"])


def page_donnees() -> None:
    st.markdown(
        '<p class="activity-title">Exploration des données</p>'
        '<p class="activity-sub">163 sorties · 1 athlète · 1.62 million de secondes</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    section_header("Activité · Exemple de sortie complète")
    p = PLOTS_DIR / "exemple_sortie.png"
    if p.exists():
        st.image(str(p), width="stretch")
        st.caption("Vitesse, fréquence cardiaque et puissance seconde par seconde sur une sortie")

    section_header("Distributions · Puissance et fréquence cardiaque")
    c1, c2 = st.columns(2)
    with c1:
        p = PLOTS_DIR / "distribution_puissance.png"
        if p.exists():
            st.image(str(p), width="stretch")
            st.caption("La plupart du temps, le cycliste pédale entre 100 et 300 W")
    with c2:
        p = PLOTS_DIR / "distribution_fc.png"
        if p.exists():
            st.image(str(p), width="stretch")
            st.caption("La fréquence cardiaque oscille surtout entre 120 et 175 bpm")

    section_header("Corrélations · Relations entre capteurs")
    p = PLOTS_DIR / "correlation_matrix.png"
    if p.exists():
        st.image(str(p), width="stretch")
        st.caption("Plus la couleur est intense, plus les deux variables sont liées")

    section_header("Scatter · Capteurs vs Puissance")
    c3, c4 = st.columns(2)
    with c3:
        p = PLOTS_DIR / "hr_vs_power.png"
        if p.exists():
            st.image(str(p), width="stretch")
            st.caption("Fréquence cardiaque vs Puissance")
    with c4:
        p = PLOTS_DIR / "cadence_vs_power.png"
        if p.exists():
            st.image(str(p), width="stretch")
            st.caption("Cadence vs Puissance")


def build_app() -> None:
    st.set_page_config(
        page_title="Virtual Power Meter",
        page_icon="🔶",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "  Données    ",
        "  Résultats  ",
        "  Démo live  ",
    ])
    with tab1:
        page_donnees()
    with tab2:
        page_resultats()
    with tab3:
        page_demo()


if __name__ == "__main__":
    build_app()
