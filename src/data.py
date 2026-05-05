"""Chargement et préparation des données pour l'évaluation des modèles."""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import DATA_DIR


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Charge le dataset préparé et retourne (X_train, X_test, y_train, y_test)."""

    chemin = DATA_DIR / "processed_dataset.csv"
    df = pd.read_csv(chemin)

    # On garde seulement les vraies sorties (ride_id numérique)
    df = df[df['ride_id'].apply(lambda x: str(x).isdigit())]

    # Les features qu'on utilise pour prédire la puissance
    features = [
        'hr', 'cad', 'alt',
        'vitesse_kmh', 'pente_pct', 'acceleration',
        'delta_hr', 'vitesse_moy_5s', 'hr_moy_5s',
        'cad_moy_5s', 'pente_moy_5s'
    ]

    # La variable qu'on veut prédire
    cible = 'power'

    # Séparation train / test : les 20 dernières sorties pour le test
    tous_les_rides = sorted(df['ride_id'].unique(), key=lambda x: int(x))
    rides_test  = tous_les_rides[-20:]
    rides_train = tous_les_rides[:-20]

    df_train = df[df['ride_id'].isin(rides_train)]
    df_test  = df[df['ride_id'].isin(rides_test)]

    X_train = df_train[features].reset_index(drop=True)
    X_test  = df_test[features].reset_index(drop=True)
    y_train = df_train[cible].reset_index(drop=True)
    y_test  = df_test[cible].reset_index(drop=True)

    return X_train, X_test, y_train, y_test
