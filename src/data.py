from __future__ import annotations

from typing import Any

import pandas as pd

from config import DATA_DIR


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    chemin = DATA_DIR / "processed_dataset.csv"
    df = pd.read_csv(chemin)

    df = df[df['ride_id'].apply(lambda x: str(x).isdigit())]

    features = [
        'hr', 'cad', 'alt',
        'vitesse_kmh', 'pente_pct', 'acceleration',
        'delta_hr', 'vitesse_moy_5s', 'hr_moy_5s',
        'cad_moy_5s', 'pente_moy_5s'
    ]

    cible = 'power'

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
