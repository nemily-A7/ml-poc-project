# Documentation du projet — Capteur Virtuel de Puissance

Ce document explique ce que fait chaque fichier du projet, ligne par ligne si nécessaire.

---

## Vue d'ensemble

Le projet suit un pipeline en 3 étapes :

```
Données brutes (163 CSV)
        ↓
  exploration.ipynb      → on regarde et comprend les données
        ↓
feature_engineering.ipynb → on crée des variables utiles pour le modèle
        ↓
    training.ipynb        → on entraîne 3 modèles et on compare les résultats
        ↓
       app.py             → on affiche les résultats dans une interface web
```

---

## Structure des fichiers

```
ml-poc-project/
├── data/
│   ├── 1.csv, 2.csv ... 163.csv   → données brutes (une sortie vélo par fichier)
│   └── processed_dataset.csv      → données enrichies générées par feature_engineering
├── models/
│   ├── linear_regression.joblib   → modèle de régression linéaire sauvegardé
│   ├── random_forest.joblib       → modèle Random Forest sauvegardé
│   └── xgboost.joblib             → modèle XGBoost sauvegardé
├── notebooks/
│   ├── exploration.ipynb          → analyse exploratoire des données
│   ├── feature_engineering.ipynb  → création des variables (features)
│   └── training.ipynb             → entraînement et évaluation des modèles
├── plots/                         → graphiques générés par les notebooks
├── results/
│   └── model_metrics.csv          → tableau des performances des 3 modèles
└── src/
    ├── config.py                  → chemins et constantes du projet
    ├── data.py                    → chargement et découpage des données
    ├── metrics.py                 → calcul des métriques de performance
    └── app.py                     → interface Streamlit
```

---

## `src/config.py` — Les chemins du projet

Ce fichier définit tous les chemins vers les dossiers du projet, pour ne pas les réécrire partout.

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
```
`Path(__file__)` = le chemin vers `config.py` lui-même.
`.parent.parent` = on remonte deux niveaux → on arrive à la racine du projet.

```python
DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR  = PROJECT_ROOT / "plots"
...
```
On construit les chemins vers chaque dossier à partir de la racine.
Le `/` ici n'est pas une division — c'est la syntaxe de `pathlib` pour assembler des chemins.

```python
for dir in [DATA_DIR, LOGS_DIR, ...]:
    dir.mkdir(exist_ok=True)
```
On crée automatiquement tous les dossiers s'ils n'existent pas encore.
`exist_ok=True` = pas d'erreur si le dossier existe déjà.

```python
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"
```
Chemin vers le fichier CSV qui contient les scores des 3 modèles.

```python
MODELS = {
    "xgboost": {
        "name": "XGBoost",
        "path": MODELS_DIR / "xgboost.joblib",
        ...
    }
}
```
Un dictionnaire qui décrit chaque modèle disponible — utilisé par l'app pour savoir où chercher les fichiers.

---

## `notebooks/exploration.ipynb` — Comprendre les données

**Objectif :** charger tous les CSV et produire des graphiques pour comprendre ce qu'on a.

### Chargement des données

lien des données: https://osf.io/6hfpz/files/9c4d8

```python
tous_les_fichiers = [f for f in os.listdir(data_folder)
                     if f.endswith('.csv') and f != 'processed_dataset.csv']
```
On liste tous les fichiers `.csv` dans le dossier `data/`, en excluant `processed_dataset.csv`
(ce fichier est généré plus tard, on ne veut pas l'inclure dans les données brutes).

```python
for fichier in tous_les_fichiers:
    df_temp = pd.read_csv(chemin)
    df_temp['ride_id'] = fichier.replace('.csv', '')
    liste_df.append(df_temp)

df = pd.concat(liste_df, ignore_index=True)
```
On charge chaque fichier CSV dans un DataFrame, on lui ajoute une colonne `ride_id` (ex: `"42"`)
pour savoir de quelle sortie vient chaque ligne, puis on fusionne tout en un seul grand DataFrame.

### Graphiques générés

| Graphique | Ce qu'il montre |
|-----------|-----------------|
| `distribution_puissance.png` | Combien de secondes à chaque niveau de puissance |
| `distribution_fc.png` | Combien de secondes à chaque fréquence cardiaque |
| `correlation_matrix.png` | À quel point chaque capteur est lié à la puissance |
| `hr_vs_power.png` | Nuage de points FC vs Puissance |
| `cadence_vs_power.png` | Nuage de points Cadence vs Puissance |
| `exemple_sortie.png` | Puissance, FC et altitude d'une sortie complète |
| `distribution_durees.png` | Durée des 163 sorties en minutes |

```python
puissance_active = df[df['power'] > 0]['power']
```
On filtre les lignes où `power > 0` car quand le cycliste est à l'arrêt, la puissance est 0
et cela fausserait les statistiques.

---

## `notebooks/feature_engineering.ipynb` — Créer des variables utiles

**Objectif :** les CSV bruts ont `hr`, `cad`, `alt`, `km`, `power`.
Ce n'est pas suffisant pour un bon modèle. On va créer des variables supplémentaires
(appelées *features*) qui vont aider le modèle à mieux prédire la puissance.

### La fonction `calculer_features(df, ride_id)`

Cette fonction prend une sortie brute et retourne la même sortie enrichie de nouvelles colonnes.

#### Vitesse en km/h

```python
delta_km   = d['km'].diff().fillna(0)
delta_secs = d['secs'].diff().fillna(1).replace(0, 1)
d['vitesse_kmh'] = (delta_km / delta_secs * 3600).clip(0, 120)
```
- `.diff()` = différence entre une ligne et la précédente (ex: km[t] - km[t-1])
- `delta_km / delta_secs` = distance parcourue par seconde → on multiplie par 3600 pour avoir des km/h
- `.clip(0, 120)` = on limite entre 0 et 120 km/h pour supprimer les valeurs aberrantes (GPS imprécis)

#### Pente en pourcentage

```python
delta_alt = d['alt'].diff().fillna(0)
delta_m   = delta_km * 1000
d['pente_pct'] = np.where(delta_m > 0, (delta_alt / delta_m) * 100, 0)
d['pente_pct'] = d['pente_pct'].clip(-30, 30)
```
- La pente = variation d'altitude / distance horizontale × 100
- `np.where(delta_m > 0, ..., 0)` = on ne calcule la pente que si on a avancé
  (si distance = 0 c'est qu'on est à l'arrêt, pas de sens de calculer une pente)
- `.clip(-30, 30)` = on limite entre -30% et +30% pour éliminer les erreurs GPS

#### Accélération et variation de FC

```python
d['acceleration'] = d['vitesse_kmh'].diff().fillna(0).clip(-10, 10)
d['delta_hr']     = d['hr'].diff().fillna(0).clip(-10, 10)
```
- `acceleration` = changement de vitesse entre deux secondes consécutives
- `delta_hr` = changement de fréquence cardiaque entre deux secondes
  (capture l'inertie cardiaque — le cœur ne réagit pas instantanément à l'effort)

#### Moyennes glissantes sur 5 secondes

```python
d['vitesse_moy_5s'] = d['vitesse_kmh'].rolling(window=5, min_periods=1).mean()
d['hr_moy_5s']      = d['hr'].rolling(window=5, min_periods=1).mean()
d['cad_moy_5s']     = d['cad'].rolling(window=5, min_periods=1).mean()
d['pente_moy_5s']   = d['pente_pct'].rolling(window=5, min_periods=1).mean()
```
- `.rolling(window=5)` = pour chaque seconde, on calcule la moyenne des 5 dernières secondes
- Cela lisse les valeurs et réduit le bruit du signal GPS/capteurs

### Résultat final

```python
df_final.to_csv('../data/processed_dataset.csv', index=False)
```
On sauvegarde le DataFrame enrichi (1.62 millions de lignes, 15 colonnes) dans un CSV.
Ce fichier sera utilisé par le notebook d'entraînement.

---

## `notebooks/training.ipynb` — Entraîner les modèles

**Objectif :** utiliser les données enrichies pour entraîner 3 modèles différents
et comparer leurs performances.

### Découpage train / test

```python
tous_les_rides = sorted([...], key=lambda x: int(x))
rides_test  = tous_les_rides[-20:]   # les 20 dernières sorties
rides_train = tous_les_rides[:-20]   # toutes les autres (143 sorties)
```
On sépare par sorties entières (pas par lignes aléatoires) pour simuler une situation réelle :
le modèle est entraîné sur les premières sorties et testé sur les suivantes.
C'est plus honnête que de mélanger les données au hasard.

### Les features utilisées

```python
features = ['hr', 'cad', 'alt', 'vitesse_kmh', 'pente_pct',
            'acceleration', 'delta_hr', 'vitesse_moy_5s',
            'hr_moy_5s', 'cad_moy_5s', 'pente_moy_5s']
cible = 'power'
```
- **features** = les colonnes qu'on donne au modèle comme "entrées" (ce qu'il voit)
- **cible** = ce qu'on veut prédire (la puissance en watts)

### Les 3 modèles

#### Régression Linéaire

```python
modele_lineaire = LinearRegression()
modele_lineaire.fit(X_train, y_train)
```
Le modèle le plus simple : il cherche des coefficients pour une équation du type :
`puissance = a × hr + b × vitesse + c × pente + ...`
Rapide, interprétable, mais limité car la réalité n'est pas linéaire.

**Résultat : R² = 0.40**

#### Random Forest

```python
modele_rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
modele_rf.fit(X_train, y_train)
```
- 50 arbres de décision entraînés chacun sur une partie aléatoire des données
- La prédiction finale = la moyenne des 50 arbres
- `n_jobs=-1` = utilise tous les cœurs du processeur pour aller plus vite
- `random_state=42` = graine aléatoire fixe pour que les résultats soient reproductibles

**Résultat : R² = 0.47**

#### XGBoost

```python
modele_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
modele_xgb.fit(X_train, y_train)
```
- 100 arbres entraînés séquentiellement : chaque arbre corrige les erreurs du précédent
- `learning_rate=0.1` = chaque correction est modérée pour ne pas sur-corriger
- Plus puissant que le Random Forest car il apprend de ses erreurs

**Résultat : R² = 0.50** ← meilleur modèle

### Sauvegarde des modèles

```python
joblib.dump(modele_xgb, '../models/xgboost.joblib')
```
`joblib` sérialise le modèle entraîné dans un fichier binaire.
On peut ensuite le recharger avec `joblib.load(...)` sans réentraîner.

---

## `src/data.py` — Charger et découper les données

Ce fichier contient une fonction réutilisable pour charger les données depuis `processed_dataset.csv`
et retourner les 4 ensembles nécessaires à l'entraînement.

```python
def load_dataset_split():
    df = pd.read_csv(DATA_DIR / "processed_dataset.csv")
    df = df[df['ride_id'].apply(lambda x: str(x).isdigit())]
```
Le filtre `isdigit()` exclut les lignes dont le `ride_id` n'est pas un nombre.
Cela évite d'inclure accidentellement le fichier `processed_dataset.csv` lui-même
s'il avait été chargé comme une sortie.

```python
tous_les_rides = sorted(df['ride_id'].unique(), key=lambda x: int(x))
rides_test  = tous_les_rides[-20:]
rides_train = tous_les_rides[:-20]
```
Même découpage chronologique que dans le notebook.

```python
return X_train, X_test, y_train, y_test
```
La fonction retourne les 4 tableaux prêts à l'emploi.

---

## `src/metrics.py` — Calculer les performances

```python
def compute_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2)}
```

| Métrique | Formule | Ce que ça mesure |
|----------|---------|-----------------|
| **R²** | variance expliquée / variance totale | Part de la puissance que le modèle explique. 1.0 = parfait, 0 = inutile |
| **MAE** | moyenne de \|réel - prédit\| | Erreur moyenne en watts. Le plus facile à comprendre |
| **RMSE** | racine de la moyenne des erreurs² | Comme le MAE mais pénalise plus les grosses erreurs |

---

## `src/app.py` — L'interface Streamlit

Ce fichier génère l'application web. Il est découpé en 4 fonctions principales.

### `build_app()` — Point d'entrée

```python
def build_app():
    st.set_page_config(...)
    st.markdown(CSS, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Résultats", "Démo live", "Données"])
    with tab1: page_resultats()
    with tab2: page_demo()
    with tab3: page_donnees()
```
Cette fonction est appelée en premier. Elle configure la page et crée les 3 onglets.
Chaque onglet appelle une fonction différente.

### `page_resultats()` — Onglet Résultats

Affiche :
- Les 4 stat-cards (R², MAE, RMSE, volume de données)
- 3 graphiques barres Plotly comparant les 3 modèles
- L'image `xgboost_reel_vs_predit.png`
- L'image `feature_importance.png`

```python
df = pd.read_csv(MODEL_METRICS_FILE)
```
Il lit les métriques depuis `results/model_metrics.csv` au lieu de les coder en dur —
ainsi si on relance les modèles avec de meilleures performances, l'app se met à jour automatiquement.

### `page_demo()` — Onglet Démo live

```python
modele = joblib.load(chemin_modele)
```
On charge le modèle XGBoost depuis le fichier `.joblib`.

```python
hr = st.slider("Fréquence cardiaque", 60, 200, 150)
...
X = np.array([[hr, cad, alt, vitesse, pente, 0.0, 0.0, vitesse, hr, cad, pente]])
puissance = max(0, round(modele.predict(X)[0]))
```
À chaque mouvement d'un curseur, Streamlit réexécute tout le code.
On reconstruit le tableau `X` avec les valeurs des curseurs et on appelle `modele.predict(X)`.
`max(0, ...)` empêche d'afficher une puissance négative.

La jauge Plotly change de couleur automatiquement selon la zone d'effort calculée par `_zone(puissance)`.

### `page_donnees()` — Onglet Données

Affiche simplement les images PNG générées par les notebooks d'exploration.

```python
p = PLOTS_DIR / "distribution_puissance.png"
if p.exists():
    st.image(str(p), width="stretch")
```
On vérifie que le fichier existe avant de l'afficher — si quelqu'un n'a pas encore
exécuté le notebook, on n'affiche pas d'erreur.

---

## Résumé des résultats

| Modèle | R² | MAE | RMSE |
|--------|-----|-----|------|
| Régression Linéaire | 0.40 | 68 W | 98 W |
| Random Forest | 0.47 | 61 W | 92 W |
| **XGBoost** | **0.50** | **58 W** | **89 W** |

**Le R² de 0.50 signifie que le modèle explique 50% de la variance de la puissance.**
Les 50% restants viennent principalement du vent et de l'aérodynamisme,
deux facteurs impossibles à mesurer avec des capteurs bas coût.
