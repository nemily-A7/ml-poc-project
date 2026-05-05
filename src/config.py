from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

MODELS = {
    "linear_regression": {
        "name": "Régression Linéaire",
        "description": "Modèle de base simple. Prédit la puissance comme une combinaison linéaire des features.",
        "path": MODELS_DIR / "linear_regression.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble de 50 arbres de décision. Capture les relations non-linéaires entre les capteurs et la puissance.",
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "xgboost": {
        "name": "XGBoost",
        "description": "Gradient boosting. Meilleur modèle du projet : R²=0.50 sur les 20 dernières sorties.",
        "path": MODELS_DIR / "xgboost.joblib",
    },
}
