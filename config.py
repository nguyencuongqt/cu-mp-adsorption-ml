from pathlib import Path


ROOT = Path(__file__).parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
MODEL_DIR = MODELS_DIR
REPORTS_DIR = OUTPUT_DIR / "reports"

DATA_FILE = DATA_DIR / "dataset.csv"
if not DATA_FILE.exists() and (DATA_DIR / "Cu_data.xlsx").exists():
    DATA_FILE = DATA_DIR / "Cu_data.xlsx"
if not DATA_FILE.exists() and (ROOT.parent / "Cu_data.xlsx").exists():
    DATA_FILE = ROOT.parent / "Cu_data.xlsx"

RANDOM_SEED = 42

TARGET = "qe"
CAT_COLS = ["AgS", "ReT", "AdC"]
NUM_COLS = ["Temp", "pH", "rpm", "Ce"]

TEST_SIZE = 0.20
N_FOLDS = 10

RF_PARAMS = {
    "n_estimators": 2000,
    "max_features": 20,
    "min_samples_leaf": 4,
}

XGB_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

LGBM_PARAMS = {
    "n_estimators": 1000,
    "num_leaves": 31,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

ENET_PARAMS = {
    "alpha": 0.1,
    "l1_ratio": 0.5,
    "max_iter": 10000,
}

QRF_LOWER = 0.05
QRF_UPPER = 0.95

FAF_CENTRAL = 0.012
FAF_BOOTSTRAP_ITERS = 2000

GLOBAL_DISCHARGE_M3_YR = 37_411e9
GLOBAL_MP_MASS_MG_M3 = 6.0
GLOBAL_CU_MG_L = 8.77e-3

PH_SCENARIOS = [6.0, 7.0, 8.0]

RIVER_DEFAULTS = {
    "pH": 7.0,
    "Temp": 20.0,
    "rpm": 150.0,
    "AgS": "Aged",
    "AdC": "Single-ion",
}

PAPERS_DIR = DATA_DIR / "papers"
DATASET_CSV = DATA_DIR / "dataset.csv"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
RIVER_DATA_FILE = DATA_DIR / "river_data.csv"
TEST_CASES_FILE = DATA_DIR / "test_cases.csv"
PAPER_METADATA_FILE = DATA_DIR / "paper_metadata.csv"

FAISS_INDEX_FILE = MODELS_DIR / "faiss_index.bin"
CHUNKS_METADATA_FILE = MODELS_DIR / "chunks_metadata.pkl"
KNOWLEDGE_GRAPH_FILE = MODELS_DIR / "knowledge_graph.gpickle"
QRF_MODEL_FILE = MODELS_DIR / "qrf_model.joblib"
FAF_RESULTS_FILE = REPORTS_DIR / "faf_results.csv"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_BATCH_SIZE = 64
CHUNK_TOKENS = 300
CHUNK_OVERLAP = 50

DOMAIN = {
    "Temp": (0.0, 80.0),
    "pH": (2.0, 12.0),
    "rpm": (0.0, 400.0),
    "Ce": (0.0, 300.0),
    "AgS": ["Aged", "Virgin", "aged", "virgin"],
    "ReT": ["CPE", "HPE", "LPE", "PA", "PBAT", "PE", "PET", "PLA", "PMMA", "PP", "PS", "PVC", "TPU"],
    "AdC": ["Mixed", "Single-ion", "mixed", "single-ion"],
}
