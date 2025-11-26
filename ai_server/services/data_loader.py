from pathlib import Path
import pandas as pd

# Path to project root: ai_server/services → ai_server → SIMPPL/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to data folder: SIMPPL/data/processed/
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_posts():
    path = DATA_DIR / "clean_reddit_10cols.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {path.resolve()}")
    return pd.read_csv(path, parse_dates=["created_utc"])  # removed deprecated arg


def load_domain_scores():
    """Loads the domain credibility scores."""
    path = DATA_DIR / "domain_scores.csv"
    
    if not path.exists():
        print(f"WARNING: Domain scores file not found at {path}")
        return pd.DataFrame(columns=["domain", "count", "score", "reasons"])
        
    return pd.read_csv(path)
def load_author_fingerprints():
    path = DATA_DIR / "author_fingerprints.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {path.resolve()}")
    return pd.read_csv(path)


def load_top_url_cascade():
    path = DATA_DIR / "top_url_cascade.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {path.resolve()}")
    return pd.read_csv(path)
