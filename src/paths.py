from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RUNS_DIR = ROOT_DIR / ".runs"

SOURCE_TOKENIZER_FILE = ROOT_DIR / "source_tokenizer.json"
TARGET_TOKENIZER_FILE = ROOT_DIR / "target_tokenizer.json"
CONFIG_FILE = ROOT_DIR / "config.json"
