from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "data/raw": ["*.csv", "*.json"],
    "data/processed": ["*.csv", "*.json"],
    "artifacts": ["*.joblib", "*.json"],
    "reports": ["metrics.json", "test_predictions.csv", "technical_report.pdf"],
    "reports/figures": ["*.png"],
}

for directory, patterns in PATTERNS.items():
    folder = ROOT / directory
    for pattern in patterns:
        for path in folder.glob(pattern):
            if path.is_file():
                path.unlink()

print("Generated artifacts removed.")

