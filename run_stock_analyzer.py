from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(ROOT / "src" / "run_predictions.py")], cwd=ROOT, check=True)
