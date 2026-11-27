from pathlib import Path
import sys

repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))

from dashboard import main


main()
