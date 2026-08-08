import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.recording.resolver import OfflineResolver
from core.base.logging_utils import set_verbose
from models.raw_recording import RawRecording

session_dir = ROOT / "debug/recordings/1785429486181"
raw = RawRecording.parse_file(session_dir / "raw_recording.json")
set_verbose(True)
resolver = OfflineResolver(raw, session_dir=session_dir, debug=True)
script = resolver.resolve()
print(f"Resolved {len(script.actions)} actions, operators={script.operators}")
