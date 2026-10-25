
import sys
from pathlib import Path
import inspect

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from alphabuilder.src.logic.selfplay.storage import record_step
from alphabuilder.src.logic.selfplay import record_step as record_step_pkg

print("Function signature from storage:", inspect.signature(record_step))
print("Function signature from package:", inspect.signature(record_step_pkg))
