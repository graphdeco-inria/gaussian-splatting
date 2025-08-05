import os
import sys

# Allow relative imports like `from utils...` to work by appending parent directory
_module_path = os.path.dirname(__file__)
sys.path.insert(0, _module_path)