# conftest.py
import sys
import os

# Calculate the absolute path to the "src" directory
repo_root = os.path.dirname(__file__)
src_dir = os.path.join(repo_root, "src")

# Insert "src/" at the front of sys.path so that "import common..." works
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)