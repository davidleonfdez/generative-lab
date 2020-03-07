import os
import sys


root_lib_path = os.path.abspath(os.path.join(os.pardir, os.pardir, os.pardir, os.pardir))
if root_lib_path not in sys.path:
    sys.path.insert(0, root_lib_path)

