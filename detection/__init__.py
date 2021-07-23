from pathlib import Path
from .model import *
import os
PACKAGE_PATH = Path(__file__).parents[1]

BASE_TRAINING_PATH = os.path.join(PACKAGE_PATH, 'detection', 'training')
BASE_DATA_PATH = os.path.join(PACKAGE_PATH, 'data')
