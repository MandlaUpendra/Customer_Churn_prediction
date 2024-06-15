import os
import pandas as pd
import joblib

from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from churn_prediction.config import config

def load_dataset(filename):
    filepath = os.path.join(config.DATAPATH,filename)
    _data = pd.read_csv(filepath)
    return _data

def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    joblib.dump(pipeline_to_save,save_path)
    print(f"Model has been saved using the name {config.MODEL_NAME}")