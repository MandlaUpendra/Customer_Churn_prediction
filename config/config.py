import pathlib
import os
import churn_prediction


PACKAGE_ROOT = pathlib.Path(churn_prediction.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,'datasets')

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

MODEL_NAME = 'churn.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')