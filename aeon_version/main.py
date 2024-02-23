from aeon.classification.fcn import FCNClassifier

from utils.helper import *

fcn_model = FCNClassifier()

df = read_dataset("ArrowHead")