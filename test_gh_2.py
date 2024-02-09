from bioconfpred.data.split import RandomSplit
from bioconfpred.model import ComENetModel
data_split = RandomSplit() # default is the split number 0
model = ComENetModel.get_model_for_data_split(data_split)