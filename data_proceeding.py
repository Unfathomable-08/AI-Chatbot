import numpy as np
# from data_load import dataset

def dialog_separator(dataset):
    pairs = []
    for data in dataset:
        dialogs = data["dialog"]
        acts = data["act"] 
        for i in range(len(dialogs) - 1):
            pairs.append((dialogs[i], acts[i + 1])) # Current dialog and next act
    
    return pairs

# train = dialog_separator(dataset["train"])