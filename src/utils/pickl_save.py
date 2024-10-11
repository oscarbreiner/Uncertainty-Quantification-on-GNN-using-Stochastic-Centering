import pickle
import os
from src.data.data_loader import Dataset

def save_data_to_pickle(name, dataset: Dataset, logits, ood_mask, all_logits):
    root = "~/stochastic-centering-for-uncertainty-estimation-on-graphs/output"
    file_path = os.path.expanduser(os.path.join(root, f"{name}.pkl")) 

    data = dataset.cpu()

    output = {
        "ood_mask": ood_mask,
        "node_features": data.x,
        "logits": logits,
        "y": data.y,
        "all_logits": all_logits
    }

    with open(file_path, "wb") as file:
        pickle.dump(output, file)