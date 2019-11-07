import os
import numpy as np
import pickle
from dvc import api

labels = ["not-python", "python"]

ctx = {}


def init(model_path, metadata):
    ctx["model"] = pickle.loads(api.read(metadata["model_path"], metadata["dvc_repo"], mode="rb"))
    ctx["pipeline"] = pickle.loads(
        api.read(metadata["pipeline_path"], metadata["dvc_repo"], mode="rb")
    )


def predict(sample, metadata):
    input_arr = np.array([sample["text"].lower()], dtype="U")
    preprocessed_arr = ctx["pipeline"].transform(input_arr)
    predicted_index = ctx["model"].predict(preprocessed_arr)
    return labels[int(predicted_index.item())]
