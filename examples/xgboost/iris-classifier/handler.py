import numpy as np

labels = ["iris-setosa", "iris-versicolor", "iris-virginica"]


def pre_inference(sample, signature, metadata):
    return {
        signature[0].name: [
            sample["sepal_length"],
            sample["sepal_width"],
            sample["petal_length"],
            sample["petal_width"],
        ]
    }


def post_inference(prediction, signature, metadata):
    predicted_class_id = prediction[0][0]
    return labels[predicted_class_id]
