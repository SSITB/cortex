import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization, run_classifier

labels = ["negative", "positive"]

with tf.Graph().as_default():
    bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
    info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
        vocab_file, do_lower_case = sess.run([info["vocab_file"], info["do_lower_case"]])
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def pre_inference(sample, signature, metadata):
    input_example = run_classifier.InputExample(guid="", text_a=sample["review"], label=0)
    input_feature = run_classifier.convert_single_example(0, input_example, [0, 1], 128, tokenizer)
    return {"input_ids": [input_feature.input_ids]}


def post_inference(prediction, signature, metadata):
    return labels[prediction["labels"][0]]
