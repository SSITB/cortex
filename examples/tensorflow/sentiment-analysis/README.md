# Deploy a BERT sentiment analysis API

This example shows how to deploy a sentiment analysis classifier trained using [BERT](https://github.com/google-research/bert).

## Define a deployment

A `deployment` specifies a set of resources that are deployed together. An `api` makes our exported model available as a web service that can serve real-time predictions. This configuration will download the model from the `cortex-examples` S3 bucket, and will preprocess the payload and postprocess the inference with functions defined in `handler.py`.

```yaml
# cortex.yaml

- kind: deployment
  name: sentiment

- kind: api
  name: classifier
  tensorflow:
    model: s3://cortex-examples/tensorflow/sentiment-analysis/bert
    request_handler: handler.py
  tracker:
    model_type: classification
```

<!-- CORTEX_VERSION_MINOR -->
You can run the code that generated the exported BERT model [here](https://colab.research.google.com/github/cortexlabs/cortex/blob/master/examples/tensorflow/sentiment-analysis/bert.ipynb).

## Add request handling

The model requires tokenized input for inference, but the API should accept strings of natural language as input. It should also map the model’s integer predictions to the actual sentiment label. This can be implemented in a request handler file. Define a `pre_inference()` function to tokenize request payloads and a `post_inference()` function to map inference output to labels before responding to the client:

```python
# handler.py

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
```

## Deploy to AWS

`cortex deploy` takes the declarative configuration from `cortex.yaml` and creates it on the cluster:

```bash
$ cortex deploy

creating classifier
```

Behind the scenes, Cortex containerizes our implementation, makes it servable using Flask, exposes the endpoint with a load balancer, and orchestrates the workload on Kubernetes.

We can track the status of a deployment using `cortex get`:

```bash
$ cortex get classifier --watch

status   up-to-date   available   requested   last update   avg latency
live     1            1           1           8s            -
```

The output above indicates that one replica of the API was requested and is available to serve predictions. Cortex will automatically launch more replicas if the load increases and spin down replicas if there is unused capacity.

## Serve real-time predictions

We can use `curl` to test our prediction service:

```bash
$ cortex get classifier

endpoint: http://***.amazonaws.com/sentiment/classifier

$ curl http://***.amazonaws.com/sentiment/classifier \
    -X POST -H "Content-Type: application/json" \
    -d '{"review": "The movie was great!"}'

"positive"
```

Any questions? [chat with us](https://gitter.im/cortexlabs/cortex).
