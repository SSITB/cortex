# Development Environment

## Prerequisites

1. Go (>=1.12.9)
1. Docker
1. eksctl
1. kubectl

## Cortex Dev Environment

Clone the project:

```bash
git clone https://github.com/cortexlabs/cortex.git
cd cortex
```

Run the tests:

```bash
make test
```

Create the AWS Elastic Container Registry:

```bash
make registry-create
```

Take note of the registry URL, this will be needed shortly.

Create the S3 buckets:

```bash
aws s3 mb s3://cortex-cluster-<your_name>
aws s3 mb s3://cortex-cli-<your_name>  # if you'll be uploading your compiled CLI
```

### Configuration

Make the config folder:

```bash
mkdir dev/config
```

Create `dev/config/cluster.yaml`. Paste the following config, and update `cortex_bucket`, `cortex_region`, `aws_access_key_id`, `aws_secret_access_key`, and all registry URLs accordingly:

```yaml
aws_access_key_id: ***
aws_secret_access_key: ***

instance_type: m5.large
min_instances: 2
max_instances: 5
bucket: cortex-cluster-<your_name>
region: us-west-2
log_group: cortex
cluster_name: cortex
telemetry: false

image_predictor_serve: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/predictor-serve:latest
image_predictor_serve_gpu: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/predictor-serve-gpu:latest
image_tf_serve: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/tf-serve:latest
image_tf_serve_gpu: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/tf-serve-gpu:latest
image_tf_api: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/tf-api:latest
image_onnx_serve: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/onnx-serve:latest
image_onnx_serve_gpu: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/onnx-serve-gpu:latest
image_operator: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/operator:latest
image_manager: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/manager:latest
image_downloader: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/downloader:latest
image_cluster_autoscaler: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/cluster-autoscaler:latest
image_metrics_server: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/metrics-server:latest
image_nvidia: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/nvidia:latest
image_fluentd: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/fluentd:latest
image_statsd: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/statsd:latest
image_istio_proxy: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/istio-proxy:latest
image_istio_pilot: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/istio-pilot:latest
image_istio_citadel: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/istio-citadel:latest
image_istio_galley: XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/cortexlabs/istio-galley:latest
```

Create `dev/config/build.sh`. Paste the following config, and update `CLI_BUCKET_NAME`, `CLI_BUCKET_REGION`, `REGISTRY_URL`, and `REGISTRY_REGION` accordingly:

```bash
export VERSION="master"

export REGISTRY_URL="XXXXXXXX.dkr.ecr.us-west-2.amazonaws.com"
export REGISTRY_REGION="us-west-2"

# optional, only used for make ci-build-and-upload-cli
export CLI_BUCKET_NAME="cortex-cli-<your_name>"
export CLI_BUCKET_REGION="us-west-2"
```

### Building

Build and push all Cortex images:

```bash
make registry-all
```

Build and configure the Cortex CLI:

```bash
make cli  # The binary will be placed in path/to/cortex/bin/cortex
path/to/cortex/bin/cortex configure
```

### Cortex Cluster

Start Cortex:

```bash
make cortex-up
```

Tear down the Cortex cluster:

```bash
make cortex-down
```

### Deployment an Example

```bash
cd examples/iris-classifier
path/to/cortex/bin/cortex deploy
```

## Off-cluster Operator

If you're making changes in the operator and want faster iterations, you can run an off-cluster operator.

1. `make operator-stop` to stop the in-cluster operator
1. `make devstart` to run the off-cluster operator (which rebuilds the CLI and restarts the Operator when files change)
1. `path/to/cortex/bin/cortex configure` (on a separate terminal) to configure your cortex CLI to use the off-cluster operator. When prompted for operator URL, use `http://localhost:8888`

Note: `make cortex-up-dev` will start Cortex without installing the operator.

If you want to switch back to the in-cluster operator:

1. `<ctrl+C>` to stop your off-cluster operator
1. `make operator-start` to install the operator in your cluster
1. `path/to/cortex/bin/cortex configure` to configure your cortex CLI to use the in-cluster operator. When prompted for operator URL, use the URL shown when running `make cortex-info`

## Dev Workflow

1. `make cortex-up-dev`
1. `make devstart`
1. Make changes
1. `make registry-dev`
1. Test your changes with projects in `examples` or your own

See `Makefile` for additional dev commands
