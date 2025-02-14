# Copyright 2019 Cortex Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import imp
import inspect

import boto3
import datadog
import dill

from cortex import consts
from cortex.lib import util
from cortex.lib.log import refresh_logger
from cortex.lib.storage import S3, LocalStorage
from cortex.lib.exceptions import CortexException, UserException
from cortex.lib.resources import ResourceMap


class Context:
    def __init__(self, **kwargs):
        if "cache_dir" in kwargs:
            self.cache_dir = kwargs["cache_dir"]
        elif "local_path" in kwargs:
            local_path_dir = os.path.dirname(os.path.abspath(kwargs["local_path"]))
            self.cache_dir = os.path.join(local_path_dir, "cache")
        else:
            raise ValueError("cache_dir must be specified (or inferred from local_path)")
        util.mkdir_p(self.cache_dir)

        if "local_path" in kwargs:
            self.ctx = util.read_msgpack(kwargs["local_path"])
        elif "obj" in kwargs:
            self.ctx = kwargs["obj"]
        elif "raw_obj" in kwargs:
            self.ctx = kwargs["raw_obj"]
        elif "s3_path":
            local_ctx_path = os.path.join(self.cache_dir, "context.msgpack")
            bucket, key = S3.deconstruct_s3_path(kwargs["s3_path"])
            S3(bucket, client_config={}).download_file(key, local_ctx_path)
            self.ctx = util.read_msgpack(local_ctx_path)
        else:
            raise ValueError("invalid context args: " + kwargs)

        self.workload_id = kwargs.get("workload_id")

        self.id = self.ctx["id"]
        self.key = self.ctx["key"]
        self.metadata_root = self.ctx["metadata_root"]
        self.cluster_config = self.ctx["cluster_config"]
        self.deployment_version = self.ctx["deployment_version"]
        self.root = self.ctx["root"]
        self.status_prefix = self.ctx["status_prefix"]
        self.app = self.ctx["app"]
        self.apis = self.ctx["apis"] or {}
        self.api_version = self.cluster_config["api_version"]
        self.monitoring = None
        self.project_id = self.ctx["project_id"]
        self.project_key = self.ctx["project_key"]

        if "local_storage_path" in kwargs:
            self.storage = LocalStorage(base_dir=kwargs["local_storage_path"])
        else:
            self.storage = S3(
                bucket=self.cluster_config["bucket"],
                region=self.cluster_config["region"],
                client_config={},
            )

        host_ip = os.environ["HOST_IP"]
        datadog.initialize(statsd_host=host_ip, statsd_port="8125")
        self.statsd = datadog.statsd

        if self.api_version != consts.CORTEX_VERSION:
            raise ValueError(
                "API version mismatch (Context: {}, Image: {})".format(
                    self.api_version, consts.CORTEX_VERSION
                )
            )

        # This affects TensorFlow S3 access
        os.environ["AWS_REGION"] = self.cluster_config.get("region", "")

        # ID maps
        self.apis_id_map = ResourceMap(self.apis) if self.apis else None
        self.id_map = self.apis_id_map

    def download_file(self, impl_key, cache_impl_path):
        if not os.path.isfile(cache_impl_path):
            self.storage.download_file(impl_key, cache_impl_path)
        return cache_impl_path

    def download_python_file(self, impl_key, module_name):
        cache_impl_path = os.path.join(self.cache_dir, "{}.py".format(module_name))
        self.download_file(impl_key, cache_impl_path)
        return cache_impl_path

    def load_module(self, module_prefix, module_name, impl_path):
        full_module_name = "{}_{}".format(module_prefix, module_name)

        if impl_path.endswith(".pickle"):
            try:
                impl = imp.new_module(full_module_name)

                with open(impl_path, "rb") as pickle_file:
                    pickled_dict = dill.load(pickle_file)
                    for key in pickled_dict:
                        setattr(impl, key, pickled_dict[key])
            except Exception as e:
                raise UserException("unable to load pickle", str(e)) from e
        else:
            try:
                impl = imp.load_source(full_module_name, impl_path)
            except Exception as e:
                raise UserException("unable to load python file", str(e)) from e

        return impl

    def get_request_handler_impl(self, api_name, project_dir):
        api = self.apis[api_name]
        model_type = api.get("tensorflow")

        if model_type is None:
            model_type = api.get("onnx")

        if model_type is None:  # unexpected
            raise CortexException(
                api_name, "failed to load request handler", "missing `tensorflow` or `onnx` key"
            )

        request_handler_path = model_type.get("request_handler")
        try:
            impl = self.load_module(
                "request_handler", api["name"], os.path.join(project_dir, request_handler_path)
            )
        except CortexException as e:
            e.wrap("api " + api_name, "failed to load request_handler", request_handler_path)
            raise
        finally:
            refresh_logger()

        try:
            _validate_impl(impl, REQUEST_HANDLER_IMPL_VALIDATION)
        except CortexException as e:
            e.wrap("api " + api_name, "request_handler " + request_handler_path)
            raise
        return impl

    def get_predictor_impl(self, api_name, project_dir):
        api = self.apis[api_name]
        try:
            impl = self.load_module(
                "predictor", api["name"], os.path.join(project_dir, api["predictor"]["path"])
            )
        except CortexException as e:
            e.wrap("api " + api_name, "failed to load predictor", api["predictor"]["path"])
            raise
        finally:
            refresh_logger()

        try:
            _validate_impl(impl, PREDICTOR_IMPL_VALIDATION)
        except CortexException as e:
            e.wrap("api " + api_name, "predictor " + api["predictor"]["path"])
            raise
        return impl

    def get_resource_status(self, resource):
        key = self.resource_status_key(resource)
        return self.storage.get_json(key, num_retries=5)

    def upload_resource_status_start(self, *resources):
        timestamp = util.now_timestamp_rfc_3339()
        for resource in resources:
            key = self.resource_status_key(resource)
            status = {
                "resource_id": resource["id"],
                "resource_type": resource["resource_type"],
                "workload_id": resource["workload_id"],
                "app_name": self.app["name"],
                "start": timestamp,
            }
            self.storage.put_json(status, key)

    def upload_resource_status_no_op(self, *resources):
        timestamp = util.now_timestamp_rfc_3339()
        for resource in resources:
            key = self.resource_status_key(resource)
            status = {
                "resource_id": resource["id"],
                "resource_type": resource["resource_type"],
                "workload_id": resource["workload_id"],
                "app_name": self.app["name"],
                "start": timestamp,
                "end": timestamp,
                "exit_code": "succeeded",
            }
            self.storage.put_json(status, key)

    def upload_resource_status_success(self, *resources):
        self.upload_resource_status_end("succeeded", *resources)

    def upload_resource_status_failed(self, *resources):
        self.upload_resource_status_end("failed", *resources)

    def upload_resource_status_end(self, exit_code, *resources):
        timestamp = util.now_timestamp_rfc_3339()
        for resource in resources:
            status = self.get_resource_status(resource)
            if status.get("end") != None:
                continue
            status["end"] = timestamp
            status["exit_code"] = exit_code
            key = self.resource_status_key(resource)
            self.storage.put_json(status, key)

    def resource_status_key(self, resource):
        return os.path.join(self.status_prefix, resource["id"], resource["workload_id"])

    def publish_metrics(self, metrics):
        if self.statsd is None:
            raise CortexException("statsd client not initialized")  # unexpected

        for metric in metrics:
            tags = ["{}:{}".format(dim["Name"], dim["Value"]) for dim in metric["Dimensions"]]
            if metric.get("Unit") == "Count":
                self.statsd.increment(metric["MetricName"], value=metric["Value"], tags=tags)
            else:
                self.statsd.histogram(metric["MetricName"], value=metric["Value"], tags=tags)


REQUEST_HANDLER_IMPL_VALIDATION = {
    "optional": [
        {"name": "pre_inference", "args": ["sample", "signature", "metadata"]},
        {"name": "post_inference", "args": ["prediction", "signature", "metadata"]},
    ]
}

PREDICTOR_IMPL_VALIDATION = {
    "optional": [{"name": "init", "args": ["model_path", "metadata"]}],
    "required": [{"name": "predict", "args": ["sample", "metadata"]}],
}


def _validate_impl(impl, impl_req):
    for optional_func in impl_req.get("optional", []):
        _validate_optional_fn_args(impl, optional_func["name"], optional_func["args"])

    for required_func in impl_req.get("required", []):
        _validate_required_fn_args(impl, required_func["name"], required_func["args"])


def _validate_required_fn(impl, fn_name):
    _validate_required_fn_args(impl, fn_name, None)


def _validate_optional_fn_args(impl, fn_name, args):
    if hasattr(impl, fn_name):
        _validate_required_fn_args(impl, fn_name, args)


def _validate_required_fn_args(impl, fn_name, args):
    fn = getattr(impl, fn_name, None)
    if not fn:
        raise UserException("function " + fn_name, "could not find function")

    if not callable(fn):
        raise UserException("function " + fn_name, "not a function")

    argspec = inspect.getargspec(fn)
    if argspec.varargs != None or argspec.keywords != None or argspec.defaults != None:
        raise UserException(
            "function " + fn_name,
            "invalid function signature, can only accept positional arguments",
        )

    if args:
        if argspec.args != args:
            raise UserException(
                "function " + fn_name,
                "expected function arguments arguments ({}) but found ({})".format(
                    ", ".join(args), ", ".join(argspec.args)
                ),
            )
