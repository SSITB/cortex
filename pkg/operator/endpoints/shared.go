/*
Copyright 2019 Cortex Labs, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package endpoints

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"

	"github.com/cortexlabs/cortex/pkg/lib/errors"
	s "github.com/cortexlabs/cortex/pkg/lib/strings"
	"github.com/cortexlabs/cortex/pkg/operator/api/schema"
	"github.com/cortexlabs/cortex/pkg/operator/config"
)

func ResDeploymentDeleted(appName string) string {
	return fmt.Sprintf("deleting %s deployment", appName)
}

func ResDeploymentUpToDate(appName string) string {
	return fmt.Sprintf("%s deployment is up to date", appName)
}

func ResDeploymentUpToDateUpdating(appName string) string {
	return fmt.Sprintf("%s deployment is already updating", appName)
}

func ResDifferentDeploymentUpdating(appName string) string {
	return fmt.Sprintf("previous %s deployment is updating (override with --force)", appName)
}

func ResCreatingAPI(apiName string) string {
	return fmt.Sprintf("creating %s api", apiName)
}

func ResUpdatingAPI(apiName string) string {
	return fmt.Sprintf("updating %s api", apiName)
}

func ResDeletingAPI(apiName string) string {
	return fmt.Sprintf("deleting %s api", apiName)
}

func Respond(w http.ResponseWriter, response interface{}) {
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}

func RespondError(w http.ResponseWriter, err error, strs ...string) {
	RespondErrorCode(w, http.StatusBadRequest, err, strs...)
}

func RespondErrorCode(w http.ResponseWriter, code int, err error, strs ...string) {
	err = errors.Wrap(err, strs...)
	errors.PrintError(err)

	w.WriteHeader(code)
	response := schema.ErrorResponse{
		Error: err.Error(),
	}
	json.NewEncoder(w).Encode(response)
}

func RecoverAndRespond(w http.ResponseWriter, strs ...string) {
	if errInterface := recover(); errInterface != nil {
		err := errors.CastRecoverError(errInterface, strs...)
		config.Telemetry.ReportError(err)
		RespondError(w, err)
	}
}

func getRequiredPathParam(paramName string, r *http.Request) (string, error) {
	param := mux.Vars(r)[paramName]
	if param == "" {
		return "", ErrorPathParamRequired(paramName)
	}
	return param, nil
}

func getRequiredQueryParam(paramName string, r *http.Request) (string, error) {
	param := r.URL.Query().Get(paramName)
	if param == "" {
		return "", ErrorQueryParamRequired(paramName)
	}
	return param, nil
}

func getOptionalQParam(paramName string, r *http.Request) string {
	return r.URL.Query().Get(paramName)
}

func getOptionalBoolQParam(paramName string, defaultVal bool, r *http.Request) bool {
	param := r.URL.Query().Get(paramName)
	paramBool, ok := s.ParseBool(param)
	if ok {
		return paramBool
	}
	return defaultVal
}
