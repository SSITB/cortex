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
	"fmt"
	"net/http"

	"github.com/cortexlabs/cortex/pkg/lib/errors"
	"github.com/cortexlabs/cortex/pkg/operator/api/schema"
	"github.com/cortexlabs/cortex/pkg/operator/config"
)

func Info(w http.ResponseWriter, r *http.Request) {
	asgs, err := config.AWS.AutoscalingGroups(map[string]string{"alpha.eksctl.io/nodegroup-name": "ng-cortex-worker"})
	if err != nil {
		RespondError(w, errors.WithStack(err))
		return
	}

	if len(asgs) != 1 {
		message := fmt.Sprintf("found %d matching autoscaling groups, expected only one", len(asgs))
		RespondError(w, errors.New(message)) // unexpected
		return
	}

	config.Cluster.MinInstances = asgs[0].MinSize
	config.Cluster.MaxInstances = asgs[0].MaxSize

	response := schema.InfoResponse{ClusterConfig: config.Cluster}
	Respond(w, response)
}
