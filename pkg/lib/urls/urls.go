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

package urls

import (
	"net/url"
	"regexp"
	"strings"

	s "github.com/cortexlabs/cortex/pkg/lib/strings"
)

var (
	dns1035Regex  = regexp.MustCompile(`^[a-z]([-a-z0-9]*[a-z0-9])?$`)
	dns1123Regex  = regexp.MustCompile(`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`)
	endpointRegex = regexp.MustCompile(`^[a-zA-Z0-9_\-\./]*$`)
)

func Parse(rawurl string) (*url.URL, error) {
	u, err := url.Parse(rawurl)
	if err != nil {
		return nil, ErrorInvalidURL(rawurl)
	}
	return u, nil
}

func Join(strs ...string) string {
	fullPath := ""
	for i, str := range strs {
		if i == 0 {
			fullPath = str
		} else {
			fullPath = s.EnsureSuffix(fullPath, "/")
			fullPath = fullPath + strings.TrimPrefix(str, "/")
		}
	}
	return fullPath
}

func CheckDNS1035(str string) error {
	if !dns1035Regex.MatchString(str) {
		return ErrorDNS1035(str)
	}
	return nil
}

func CheckDNS1123(str string) error {
	if !dns1123Regex.MatchString(str) {
		return ErrorDNS1123(str)
	}
	return nil
}

func ValidateEndpoint(str string) (string, error) {
	if !endpointRegex.MatchString(str) {
		return "", ErrorEndpoint(str)
	}

	if strings.Contains(str, "//") {
		return "", ErrorEndpointDoubleSlash(str)
	}

	path := CanonicalizeEndpoint(str)

	if path == "/" {
		return "", ErrorEndpointEmptyPath()
	}

	return path, nil
}

func CanonicalizeEndpoint(str string) string {
	if str == "" || str == "/" {
		return "/"
	}
	return strings.TrimSuffix(s.EnsurePrefix(str, "/"), "/")
}
