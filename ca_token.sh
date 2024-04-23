#!/usr/bin/env bash

# Get new token
CODEARTIFACT_TOKEN=$(aws codeartifact get-authorization-token --duration-seconds 900 --domain satellitevu --domain-owner 496649203775 --query authorizationToken --output text)
errcode=$?
if [[ $errcode -ne 0 ]]; then
    echo "Failed to acquire codeartifact token"

    # If *not* being sourced, or the user explicitly asks for it, exit the script
    # (if we are being sourced, this would exit the user's shell, which isn't very friendly)
    if [[ ! "${BASH_SOURCE[0]}" != "${0}" ]] || [[ ! -z "${CA_TOKEN_EXIT_ON_ERROR}" ]] ; then
        exit $errcode
    fi
fi

# Set auth for source named "artifact"
export POETRY_HTTP_BASIC_ARTIFACT_USERNAME=aws
export POETRY_HTTP_BASIC_ARTIFACT_PASSWORD=${CODEARTIFACT_TOKEN}
