#!/bin/bash

# ==============================================================================
# Script: v2xsim_postprocess.sh
# Description: Wrapper script to post-process the generated .pkl and .json 
#              annotation files for the V2X-Sim datasets.
# ==============================================================================

if [ -z "$1" ]; then
    echo "Usage: bash $0 <TARGET_DATA_NAME>"
    echo "Example: bash $0 V2X-Sim-full-id1"
    exit 1
fi

DATA_NAME=$1

export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# Call the parameterized Python script
python tools/v2xsim_data_converter/v2xsim_postprocess.py ${DATA_NAME}