#!/bin/bash

# ==============================================================================
# Script: v2xsim_filter_scenes.sh
# Description: Wrapper script to filter valid scenarios (V2V or V2I) from the
#              generated .pkl files.
# ==============================================================================

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash $0 <TARGET_DATA_NAME> <COOP_TYPE: v2v|v2i>"
    echo "Example: bash $0 V2X-Sim-full-id1 v2v"
    exit 1
fi

DATA_NAME=$1
COOP_TYPE=$2

export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# Call the parameterized Python script
python tools/v2xsim_data_converter/filter_v2xsim_scenes.py ${DATA_NAME} ${COOP_TYPE}