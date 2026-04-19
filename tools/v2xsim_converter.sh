#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash $0 <DATA_NAME> <TYPE: id0|id1|id2|v2v|v2v-delay|v2i|v2i-delay>"
    echo "Example: bash $0 V2X-Sim-full-v2i v2i"
    exit 1
fi

DATA_NAME=$1
TYPE=$2

if [ "$TYPE" == "id0" ]; then SCRIPT="v2xsim_create_data_id0.py"; FLAGS=""
elif [ "$TYPE" == "id1" ]; then SCRIPT="v2xsim_create_data_id1.py"; FLAGS="--mono"
elif [ "$TYPE" == "id2" ]; then SCRIPT="v2xsim_create_data_id2.py"; FLAGS="--mono"
elif [ "$TYPE" == "v2v" ]; then SCRIPT="v2xsim_create_data_v2v.py"; FLAGS="--coop --mono"
elif [ "$TYPE" == "v2v-delay" ]; then SCRIPT="v2xsim_create_data_v2v_delay.py"; FLAGS="--coop --mono"
elif [ "$TYPE" == "v2i" ]; then SCRIPT="v2xsim_create_data_v2i.py"; FLAGS="--coop --mono"
elif [ "$TYPE" == "v2i-delay" ]; then SCRIPT="v2xsim_create_data_v2i_delay.py"; FLAGS="--coop --mono"
else echo "[Error] Unknown type: $TYPE" && exit 1; fi

export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python tools/v2xsim_data_converter/v2xsim_utils.py --root-path ./datasets/${DATA_NAME}

python tools/v2xsim_data_converter/${SCRIPT} nuscenes \
       --root-path ./datasets/${DATA_NAME} \
       --out-dir ./data/infos/${DATA_NAME} \
       --extra-tag nuscenes \
       --version v1.0 \
       --canbus ./datasets/${DATA_NAME} \
       ${FLAGS}