#!/bin/bash

# ==============================================================================
# Script: prepare_v2xsim_structure.sh
# Description: Automates the directory setup, symlinking, and JSON modification
#              for different V2X-Sim dataset configurations.
# ==============================================================================

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash $0 <BASE_DIR> <TYPE: id0|id1|id2|v2v|v2i|v2v-delay|v2i-delay>"
    echo "Example: bash $0 datasets/V2X-Sim-2.0 id1"
    exit 1
fi

# BASE_DIR="datasets/V2X-Sim-2.0"
BASE_DIR=$1
TYPE=$2
TARGET_DIR="datasets/V2X-Sim-full-${TYPE}"

# 1. Determine the Ego ID based on the input type to modify sensor.json correctly
case $TYPE in
    id0) EGO_ID="0" ;;
    id1) EGO_ID="1" ;;
    id2) EGO_ID="2" ;;
    v2v) EGO_ID="1" ;;  # For v2v, the ego vehicle is id_1
    v2i) EGO_ID="1" ;;  # For v2i, the ego vehicle is id_1
    v2v-delay) EGO_ID="1" ;;  # For v2v-delay, the ego vehicle is id_1
    v2i-delay) EGO_ID="1" ;;  # For v2i-delay, the ego vehicle is id_1
    *) 
        echo "[Error] Unknown type: $TYPE. Valid options are: id0, id1, id2, v2v, v2i, v2v-delay, v2i-delay"
        exit 1 
        ;;
esac

echo "========================================================"
echo "Preparing directory structure for: ${TARGET_DIR}"
echo "Ego Sensor ID to replace: LIDAR_TOP_id_${EGO_ID}"
echo "========================================================"

# 2. Check if the base dataset exists
if [ ! -d "$BASE_DIR" ]; then
    echo "[Error] Base dataset directory not found: $BASE_DIR"
    exit 1
fi

# 3. Create target directory
mkdir -p "${TARGET_DIR}"

# 4. Copy v2.0 to v1.0-trainval
echo "-> Copying annotations..."
rm -rf "${TARGET_DIR}/v1.0-trainval"
cp -r "${BASE_DIR}/v2.0" "${TARGET_DIR}/v1.0-trainval"

# 5. Restore sample_annotation_old.json if it was previously modified
if [ -f "${TARGET_DIR}/v1.0-trainval/sample_annotation_old.json" ]; then
    echo "-> Found sample_annotation_old.json. Restoring to sample_annotation.json..."
    mv "${TARGET_DIR}/v1.0-trainval/sample_annotation_old.json" "${TARGET_DIR}/v1.0-trainval/sample_annotation.json"
fi

# 6. Create lidarseg and symlink
echo "-> Creating lidarseg symlink..."
mkdir -p "${TARGET_DIR}/lidarseg"
ln -sfn "$(realpath ${BASE_DIR}/lidarseg/v1.0-mini)" "${TARGET_DIR}/lidarseg/v1.0-trainval"

# 7. Symlink sweeps and sweeps_pcd (from sweeps_90deg)
echo "-> Creating sweeps and sweeps_pcd symlinks..."
ln -sfn "$(realpath ${BASE_DIR}/sweeps)" "${TARGET_DIR}/sweeps"
ln -sfn "$(realpath ${BASE_DIR}/sweeps_90deg)" "${TARGET_DIR}/sweeps_pcd"

# 8. Create empty can_bus directory
echo "-> Creating can_bus directory..."
mkdir -p "${TARGET_DIR}/can_bus"

# 9. Copy map directory
echo "-> Copying map files..."
cp -r "${BASE_DIR}/maps" "${TARGET_DIR}/"

# 10. Modify sensor.json (Replace specific LIDAR_TOP_id_X with LIDAR_TOP)
echo "-> Modifying sensor.json..."
SENSOR_JSON_PATH="${TARGET_DIR}/v1.0-trainval/sensor.json"
if [ -f "$SENSOR_JSON_PATH" ]; then
    # Use standard sed replacement
    sed -i "s/LIDAR_TOP_id_${EGO_ID}/LIDAR_TOP/g" "$SENSOR_JSON_PATH"
    echo "Successfully replaced LIDAR_TOP_id_${EGO_ID} with LIDAR_TOP."
else
    echo "[Warning] sensor.json not found in ${TARGET_DIR}/v1.0-trainval/"
fi

echo "Setup complete for ${TARGET_DIR}!"

# 11. Apply temporal shift for delay configurations
if [[ "$TYPE" == *"delay"* ]]; then
    echo "-> Delay configuration detected. Applying temporal shift to JSONs..."
    export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
    python tools/v2xsim_data_converter/gen_scene_delay.py "${TARGET_DIR}"
fi

echo "All processes finished for ${TARGET_DIR}!"