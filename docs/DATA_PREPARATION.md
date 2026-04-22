# Data Preparation

## V2X-Seq

We use the [V2X-Seq-SPD](https://drive.google.com/drive/folders/1gnrw5llXAIxuB9sEKKCm6xTaJ5HQAw2e?usp=sharing) dataset, which is the pioneering real-world sequential V2X perception dataset. Please follow the steps below to download and preprocess the data for training and evaluation.

### 1. Download and Organize Data
Download the V2X-Seq-SPD dataset and place it inside the `datasets/` folder under the project root. Ensure the directory is structured exactly as follows:
```text
{XET-V2X_ROOT}/datasets/V2X-Seq-SPD
├── cooperative/
│   ├── label/
│   ├── data_info.json
├── infrastructure-side/
│   ├── velodyne/
│   ├── image/
│   ├── calib/
│   ├── label/
│   ├── data_info.json
├── maps/
│   ├── yizhuang02.json
│   ├── yizhuangxx.json         # other map files
├── vehicle-side/
│   ├── velodyne/
│   ├── image/
│   ├── calib/
│   ├── label/
│   ├── data_info.json
```

### 2. Filter LiDAR Field of View (FOV)

To align the sensor modalities and focus the perception range, we filter the raw 360-degree vehicle-side point clouds to retain only the frontal 180-degree Field of View (FOV).

Run the following script to generate the filtered point cloud files. The `--fov 90.0` argument specifies a range of ±90.0 degrees (180 degrees total) relative to the vehicle's forward axis.

```bash
cd {XET-V2X_ROOT}
python tools/v2xseq_data_converter/filter_front_fov.py \
    ./datasets/V2X-Seq-SPD/vehicle-side/velodyne \
    ./datasets/V2X-Seq-SPD/vehicle-side/velodyne_180deg \
    --fov 90.0
```

### 3. Generate Communication Delay Annotations

To simulate real-world communication delays in vehicle-to-everything (V2X) cooperative perception (e.g., $k=1, 2, 3$ frames corresponding to 100ms, 200ms, and 300ms delays at 10Hz), we need to generate asynchronous data frame pairs. 

We provide a unified script that automatically searches backward in the temporal sequences to match the vehicle's current frame with the infrastructure's delayed historical frames. 

Run the following command from the project root to process `vehicle-side`, `infrastructure-side`, and `cooperative` annotations simultaneously:

```bash
cd {XET-V2X_ROOT}
python tools/v2xseq_data_converter/gen_xetv2x_data_info.py --base-dir ./datasets/V2X-Seq-SPD
```

After successful execution, your dataset directories will be populated with the newly generated delay-specific JSON indexes.

```text
{XET-V2X_ROOT}/datasets/V2X-Seq-SPD
├── cooperative/
│   ├── label/
│   ├── data_info.json          # Original synchronous data (0 delay)
│   ├── data_info_1.json        # 1-frame delay index (100ms)
│   ├── data_info_2.json        # 2-frame delay index (200ms)
│   ├── data_info_3.json        # 3-frame delay index (300ms)
│   ├── data_info_0_1_2_3.json  # Aggregated index for robust training
├── infrastructure-side/
│   ├── velodyne/
│   ├── image/
│   ├── calib/
│   ├── label/
│   ├── data_info.json          # Original synchronous data (0 delay)
│   ├── data_info_1.json        # 1-frame delay index (100ms)
│   ├── data_info_2.json        # 2-frame delay index (200ms)
│   ├── data_info_3.json        # 3-frame delay index (300ms)
│   ├── data_info_0_1_2_3.json  # Aggregated index for robust training
├── maps/
│   ├── yizhuang02.json
│   ├── yizhuangxx.json         # other map files
├── vehicle-side/
│   ├── velodyne/
│   ├── image/
│   ├── calib/
│   ├── label/
│   ├── data_info.json          # Original synchronous data (0 delay)
│   ├── data_info_1.json        # 1-frame delay index (100ms)
│   ├── data_info_2.json        # 2-frame delay index (200ms)
│   ├── data_info_3.json        # 3-frame delay index (300ms)
│   ├── data_info_0_1_2_3.json  # Aggregated index for robust training
```


### 4. Construct Delay-Specific Dataset Variants

To evaluate the perception system under varying communication latencies, dataset variants corresponding to different delay frames must be constructed.

The parameter $k$ represents the delay frame index ($k \in \{0, 1, 2, 3\}$), where $k=0$ denotes synchronous data (no delay), and $k>0$ represents asynchronous data with a $k \times 100$ ms delay (assuming a 10 Hz sampling frequency). Crucially, the sequence identifiers are mapped to the delay parameter $k$ by adding an offset of $k \times 100$ to the base sequence IDs (e.g., base sequence `0015` becomes `0115` for $k=1$).

You can generate the specific dataset variants by executing the corresponding commands below based on your desired delay configuration:

**For $k = 0$ (Synchronous Data):**
```bash
python tools/v2xseq_data_converter/gen_xetv2x_dataset.py \
    --input ./datasets/V2X-Seq-SPD \
    --ln-input ./datasets/V2X-Seq-SPD \
    --output ./datasets/V2X-Seq-SPD-Delay-0 \
    --sequences 0000 0001 0002 0003 0004 0005 0007 0008 0010 0014 0015 0016 0017 0018 0020 0021 0022 0023 0025 0029 0030 0032 0033 0034 0035 0036 0037 0040 0041 0042 0047 0048 0049 0050 0052 0054 0055 0056 0057 0058 0059 0060 0061 0062 0063 0066 0068 0070 0071 0072 0073 0075 0077 0078 0079 0080 0081 0082 0084 0085 0086 0087 0088 0089 0092 0093 0094 \
    --update-label \
    --freq 10 \
    --delay 0
```

**For $k = 1$ (100 ms Delay):**
```bash
python tools/v2xseq_data_converter/gen_xetv2x_dataset.py \
    --input ./datasets/V2X-Seq-SPD \
    --ln-input ./datasets/V2X-Seq-SPD \
    --output ./datasets/V2X-Seq-SPD-Delay-1 \
    --sequences 0100 0101 0102 0103 0104 0105 0107 0108 0110 0114 0115 0116 0117 0118 0120 0121 0122 0123 0125 0129 0130 0132 0133 0134 0135 0136 0137 0140 0141 0142 0147 0148 0149 0150 0152 0154 0155 0156 0157 0158 0159 0160 0161 0162 0163 0166 0168 0170 0171 0172 0173 0175 0177 0178 0179 0180 0181 0182 0184 0185 0186 0187 0188 0189 0192 0193 0194 \
    --update-label \
    --freq 10 \
    --delay 1
```

**For $k = 2$ (200 ms Delay):**
```bash
python tools/v2xseq_data_converter/gen_xetv2x_dataset.py \
    --input ./datasets/V2X-Seq-SPD \
    --ln-input ./datasets/V2X-Seq-SPD \
    --output ./datasets/V2X-Seq-SPD-Delay-2 \
    --sequences 0200 0201 0202 0203 0204 0205 0207 0208 0210 0214 0215 0216 0217 0218 0220 0221 0222 0223 0225 0229 0230 0232 0233 0234 0235 0236 0237 0240 0241 0242 0247 0248 0249 0250 0252 0254 0255 0256 0257 0258 0259 0260 0261 0262 0263 0266 0268 0270 0271 0272 0273 0275 0277 0278 0279 0280 0281 0282 0284 0285 0286 0287 0288 0289 0292 0293 0294 \
    --update-label \
    --freq 10 \
    --delay 2
```

**For $k = 3$ (300 ms Delay):**
```bash
python tools/v2xseq_data_converter/gen_xetv2x_dataset.py \
    --input ./datasets/V2X-Seq-SPD \
    --ln-input ./datasets/V2X-Seq-SPD \
    --output ./datasets/V2X-Seq-SPD-Delay-3 \
    --sequences 0300 0301 0302 0303 0304 0305 0307 0308 0310 0314 0315 0316 0317 0318 0320 0321 0322 0323 0325 0329 0330 0332 0333 0334 0335 0336 0337 0340 0341 0342 0347 0348 0349 0350 0352 0354 0355 0356 0357 0358 0359 0360 0361 0362 0363 0366 0368 0370 0371 0372 0373 0375 0377 0378 0379 0380 0381 0382 0384 0385 0386 0387 0388 0389 0392 0393 0394 \
    --update-label \
    --freq 10 \
    --delay 3
```

**Note:** This script extracts the specified sequence data and organizes it into a new directory named `V2X-Seq-SPD-Delay-{k}`. Soft links (`--ln-input`) are utilized for heavy sensor data (such as images and point clouds) to optimize disk space usage.

### 5. Convert to nuScenes Format

To ensure compatibility with our 3D perception pipeline—which is built upon the OpenMMLab ecosystem—the generated datasets must be converted into the standard **nuScenes format**.

Run the following bash script to perform the conversion. Make sure to use the same delay index `k` that you specified in the previous step:

```bash
# Ensure 'k' matches the delay index used in Step 3
k=0

bash tools/v2xseq_converter.sh V2X-Seq-SPD-Delay-${k}
```
**Note**: This script will parse the raw sensor data and labels, ultimately generating the `.pkl` annotation files required by the data loaders during model training and evaluation.

### 6. Prepare Model Checkpoints

**Pre-trained Backbone Weights** To accelerate convergence, we initialize our image feature extractor using the official [BEVFormer](https://github.com/fundamentalvision/BEVFormer) pre-trained weights. 
Create a `ckpts` directory and download the weights using the following commands:

```bash
mkdir -p ckpts
cd ckpts

# Download BEVFormer-Base (ResNet-101) pre-trained weights
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth

# Return to the project root
cd ..
```

**Fully Trained XET-V2X Models** 
For quick reproduction and performance evaluation, we provide our fully trained network checkpoints. 
Please refer to the [Training and Evaluation](TRAIN_EVAL.md) documentation for the download links and detailed testing instructions.


## V2X-Sim

We also evaluate our framework on [V2X-Sim](https://github.com/ai4ce/V2X-Sim), an extensive simulated dataset and benchmark for V2X collaborative perception. Please follow the steps below to prepare the V2X-Sim data.

### 1. Download and Organize Data
Download the V2X-Sim 2.0 dataset from the [official repository](https://huggingface.co/datasets/ai4ce-drive/V2X-Sim-2.0) and place it inside the `datasets/` folder. Ensure the directory is structured as follows:

```text
{XET-V2X_ROOT}/datasets/V2X-Sim-2.0
├── maps/
├── samples/
│   ├── CAM_FRONT_id_0/
│   ├── LIDAR_TOP_id_0/
│   └── ... (other sensors and agents)
├── sweeps/
│   ├── LIDAR_TOP_id_0/
│   ├── LIDAR_TOP_id_1/
│   └── LIDAR_TOP_id_2/
└── v1.0-trainval/              # nuScenes-format JSON annotations
```

### 2. Filter LiDAR Field of View (FOV)

To simulate realistic sensor limitations and align the perception range across different agents, we filter the raw point clouds. For V2X-Sim, we apply a 90-degree frontal FOV filter ($\pm45^\circ$) to specific agents (`id_1` and `id_2`), while keeping `id_0` unfiltered.

Run the following scripts to process the LiDAR sweeps. Notice the omission of the `--filter_front` flag for `id_0`:

```bash
cd {XET-V2X_ROOT}

# Filter agent id_1 (apply frontal FOV filter)
python tools/v2xsim_data_converter/filter_front_fov.py \
    ./datasets/V2X-Sim-2.0/sweeps/LIDAR_TOP_id_1 \
    ./datasets/V2X-Sim-2.0/sweeps_90deg/LIDAR_TOP_id_1 \
    --fov 45.0 --filter_front

# Filter agent id_2 (apply frontal FOV filter)
python tools/v2xsim_data_converter/filter_front_fov.py \
    ./datasets/V2X-Sim-2.0/sweeps/LIDAR_TOP_id_2 \
    ./datasets/V2X-Sim-2.0/sweeps_90deg/LIDAR_TOP_id_2 \
    --fov 45.0 --filter_front

# Process agent id_0 (do NOT apply frontal filter, just copy/format)
python tools/v2xsim_data_converter/filter_front_fov.py \
    ./datasets/V2X-Sim-2.0/sweeps/LIDAR_TOP_id_0 \
    ./datasets/V2X-Sim-2.0/sweeps_90deg/LIDAR_TOP_id_0
```

### 3. Prepare Dataset Directory Structures

Different evaluation tracks (e.g., single-agent ego, V2V, V2I, and communication delays) require specific directory layouts, symlinks, and ego-vehicle designations.

To avoid duplicating heavy sensor data and manually modifying JSON files, we provide an automated setup script. Run the following commands to construct the required structures for all evaluation settings:

> **Note on Delay Configurations:** For setups with asynchronous communication (`v2v-delay`, `v2i-delay`), the setup script will automatically execute a temporal shifting module (`gen_scene_delay.py`) at the end of the process. This module truncates the earliest frames and resets the temporal pointers (`prev`/`next`) to simulate the required communication latency.

```bash
# Setup Single-Agent setups
bash tools/v2xsim_data_converter/prepare_v2xsim_structure.sh datasets/V2X-Sim-2.0 id1  # ego
bash tools/v2xsim_data_converter/prepare_v2xsim_structure.sh datasets/V2X-Sim-2.0 id2  # CAV
bash tools/v2xsim_data_converter/prepare_v2xsim_structure.sh datasets/V2X-Sim-2.0 id0  # RSU
# Setup Cooperative setups (Synchronous)
bash tools/v2xsim_data_converter/prepare_v2xsim_structure.sh datasets/V2X-Sim-2.0 v2v  # id1 & id2
bash tools/v2xsim_data_converter/prepare_v2xsim_structure.sh datasets/V2X-Sim-2.0 v2i  # id1 & id0
# Setup Cooperative setups (Asynchronous / Delay)
bash tools/v2xsim_data_converter/prepare_v2xsim_structure.sh datasets/V2X-Sim-2.0 v2v-delay  # id1 & id2
bash tools/v2xsim_data_converter/prepare_v2xsim_structure.sh datasets/V2X-Sim-2.0 v2i-delay  # id1 & id0
```

### 4. Convert to nuScenes Format

```bash
bash tools/v2xsim_converter.sh V2X-Sim-full-id1 id1  # ego
bash tools/v2xsim_converter.sh V2X-Sim-full-id2 id2  # CAV
bash tools/v2xsim_converter.sh V2X-Sim-full-id0 id0  # RSU
bash tools/v2xsim_converter.sh V2X-Sim-full-v2v v2v  # id1 & id2
bash tools/v2xsim_converter.sh V2X-Sim-full-v2i v2i  # id1 & id0
bash tools/v2xsim_converter.sh V2X-Sim-full-v2v-delay v2v-delay  # id1 & id2
bash tools/v2xsim_converter.sh V2X-Sim-full-v2i-delay v2i-delay  # id1 & id0
```

### 5. Post-Process Annotations

After generating the nuScenes format `.pkl` files, we must apply a final post-processing step. This step updates the LiDAR paths (pointing them to the filtered `sweeps_pcd`), removes invalid instances and bounding boxes based on visibility flags, and cleans up broken temporal pointers (`prev`/`next`) in the JSON annotations.

Run the post-processing script for all generated dataset variants:

```bash
# Post-process Single-Agent setups
bash tools/v2xsim_data_converter/v2xsim_postprocess.sh V2X-Sim-full-id1
bash tools/v2xsim_data_converter/v2xsim_postprocess.sh V2X-Sim-full-id2
bash tools/v2xsim_data_converter/v2xsim_postprocess.sh V2X-Sim-full-id0

# Post-process Cooperative setups (Synchronous)
bash tools/v2xsim_data_converter/v2xsim_postprocess.sh V2X-Sim-full-v2v
bash tools/v2xsim_data_converter/v2xsim_postprocess.sh V2X-Sim-full-v2i

# Post-process Cooperative setups (Asynchronous / Delay)
bash tools/v2xsim_data_converter/v2xsim_postprocess.sh V2X-Sim-full-v2v-delay
bash tools/v2xsim_data_converter/v2xsim_postprocess.sh V2X-Sim-full-v2i-delay
```

### 6. Filter Valid Cooperative Scenes

Because different collaboration paradigms (V2V vs. V2I) have different valid interaction scenes, we must filter the generated sequence data (`.pkl` files) using predefined scene mappings to ensure fair evaluation metrics.

Run the scene filtering script for the corresponding tracks. Note that `id1` (the ego vehicle) participates in both paradigms and must be filtered twice to generate respective outputs.

```bash
# Filter Ego Vehicle (id1) for both V2V and V2I evaluation
bash tools/v2xsim_filter_scenes.sh V2X-Sim-full-id1 v2v
bash tools/v2xsim_filter_scenes.sh V2X-Sim-full-id1 v2i

# Filter CAV (id2) - Only participates in V2V
bash tools/v2xsim_filter_scenes.sh V2X-Sim-full-id2 v2v

# Filter RSU (id0) - Only participates in V2I
bash tools/v2xsim_filter_scenes.sh V2X-Sim-full-id0 v2i

# Filter Cooperative setups (Synchronous & Asynchronous)
bash tools/v2xsim_filter_scenes.sh V2X-Sim-full-v2v v2v
bash tools/v2xsim_filter_scenes.sh V2X-Sim-full-v2v-delay v2v

bash tools/v2xsim_filter_scenes.sh V2X-Sim-full-v2i v2i
bash tools/v2xsim_filter_scenes.sh V2X-Sim-full-v2i-delay v2i
```

### 7. Prepare Model Checkpoints

Pre-trained Backbone Weights Similar to the real-world dataset setup, we initialize the image feature extractor using BEVFormer pre-trained weights. If you haven't downloaded them yet:

```bash
mkdir -p ckpts
cd ckpts
wget [https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth)
cd ..
```

Fully Trained XET-V2X Models We provide fully trained checkpoints evaluated on V2X-Sim (including V2V and V2I variants). Please refer to the [Training and Evaluation](TRAIN_EVAL.md) documentation to download the models and run standard benchmark testing.
