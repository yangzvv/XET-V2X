# Training and Evaluation

This document provides detailed instructions for training and evaluating models within the XET-V2X framework. 

Before proceeding, please ensure you have successfully set up the environment and prepared the datasets as described in the [Data Preparation](data_preparation.md) documentation.

---

## 🚀 Training Pipeline

We provide scripts for both single-GPU and distributed multi-GPU training. By default, the training checkpoints and logs will be saved in the specified `--work-dir`.

### 1. Single-GPU Training

If you are debugging or training on a single GPU, use the standard `train.py` script. You can pass the pre-trained weights (e.g., BEVFormer initialization) using the `--cfg-options` argument:

```bash
# Train the XET-V2X baseline model on a single GPU
python tools/train.py \
    ./projects/configs/v2x_seq_spd/xet_v2x.py \
    --work-dir ./projects/work_dirs/v2x_seq_spd/xet_v2x \
    --deterministic
```

### 2. Distributed Multi-GPU Training (Recommended)

For full model training on massive V2X datasets, we highly recommend using the distributed training script to accelerate the process. Specify the configuration file and the number of GPUs to use.

```bash
# Train the model using 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/xet_v2x_dist_train.sh \
    ./projects/configs/v2x_seq_spd/xet_v2x.py \
    4  # Number of GPUs
```

**Note:** You can replace `./projects/configs/v2x_seq_spd/xet_v2x.py` with your custom XET-V2X configuration file depending on the specific fusion strategy or delay setting you wish to train.
If you do not explicitly override the output path, the distributed training script will automatically route the target workspace directory based on your configuration file's path. The underlying logic uses the following string replacement:

```bash
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
```

**Example:** If your input configuration path is `./projects/configs/v2x_seq_spd/xet_v2x.py`, the script will automatically strip the `.py` extension, replace `configs` with `work_dirs`, and save all checkpoints and logs to `./projects/work_dirs/v2x_seq_spd/xet_v2x/`.

## 📊 Inference and Evaluation

To evaluate a fully trained model on the validation or test set, use the distributed evaluation script. 

**Quick Inference**

Run the following command to perform evaluation. You need to provide the config file, the trained checkpoint path, and the number of GPUs. 

```Bash
# Evaluate the trained model on 4 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/xet_v2x_dist_eval.sh \
    ./projects/configs/v2x_seq_spd/xet_v2x.py \
    ./ckpts/v2x_seq_spd_xet_v2x.pth \
    4  # Number of GPUs
```

**Advanced Evaluation Options**

* Multi-GPU Evaluation: If your evaluation dataset is extremely large, you can speed up inference by increasing the GPU count (e.g., change 1 to 4 or 8).
* Evaluating Communication Delay: To test the model's robustness against varying communication latencies ($k=1, 2, 3$), ensure that your configuration file is pointing to the corresponding delay index (e.g., `data_info_1.json`) generated during the Data Preparation phase. 
