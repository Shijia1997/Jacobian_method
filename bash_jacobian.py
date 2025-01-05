#!/bin/bash

#SBATCH --job-name=Jacobian_bash
#SBATCH --partition=bstgpu3
#SBATCH --gpus=1            
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --time=200:00:00
#SBATCH --output=Jacobian_resulr.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shijiazhang1997@gmail.com

# Load necessary modules
module load conda
source activate ML_env

# Set the variables
DATA_PATH="df_first_folds.csv"
OUTPUT_FOLDER="/dcs07/zwang/data/jacobian_cv"
MODEL_NAME="J_CNN3D"  # Options: J_CNN3D, JAL, Res10
CLS_TYPE="cn_vs_mci_vs_ad"  # Options: cn_vs_mci_vs_ad, cn_vs_rest, cn_vs_ad, cn_vs_mci, mci_vs_ad
NUM_CLASSES=3  # Adjust based on CLS_TYPE
NUM_EPOCHS=50
LEARNING_RATE=0.0001
DEVICE="cuda"
DATA_TYPES="affine_registered,syn_registered"
INPUT_TYPES="single,concat,attention"

# Run the Python script
python run_cross_validate.py \
  --data_path $DATA_PATH \
  --output_folder $OUTPUT_FOLDER \
  --model_name $MODEL_NAME \
  --cls_type $CLS_TYPE \
  --num_classes $NUM_CLASSES \
  --device $DEVICE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --data_types $DATA_TYPES \
  --input_types $INPUT_TYPES
