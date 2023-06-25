#!/bin/bash
#SBATCH --gpus 1
# ·3 days:  SBATCH -t 3-00:00:00
#SBATCH -t 08:00:00
#SBATCH -A berzelius-2023-89

# The '-A' SBATCH switch above is only necessary if you are member of several
# projects on Berzelius, and can otherwise be left out.

# Apptainer images can only be used outside /home. In this example the
# image is located here
# cd /proj/my_project_storage/users/$(id -un)

# Execute my Apptainer image binding in the current working directory
# containing the Python script I want to execute
# apptainer exec --nv -b $(pwd) Example_Image.sif python train_my_DNN.py

# Change to directory:
cd /proj/berzelius-2023-89/users/x_rampo/1NoiseModelling/lightning_CNN

# Activate modules:
module load Anaconda/2021.05-nsc1

# load environment
conda activate lightningNN

# Run
srun python main_unet.py

