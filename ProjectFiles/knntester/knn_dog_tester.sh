#!/bin/bash

# ###############################################################################
#
# Submit file for a batch job on Rosie.
#
# To submit your job, run 'sbatch <jobfile>'
# To view your jobs in the Slurm queue, run 'squeue -l -u <your_username>'
# To view details of a running job, run 'scontrol show jobid -d <jobid>'
# To cancel a job, run 'scancel <jobid>'
#
# ###############################################################################

# You _must_ specify the partition
#SBATCH --partition=teaching

# The number of nodes to request
#SBATCH --nodes=1

# The number of CPUs to request (KNN is CPU-based, so we don't need GPU)
#SBATCH --cpus-per-task=20

# The error and output files
#SBATCH --error='knn_test_%j.err'
#SBATCH --output='knn_test_%j.out'

# Kill the job if it takes longer than the specified time
#SBATCH --time=0-1:0

# Memory per CPU
#SBATCH --mem-per-cpu=4G

# ###
#
# Here's the actual job code.
#
# ###

# Path to container
container="/data/containers/msoe-tensorflow-20.07-tf2-py3.sif"

# First, train the model
command="python knn_dog_tester.py \
    --train \
    --data /home/kettnert/FinalProject/data/70-dog-breedsimage-data-set-updated \
    --model /home/kettnert/FinalProject/models/dog_breed_knn.joblib"

# Execute singularity container for training
echo "Training KNN model..."
singularity exec -B /data:/data ${container} /usr/local/bin/nvidia_entrypoint.sh ${command}

# Then run the prediction
command="python knn_dog_tester.py \
    --image /home/kettnert/FinalProject/AdaCarRide.jpg \
    --model /home/kettnert/FinalProject/models/dog_breed_knn.joblib"

# Execute singularity container for prediction
echo "Running prediction..."
singularity exec -B /data:/data ${container} /usr/local/bin/nvidia_entrypoint.sh ${command}