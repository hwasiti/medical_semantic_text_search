#!/bin/bash
#SBATCH --account=project_2007072
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=11:59:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL,ARRAY_TASKS 
#SBATCH --mail-user=wasiti14@gmail.com

module purge
module load pytorch
module load htop

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

NOTEBOOKPORT=18888
TUNNELPORT=18888

# set a random access token
TOKEN=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 49 | head -n 1`

echo "On your local machine, you should have already ran this before job started:"
echo "ssh -L18888:localhost:$TUNNELPORT $USER@mahti.csc.fi"
echo "So I will be SURE that I have logged in with the same login server number: $SLURM_SUBMIT_HOST that the sbatch job submitted which made the reverse ssh to the same login server"
echo ""
echo "and point your browser to http://localhost:18888/?token=$TOKEN"

# Set up a reverse SSH tunnel from the compute node back to the submitting host (login01 or login02)
# This is the machine we will connect to with SSH forward tunneling from our client.
# ssh -R$TUNNELPORT\:localhost:$NOTEBOOKPORT $SLURM_SUBMIT_HOST -N -f
../passh/passh -p huhuhu14!CcCc ssh -R$TUNNELPORT\:localhost:$NOTEBOOKPORT $SLURM_SUBMIT_HOST -N -f

# Start the notebook
# srun -n1 $(python3 -m site --user-base)/bin/jupyter-notebook --no-browser --port=$NOTEBOOKPORT --NotebookApp.token=$TOKEN --log-level WARN
srun -n1 jupyter lab --no-browser --port=$NOTEBOOKPORT --ip 0.0.0.0 --NotebookApp.token=$TOKEN --log-level WARN

# To stop the notebook, use 'scancel'


