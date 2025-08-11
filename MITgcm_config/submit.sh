#!/bin/bash                                                                 
#
#SBATCH --account=XXXX
#SBATCH --time=X-XX:XX:XX
#SBATCH --job-name=XXXX
#SBATCH --ntasks=XXXX
#SBATCH --nodes=XXXX
#SBATCH --mem=XXXX
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XXXX
#SBATCH --partition=XXXX

echo "Simulaiton start at: "`date` 
echo 'Run: mpirun -np XX ./mitgcmuv'
time -p ( mpirun -np XX ./mitgcmuv )
echo "" ; echo "Simulation end at: "`date`
