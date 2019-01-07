#!/bin/bash
#SBATCH --job-name=process
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --mem=64Gb
#SBATCH --workdir=/gscratch/abc/xyz/clustering_folder
#SBATCH --partition=your_partition
#SBATCH --account=your_partition

module load anaconda3_4.3.1

# "Usage: python cluster_kmean_hb.py energyFile keyAtomFile maxK"
time python cluster_kmean_hb.py combine_pm6_energy.txt atomlist 100
