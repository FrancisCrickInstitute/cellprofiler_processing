#!/bin/sh
#SBATCH --job-name=Joe_Stan_Astrocytes_cell_paint_30062025
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-14

module load Singularity/3.6.4
module load Java/21.0.2

SEEDFILE=/nemo/stp/hts/working/Phenix/Joe_Stan_Astrocytes_cell_paint_30062025/cellprofiler/cp_commands.txt
SEED=$(awk "NR==$SLURM_ARRAY_TASK_ID" "$SEEDFILE")
eval "$SEED"
