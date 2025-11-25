#!/bin/sh
#SBATCH --job-name=Joe_Stan_Astrocytes_cell_paint_30062025
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --dependency=singleton

~/.local/bin/cp_config_collect \
    --input=/nemo/stp/hts/working/Phenix/Joe_Stan_Astrocytes_cell_paint_30062025/cellprofiler/output \
    --output=/nemo/stp/hts/working/Phenix/Joe_Stan_Astrocytes_cell_paint_30062025/cellprofiler/output \
    --format=parquet && \
rm -rf /nemo/stp/hts/working/Phenix/Joe_Stan_Astrocytes_cell_paint_30062025/cellprofiler/output/results_*
