#!/bin/bash
#SBATCH --job-name=cp_replot_optimized
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Exit on any error
set -e

# Set working directory
cd /nemo/stp/hts/working/Joe_Tuersley/code/cellprofiler_processing

# Configuration
OUTPUT_DIR="/nemo/project/proj-prosperity/hts/raw/projects/20250612_HaCaT_GSK_chemogenetic_JUMP_SGC_CCA_cell_paint/cellprofiler/processed_data_5"

echo " Starting optimized visualization replotting"
echo "Output: $OUTPUT_DIR"

# Check if output directory and coordinate files exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo " Output directory not found: $OUTPUT_DIR"
    exit 1
fi

# Activate conda environment
source ~/.bashrc
conda activate cellprofiler_analysis

# Run replotting
python3 main.py \
    --mode replot \
    --output "$OUTPUT_DIR"

echo " Replotting completed! Results in: $OUTPUT_DIR/visualizations_redo/"