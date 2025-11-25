#!/bin/bash
#SBATCH --job-name=cp_chemogenetic_JUMP_SGC_CCA
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


# Exit on any error
set -e

# Set working directory
cd /nemo/stp/hts/working/Joe_Tuersley/code/cellprofiler_processing

# Configuration
INPUT_FILE="/nemo/project/proj-prosperity/hts/raw/projects/20250612_HaCaT_GSK_chemogenetic_JUMP_SGC_CCA_cell_paint/cellprofiler/output/Image.parquet"
METADATA_FILE="/nemo/project/proj-prosperity/hts/raw/projects/20250612_HaCaT_GSK_chemogenetic_JUMP_SGC_CCA_cell_paint/data/meta_data/GSK_chemogenetic_JUMP_SGC_CCA_corrected_JUMP.csv"
CONFIG_FILE="./cellprofiler_analyser/config/config_20250612_HaCaT_GSK_chemogenetic_JUMP_SGC_CCA_cell_paint_corrected_JUMP.yml"
OUTPUT_DIR="/nemo/project/proj-prosperity/hts/raw/projects/20250612_HaCaT_GSK_chemogenetic_JUMP_SGC_CCA_cell_paint/cellprofiler/processed_data_26"

echo " Starting Cell Profiler Data Processing Pipeline"
echo "Input: $INPUT_FILE"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo " Config file not found: $CONFIG_FILE"
    echo "Please create the unified config file first."
    exit 1
fi

# Activate conda environment
source ~/.bashrc
conda activate cellprofiler_analysis

# Run the main script with unified config
# All parameters (quality control + visualization) are now in the config file
python3 main.py \
    --input "$INPUT_FILE" \
    --metadata "$METADATA_FILE" \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_DIR"

echo " Pipeline completed! Results in: $OUTPUT_DIR"
echo " Check comprehensive_summary.txt for detailed results"