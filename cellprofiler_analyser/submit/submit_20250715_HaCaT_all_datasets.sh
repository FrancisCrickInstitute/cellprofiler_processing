#!/bin/bash
#SBATCH --job-name=cp_analyser
#SBATCH --mem=1950G
#SBATCH --time=24:00:00
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


# Exit on any error
set -e

# Set working directory - CHANGED: Stay in cellprofiler_processing, not cellprofiler_analyser
cd /nemo/stp/hts/working/Joe_Tuersley/code/cellprofiler_processing

# Configuration
INPUT_FILE="/nemo/project/proj-prosperity/hts/raw/projects/20250715_HaCaT_all_datasets_cell_paint/cellprofiler/output/Image.parquet"
METADATA_FILE="/nemo/project/proj-prosperity/hts/raw/projects/20250715_HaCaT_all_datasets_cell_paint/data/meta_data/20250715_all_datasets_corrected.csv"
CONFIG_FILE="./cellprofiler_analyser/config/config_20250715_HaCaT_all_datasets.yml"
OUTPUT_DIR="/nemo/project/proj-prosperity/hts/raw/projects/20250715_HaCaT_all_datasets_cell_paint/cellprofiler/processed_data"

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
# CHANGED: Add cellprofiler_analyser/ prefix to main.py
python3 cellprofiler_analyser/main.py \
    --input "$INPUT_FILE" \
    --metadata "$METADATA_FILE" \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_DIR" \
    --run-landmark-analysis

echo " Pipeline completed! Results in: $OUTPUT_DIR"
echo " Check comprehensive_summary.txt for detailed results"