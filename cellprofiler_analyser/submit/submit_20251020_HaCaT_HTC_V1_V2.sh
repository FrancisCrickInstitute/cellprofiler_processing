#!/bin/bash
#SBATCH --job-name=cp_analyser
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Exit on any error
set -e

# Set working directory
cd /nemo/stp/hts/working/Joe_Tuersley/code/cellprofiler_processing

# Configuration
INPUT_FILE="/nemo/project/proj-prosperity/hts/raw/projects/20251020_HaCaT_GSK_HTC_V1_V2_cell_paint/cellprofiler/output/Image.parquet"
METADATA_FILE="/nemo/project/proj-prosperity/hts/raw/projects/20251020_HaCaT_GSK_HTC_V1_V2_cell_paint/data/meta_data/20251020_all_datasets_corrected.csv"
CONFIG_FILE="./cellprofiler_analyser/config/config_20251020_HaCaT_HTC_V1_V2.yml"
OUTPUT_DIR="/nemo/project/proj-prosperity/hts/raw/projects/20251020_HaCaT_GSK_HTC_V1_V2_cell_paint/cellprofiler/processed_data"

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
python3 -m cellprofiler_analyser.main \
    --input "$INPUT_FILE" \
    --metadata "$METADATA_FILE" \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_DIR" \
    --run-landmark-analysis

echo " Pipeline completed! Results in: $OUTPUT_DIR"
echo " Check comprehensive_summary.txt for detailed results"