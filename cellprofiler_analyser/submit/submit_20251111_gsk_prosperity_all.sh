#!/bin/bash
#SBATCH --job-name=cp_analyser_all_datasets
#SBATCH --mem=1900G
#SBATCH --time=24:00:00
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Exit on any error
set -e

# Set working directory
cd /nemo/stp/hts/working/Joe_Tuersley/code/cellprofiler_processing

# Configuration - NOW ONLY THE CONFIG FILE!
CONFIG_FILE="./cellprofiler_analyser/config/config_20251111_gsk_prosperity_all.yml"

echo " Starting Cell Profiler Data Processing Pipeline"
echo "Config: $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo " Config file not found: $CONFIG_FILE"
    echo "Please create the unified config file first."
    exit 1
fi

# Activate conda environment
source ~/.bashrc
conda activate cellprofiler_analysis

# Run the main script - ALL parameters now come from config
python3 -m cellprofiler_analyser.main \
    --config "$CONFIG_FILE"

echo " Pipeline completed! Check config for output directory location"
echo " Check comprehensive_summary.txt for detailed results"