#!/bin/bash
#SBATCH --job-name=cp_astrocytes_stan
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Cell Profiler data processing pipeline with mode selection

# Exit on any error
set -e

# Set working directory
cd /nemo/stp/hts/working/Joe_Tuersley/code/cellprofiler_processing

# =============================================================================
# MODE SELECTION - Change this to switch between full pipeline and replot only
# =============================================================================
MODE="full"  # Change to "full" for complete pipeline, or "replot" for plot umap/tsne from coordinates only

# Configuration
INPUT_FILE="/nemo/stp/hts/working/Phenix/Joe_Stan_Astrocytes_cell_paint_30062025/cellprofiler/output/Image.parquet"
METADATA_FILE="/nemo/stp/hts/working/Phenix/Joe_Stan_Astrocytes_cell_paint_30062025/data/meta_data/meta_data.csv"
CONFIG_FILE="./cellprofiler_analyser/config/config_20250710_stan_astrocytes_cell_paint.yml"
OUTPUT_DIR="/nemo/stp/hts/working/Phenix/Joe_Stan_Astrocytes_cell_paint_30062025/cellprofiler/processed_data_3"

# Activate conda environment
source ~/.bashrc
conda activate cellprofiler_analysis

if [ "$MODE" = "full" ]; then
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

    # Run the main script with unified config
    python3 main.py \
        --input "$INPUT_FILE" \
        --metadata "$METADATA_FILE" \
        --config "$CONFIG_FILE" \
        --output "$OUTPUT_DIR"

    echo " Pipeline completed! Results in: $OUTPUT_DIR"
    echo " Check comprehensive_summary.txt for detailed results"

elif [ "$MODE" = "replot" ]; then
    echo " Replotting UMAP/t-SNE visualizations from coordinates"
    echo "Output: $OUTPUT_DIR"

    # Simple replot - just recreate the interactive plots
    python3 main.py \
        --mode replot \
        --output "$OUTPUT_DIR" \

    echo " Replotting completed! New plots in: $OUTPUT_DIR/visualizations_redo/"

else
    echo " Invalid MODE: $MODE. Use 'full' or 'replot'"
    exit 1
fi