#!/bin/bash
#SBATCH --job-name=cp_astrocytes_stan
#SBATCH --mem=164G
#SBATCH --time=24:00:00
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Cell Profiler data processing pipeline with mode selection

# Exit on any error
set -e

# Set working directory
cd /nemo/stp/hts/working/Joe_Tuersley/code/cellprofiler_processing

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "========================================"

# =============================================================================
# MODE SELECTION - Change this to switch between full pipeline and replot only
# =============================================================================
MODE="full"  # Change to "full" for complete pipeline, or "replot" for plot umap/tsne from coordinates only

# Configuration
INPUT_FILE="/nemo/stp/hts/working/Phenix/Joe_Stan_Astrocytes_cell_paint_30062025/cellprofiler/output/Image.parquet"
METADATA_FILE="/nemo/stp/hts/working/Phenix/Joe_Stan_Astrocytes_cell_paint_30062025/data/meta_data/meta_data.csv"
CONFIG_FILE="./cellprofiler_analyser/config/config_20250710_stan_astrocytes_cell_paint.yml"
OUTPUT_DIR="/nemo/stp/hts/working/Phenix/Joe_Stan_Astrocytes_cell_paint_30062025/cellprofiler/processed_data"

echo "Configuration:"
echo "  Mode: $MODE"
echo "  Input File: $INPUT_FILE"
echo "  Metadata File: $METADATA_FILE"
echo "  Config File: $CONFIG_FILE"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# Check if files exist
echo "File Existence Check:"
echo "  Input file exists: $([ -f "$INPUT_FILE" ] && echo "YES" || echo "NO")"
echo "  Metadata file exists: $([ -f "$METADATA_FILE" ] && echo "YES" || echo "NO")"
echo "  Config file exists: $([ -f "$CONFIG_FILE" ] && echo "YES" || echo "NO")"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source ~/.bashrc
conda activate cellprofiler_analysis

# Check Python and package availability
echo "Environment Check:"
echo "  Python version: $(python --version)"
echo "  Python path: $(which python)"
echo "  Current conda env: $CONDA_DEFAULT_ENV"
echo ""

# Test imports
echo "Testing critical imports..."
python -c "
try:
    import pandas as pd
    print('  ✓ pandas available:', pd.__version__)
except ImportError as e:
    print('  ✗ pandas import failed:', e)

try:
    import numpy as np
    print('  ✓ numpy available:', np.__version__)
except ImportError as e:
    print('  ✗ numpy import failed:', e)

try:
    import morar
    print('  ✓ morar available')
except ImportError as e:
    print('  ✗ morar import failed:', e)

try:
    import umap
    print('  ✓ umap available')
except ImportError as e:
    print('  ✗ umap import failed:', e)

try:
    import plotly
    print('  ✓ plotly available:', plotly.__version__)
except ImportError as e:
    print('  ✗ plotly import failed:', e)
"
echo ""

if [ "$MODE" = "full" ]; then
    echo "========================================"
    echo "STARTING FULL PIPELINE"
    echo "========================================"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not found: $CONFIG_FILE"
        echo "Please create the unified config file first."
        exit 1
    fi

    # Run the main script with unified config
    echo "Launching main.py..."
    echo "Command: python3 main.py --input \"$INPUT_FILE\" --metadata \"$METADATA_FILE\" --config \"$CONFIG_FILE\" --output \"$OUTPUT_DIR\""
    echo ""
    
    python3 main.py \
        --input "$INPUT_FILE" \
        --metadata "$METADATA_FILE" \
        --config "$CONFIG_FILE" \
        --output "$OUTPUT_DIR"

    echo ""
    echo "========================================"
    echo "PIPELINE COMPLETED SUCCESSFULLY!"
    echo "========================================"
    echo "Results in: $OUTPUT_DIR"
    echo "Check comprehensive_summary.txt for detailed results"

elif [ "$MODE" = "replot" ]; then
    echo "========================================"
    echo "STARTING REPLOT MODE"
    echo "========================================"
    
    # Simple replot - just recreate the interactive plots
    echo "Launching replot mode..."
    python3 main.py \
        --mode replot \
        --output "$OUTPUT_DIR"

    echo ""
    echo "========================================"
    echo "REPLOTTING COMPLETED!"
    echo "========================================"
    echo "New plots in: $OUTPUT_DIR/visualizations_redo/"

else
    echo "ERROR: Invalid MODE: $MODE. Use 'full' or 'replot'"
    exit 1
fi

echo ""
echo "========================================"
echo "JOB COMPLETED"
echo "========================================"
echo "End Time: $(date)"
echo "Output files:"
echo "  Standard output: slurm_${SLURM_JOB_ID}.out"
echo "  Standard error: slurm_${SLURM_JOB_ID}.err"