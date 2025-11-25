#!/bin/bash
#SBATCH --job-name=cp_flexible
#SBATCH --mem=1000G
#SBATCH --time=24:00:00
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Exit on any error
set -e

# ============================================
# USAGE INSTRUCTIONS:
# ============================================
#
# This flexible pipeline allows you to start from different points
# to save time when re-running analyses or testing specific components.
#
# OPTION 1: FULL PIPELINE (runs everything from scratch)
#   sbatch submit_flexible.sh full
#   - Loads raw CellProfiler Image.parquet files
#   - Performs feature selection and quality control
#   - Normalizes data (Z-score or MAD)
#   - Aggregates to well level
#   - Creates UMAP/t-SNE visualizations
#   - Runs landmark analysis (if enabled in config)
#   - Runs hierarchical clustering
#   - Runs threshold analysis (if enabled)
#   - Creates visualization export file
#
# OPTION 2: START FROM WELL-LEVEL DATA (skip heavy processing)
#   sbatch submit_flexible.sh well
#   - Loads pre-computed well-level aggregated data
#   - Skips: image loading, feature selection, normalization, aggregation
#   - Runs: landmark analysis, clustering, threshold analysis, viz export
#   - Useful for: testing landmark/clustering parameters
#
# OPTION 3: LANDMARK ANALYSIS ONLY (for testing landmark parameters)
#   sbatch submit_flexible.sh landmark
#   - Loads pre-computed well-level data and embeddings
#   - Runs: landmark analysis, threshold analysis, hierarchical clustering, viz export
#   - Skips: all data processing and visualization generation
#   - Useful for: tuning landmark thresholds or testing different reference sets
#
# OPTION 4: POST-LANDMARK FULL DISTANCE MATRICES (resume after Step 13)
#   sbatch submit_flexible.sh post-landmark-full-dist
#   - Loads existing landmarks and centroids from previous run
#   - Runs: Step 13 (full distance matrix generation)
#   - Continues with: hierarchical clustering, threshold analysis, viz export
#   - Useful for: resuming after Step 13 failure, testing Step 13 optimizations
#
# OPTION 5: POST-CLUSTERING (threshold + viz only - fastest targeted mode)
#   sbatch submit_flexible.sh post-clustering
#   - Loads existing well-level data, landmarks, and clustering results
#   - Runs: threshold analysis, viz export ONLY
#   - Skips: all processing, landmark analysis, Step 13, clustering
#   - Useful for: fixing threshold/viz bugs, regenerating final outputs
#
# OPTION 6: REGENERATE PLOTS ONLY (quickest visualization refresh)
#   sbatch submit_flexible.sh replot
#   - Loads saved UMAP/t-SNE coordinates
#   - Regenerates all visualization plots with current config settings
#   - Useful for: changing plot colors, styles, or parameters
#
# OPTIONAL: Override the previous run directory
#   By default, uses the path in config file's skip_mode_paths section
#   To use a different previous run:
#   sbatch submit_flexible.sh well /path/to/specific/previous/results
#   sbatch submit_flexible.sh post-clustering /path/to/specific/previous/results
#
# ============================================
# DECISION TREE: Which mode should I use?
# ============================================
#
# Starting fresh? → full
# Need to reprocess data? → full
#
# Have well-level data? → well
# Testing landmark parameters? → landmark or post-clustering
# Testing clustering parameters? → well or landmark
#
# Step 13 failed? → post-landmark-full-dist
# Clustering failed? → post-clustering
# Threshold analysis failed? → post-clustering
# Viz export failed? → post-clustering
#
# Just want new plots? → replot
#
# ============================================

# Get the mode from command line (default to 'full')
MODE="${1:-full}"
PREVIOUS_RUN_DIR="${2:-}"  # Optional second argument

# Validate mode
if [[ ! "$MODE" =~ ^(full|well|landmark|post-landmark-full-dist|post-clustering|replot)$ ]]; then
    echo "ERROR: Invalid mode '$MODE'"
    echo "Valid modes: full, well, landmark, post-landmark-full-dist, post-clustering, replot"
    exit 1
fi

# Set working directory
cd /nemo/stp/hts/working/Joe_Tuersley/code/cellprofiler_processing

# Configuration file
CONFIG_FILE="./cellprofiler_analyser/config/config_20251111_crispr_only.yml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=============================================="
echo "Cell Painting Pipeline"
echo "=============================================="
echo "Mode: $MODE"
echo "Config: $CONFIG_FILE"
if [ ! -z "$PREVIOUS_RUN_DIR" ]; then
    echo "Previous run: $PREVIOUS_RUN_DIR"
else
    echo "Previous run: Using path from config file"
fi
echo "=============================================="
echo ""

# Activate conda environment
source ~/.bashrc
conda activate cellprofiler_analysis

# Run with appropriate mode
echo "Starting pipeline in $MODE mode..."

if [ -z "$PREVIOUS_RUN_DIR" ]; then
    # No previous directory specified - use from config
    python3 -m cellprofiler_analyser.main \
        --config "$CONFIG_FILE" \
        --start-from "$MODE"
else
    # Previous directory specified on command line
    python3 -m cellprofiler_analyser.main \
        --config "$CONFIG_FILE" \
        --start-from "$MODE" \
        --previous-run-dir "$PREVIOUS_RUN_DIR"
fi

echo ""
echo "=============================================="
echo "Pipeline completed successfully!"
echo "Mode: $MODE"
echo "Check output directory for results"
echo "=============================================="



