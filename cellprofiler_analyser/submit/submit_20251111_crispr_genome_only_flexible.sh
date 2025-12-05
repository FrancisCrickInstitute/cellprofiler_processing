#!/bin/bash
#SBATCH --job-name=cp_analysis_genome_only
#SBATCH --mem=1000G
#SBATCH --time=24:00:00
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Exit on any error
set -e

# ============================================
# CELL PAINTING PIPELINE - THREE MODES
# ============================================
#
# MODE 1: FULL - Complete analysis from raw data
#   sbatch submit.sh full
#
# MODE 2: WELL - Start from well-level data
#   sbatch submit.sh well /path/to/previous/run
#   OR just: sbatch submit.sh well  (if previous_run_base is set in config)
#
# MODE 3: REPLOT - Regenerate UMAP/t-SNE plots only 
#   sbatch submit.sh replot /path/to/previous/run
#   OR just: sbatch submit.sh replot  (if previous_run_base is set in config)
#   NOTE: Only regenerates visualization plots, NOT landmark analysis or clustering
#
# ============================================
# ANALYSIS FLAGS (set in config.yml):
# ============================================
#
# analysis:
#   run_landmark_analysis: true/false              # Identify landmarks
#   run_hierarchical_clustering: true/false        # Generate clustering PDFs
#   run_landmark_threshold_analysis: true/false    # Test multiple MAD thresholds
#
# visualization:
#   skip_embedding_generation: true/false          # Reuse UMAP/t-SNE coordinates
#
# Note: replot mode ignores all analysis flags and only regenerates plots
#
# ============================================
# QUICK REFERENCE:
# ============================================
#
# First run (everything):
#   - Mode: full
#   - Flags: run_landmark_analysis=true, skip_embedding_generation=false

#
# Test landmark parameters (fast iteration):
#   - Mode: well /path/to/full/run
#   - Flags: run_landmark_analysis=true, skip_embedding_generation=true

#
# Test MAD thresholds:
#   - Mode: well /path/to/full/run
#   - Flags: run_landmark_analysis=true, run_landmark_threshold_analysis=true, 
#            skip_embedding_generation=true

#
# Regenerate UMAP/t-SNE plots only (tweak colors, styling, etc.):
#   - Mode: replot /path/to/any/run
#   - Flags: None (replot ignores all flags)
#   - Output: New timestamped directory with visualizations_redo/ subfolder

# ============================================
# MODE COMPARISON (UPDATED)
# ============================================
#
# Feature                    | full Mode              | well Mode              | replot Mode
# ---------------------------|------------------------|------------------------|------------------
# Raw data processing        | ✓ Always               | ✗ Skipped              | ✗ Skipped
# UMAP/t-SNE computation     | ✓ or skip (YAML)       | ✓ or skip (YAML)       | ✗ Skipped
# UMAP/t-SNE plotting        | ✓ (note 1)             | ✓ or skip (YAML)       | ✓ From coordinates
# Landmark analysis          | Optional (YAML)        | Optional (YAML)        | ✗ NOT regenerated
# Hierarchical clustering    | Optional (note 2)      | Optional (note 2)      | ✗ NOT regenerated
# Landmark threshold analysis| Optional (note 2)      | Optional (note 2)      | ✗ NOT regenerated
# Viz export (CSV)           | ✓ Generated            | ✓ Generated            | ✗ NOT regenerated
# Output folder              | YYYYMMDD_results       | YYYYMMDD_from_well_*   | YYYYMMDD_replot_results
#
# ============================================
# FLAG DEPENDENCIES (auto-handled):
# ============================================
# - Clustering requires landmark analysis (auto-enabled if needed) [note 2]
# - Threshold analysis requires landmark analysis (auto-enabled if needed) [note 2]
# - skip_embedding_generation requires previous run coordinates
# - replot mode requires previous run with embedding coordinates
# - In full mode, UMAP/t-SNE plotting only runs when computation is enabled [note 1]
#   (no previous coordinates available if computation is skipped)
#
# Control flags (YAML path):
#   - visualization.skip_embedding_generation     (full/well modes)
#   - analysis.run_landmark_analysis              (full/well modes)
#   - analysis.run_hierarchical_clustering        (full/well modes)
#   - analysis.run_landmark_threshold_analysis    (full/well modes)
#

# Get mode from command line
MODE="${1:-full}"
PREVIOUS_RUN_DIR="${2:-}"

# Set working directory FIRST (needed for relative config path)
cd /nemo/stp/hts/working/Joe_Tuersley/code/cellprofiler_processing

# Configuration file - DEFINE BEFORE USING IT
CONFIG_FILE="./cellprofiler_analyser/config/config_20251111_crispr_only.yml"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Validate mode
if [[ ! "$MODE" =~ ^(full|well|replot)$ ]]; then
    echo "ERROR: Invalid mode '$MODE'"
    echo "Valid modes: full, well, replot"
    echo ""
    echo "Usage:"
    echo "  sbatch submit.sh full"
    echo "  sbatch submit.sh well /path/to/previous/run"
    echo "  sbatch submit.sh replot /path/to/previous/run"
    exit 1
fi

# Check if previous run dir required for well/replot modes
if [[ "$MODE" =~ ^(well|replot)$ ]] && [ -z "$PREVIOUS_RUN_DIR" ]; then
    # Try to get from config file
    PREVIOUS_RUN_DIR=$(grep "previous_run_base:" "$CONFIG_FILE" 2>/dev/null | sed 's/.*: *"\([^"]*\)".*/\1/' | tr -d ' ')
    
    if [ -z "$PREVIOUS_RUN_DIR" ] || [ ! -d "$PREVIOUS_RUN_DIR" ]; then
        echo "ERROR: Mode '$MODE' requires previous run directory"
        echo ""
        echo "Either provide it as argument:"
        echo "  sbatch submit.sh well /path/to/previous/run"
        echo "  sbatch submit.sh replot /path/to/previous/run"
        echo ""
        echo "Or set it in config.yml:"
        echo "  skip_mode_paths:"
        echo "    previous_run_base: \"/path/to/previous/run\""
        exit 1
    fi
    
    echo "Using previous_run_base from config: $PREVIOUS_RUN_DIR"
fi

# Display run info
echo "=============================================="
echo "Cell Painting Pipeline"
echo "=============================================="
echo "Mode: $MODE"
echo "Config: $CONFIG_FILE"
if [ ! -z "$PREVIOUS_RUN_DIR" ]; then
    echo "Previous run: $PREVIOUS_RUN_DIR"
fi
echo ""
if [ "$MODE" != "replot" ]; then
    echo "Flags from config:"
    echo "  - analysis.run_landmark_analysis"
    echo "  - analysis.run_hierarchical_clustering"
    echo "  - analysis.run_landmark_threshold_analysis"
    echo "  - visualization.skip_embedding_generation"
else
    echo "Replot mode: Ignoring all analysis flags"
    echo "  - Only regenerating visualizations from coordinates"
fi
echo "=============================================="

# Activate environment
source ~/.bashrc
conda activate cellprofiler_analysis

# Run pipeline
echo "Starting pipeline in $MODE mode..."
echo ""

if [ -z "$PREVIOUS_RUN_DIR" ]; then
    python3 -m cellprofiler_analyser.main \
        --config "$CONFIG_FILE" \
        --start-from "$MODE"
else
    python3 -m cellprofiler_analyser.main \
        --config "$CONFIG_FILE" \
        --start-from "$MODE" \
        --previous-run-dir "$PREVIOUS_RUN_DIR"
fi

# Completion message
echo ""
echo "=============================================="
echo "Pipeline completed successfully!"
echo "=============================================="
echo "Mode: $MODE"
echo ""
echo "Key outputs:"
if [ "$MODE" == "full" ]; then
    echo "  - data/ - Processed datasets"
    echo "  - visualizations/ - UMAP/t-SNE plots and coordinates"
    echo "  - analysis/ - PCA, correlation, histograms"
    echo "  - landmark_analysis/ - Landmark results (if enabled)"
    echo "  - hierarchical_clustering/ - Dendrograms (if enabled)"
    echo "  - threshold_analysis/ - Threshold tests (if enabled)"
    echo "  - data/cp_for_viz_app.csv - Final visualization export"
elif [ "$MODE" == "well" ]; then
    echo "  - landmark_analysis/ - Landmark results (if enabled)"
    echo "  - hierarchical_clustering/ - Dendrograms (if enabled)"
    echo "  - threshold_analysis/ - Threshold tests (if enabled)"
    echo "  - data/cp_for_viz_app.csv - Final visualization export"
elif [ "$MODE" == "replot" ]; then
    echo "  - New timestamped directory: YYYYMMDD_HHMMSS_replot_results/"
    echo "  - visualizations_redo/ - Regenerated UMAP/t-SNE plots ONLY"
    echo "  - NOTE: Landmark analysis and clustering NOT regenerated"
fi
echo "=============================================="