# Cell Painting Analysis Pipeline

Statistical analysis pipeline for astrocyte morphology from Cell Painting data.

## Prerequisites

### 1. CellProfiler Processing

Before running this analysis, you must first process your raw imaging data through the CellProfiler pipeline:

**CellProfiler Pipeline Repository:**  
[https://github.com/FrancisCrickInstitute/cellprofiler_processing/]

The CellProfiler pipeline generates the required input file:
- `processed_image_data_well_level.parquet`

### 2. Python Environment

Create the conda environment from the provided environment file:

```bash
conda env create -f environment.yml
conda activate cellpainting_analysis
```

## Configuration

All analysis settings can be modified in the `AnalysisConfig` class at the top of the script:

```python
class AnalysisConfig:
    # Data paths
    DATA_PATH = "/path/to/processed_image_data_well_level.parquet"
    OUTPUT_DIR = "/path/to/output/directory"
    
    # Experimental design
    CONTROL_LINES = ['CTRL1', 'CTRL2', 'CTRL3', 'CTRL5', 'CTRL6']
    MUTANT_LINES = ['CB1D', 'CB1E', 'NC2', 'GliA', 'GliB']
    
    # Analysis parameters
    FDR_THRESHOLD = 0.05
    TOP_FEATURES_FOR_PLOTS = 8
    TOP_FEATURES_FOR_HEATMAP = 50
```

**Update these paths before running:**
1. `DATA_PATH` - Path to your processed parquet file
2. `OUTPUT_DIR` - Where you want results saved
3. `CONTROL_LINES` and `MUTANT_LINES` - Match your experimental design

## Running the Analysis

Simply run the script:

```bash
python cell_painting_analysis.py
```

The script will:
1. Load well-level data
2. Perform quality control
3. Create biological replicates (averaging technical replicates)
4. Run statistical comparisons (Mann-Whitney U tests with FDR correction)
5. Generate visualizations
6. Export results to CSV files

## Output Files

All results are saved to the specified `OUTPUT_DIR`:

### Data Files
- `biological_replicates_averaged.csv` - Key file for downstream analysis
- `*_statistical_results.csv` - Statistical test results for each comparison
- `all_significant_features_combined.csv` - All significant features across comparisons
- `analysis_summary_statistics.csv` - Overall summary metrics

### Plots (in `plots/` subdirectory)
- `significant_features_summary.png` - Bar chart of significant feature counts
- `volcano_plots_*.png` - Traditional and effect size volcano plots
- `mean_difference_heatmap_*.png` - Heatmap of top differential features
- `enhanced_boxplots_*.png` - Box plots of top features
- `features_heatmap.png` - Cross-comparison heatmap

## Analysis Approach

The pipeline performs three paired comparisons against Control Untreated:

1. **ALS Untreated vs Control Untreated**
2. **ALS Treated vs Control Untreated**  
3. **Control Treated vs Control Untreated**

Statistical testing uses:
- Mann-Whitney U test (Wilcoxon rank-sum)
- Benjamini-Hochberg FDR correction
- Effect size calculation (Cohen's d)

## Customization

Key parameters in `AnalysisConfig`:
- `FDR_THRESHOLD` - Statistical significance threshold (default: 0.05)
- `MISSING_THRESHOLD` - Max missing data per feature (default: 0.05)
- `EXTREME_FEATURE_THRESHOLD` - Multiplier for extreme value detection (default: 100)
- `TOP_FEATURES_FOR_PLOTS` - Number of features to plot (default: 8)

## Troubleshooting

**Error: File not found**
- Check that `DATA_PATH` points to your processed parquet file
- Ensure CellProfiler processing completed successfully

**Error: Missing required columns**
- Verify your parquet file contains: `Metadata_Patient_Line`, `Metadata_Treatment`, `Metadata_lib_plate_order`

**No significant features found**
- Check `FDR_THRESHOLD` setting
- Review quality control parameters
- Verify sufficient replicates per condition

## Support

For issues with:
- **CellProfiler processing**: See CellProfiler pipeline repository
- **This analysis script**: Check configuration settings and input data format


## CellProfiler Configuration

This configuration file controls HPC implementation of CellProfiler

**Original source:** [Scott Warchal's CP Config Repo](https://github.com/FrancisCrickInstitute/cp_config)

Example configuration files are provided in the `cp_config/` directory:

- `pipeline.cppipe` - CellProfiler pipeline for HPC implementation
- `cp_config.yml` - CellProfiler configuration
- `submit_analysis.sh` - SLURM script for running CellProfiler analysis
- `submit_collate.sh` - SLURM script for collating results


