# Expected Data Structure

This document describes the expected directory structure after running the CellProfiler pipeline and analysis scripts.

## CellProfiler Output Structure

After running the CellProfiler pipeline (`astrocytes_cell_paint_pipeline.cppipe`), you will have:
```
project_directory/
├── output/                           # CellProfiler outputs
│   ├── Image.parquet                # Main input for analysis pipeline
│   ├── Cells.parquet
│   ├── Nuclei.parquet
│   ├── Cytoplasm.parquet
│   ├── Mitochondria.parquet
│   ├── Nucleoli.parquet
│   ├── AGPSpots.parquet
│   └── RelateObjects*.parquet       # Relationship tables
│
├── loaddata/                         # CellProfiler input CSVs (optional to keep)
│   └── *.csv
│
└── pipelines/                        # Your CellProfiler pipelines
    ├── astrocytes_cell_paint_V1.cppipe
    └── astrocytes_cell_paint_V1.cpproj
```

**Key file:** `Image.parquet` is the required input for the downstream analysis pipeline.

---

## Analysis Pipeline Output Structure

After running the Cell Painting analysis pipeline (cellprofiler_processing), you will have:
```
processed_data/
├── data/
│   ├── processed_image_data.parquet              # After feature selection
│   ├── processed_image_data_normalized.parquet   # After normalization
│   └── processed_image_data_well_level.parquet   # Well-aggregated (median)
│
├── samples/                                       # CSV samples for inspection
│   └── *.csv
│
├── analysis/
│   ├── pca/                                       # PCA analysis
│   ├── correlation/                               # Feature correlations
│   └── histograms/                                # Feature distributions
│
├── visualizations/
│   ├── umap/                                      # UMAP embeddings
│   └── tsne/                                      # t-SNE embeddings
│
├── hierarchical_clustering/
│   └── hierarchical_cluster_map/                  # Clustermaps
│
└── comprehensive_summary.txt                      # Processing report
```

---

## Statistical Analysis Output Structure

After running the statistical analysis script (`feature_analysis_stan_astrocyte_V2.py`), you will have:
```
output_directory/
├── biological_replicates_averaged.csv             # Main results file
├── *_statistical_results.csv                      # Test results per comparison
├── all_significant_features_combined.csv          # All significant features
├── analysis_summary_statistics.csv                # Summary metrics
│
└── plots/
    ├── significant_features_summary.png
    ├── volcano_plots_*.png
    ├── mean_difference_heatmap_*.png
    ├── enhanced_boxplots_*.png
    └── features_heatmap.png
```

---
