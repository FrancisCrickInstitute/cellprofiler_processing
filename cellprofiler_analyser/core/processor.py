"""
Main processor class for Enhanced Cell Painting data processing
UPDATED: Now uses Z-score normalization as default with both baseline options
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
import shutil
import os

try:
    import morar
except ImportError:
    morar = None

from .data_loader import DataLoader
from .feature_selection import FeatureSelector
from .normalization import DataNormalizer
from .visualization import DataVisualizer
from ..io.data_io import save_all_datasets
from ..io.config_loader import load_config, get_viz_parameters, get_quality_control_params, get_visualization_flags
from ..utils.logging_utils import get_logger
from ..analysis.landmark_analysis import LandmarkAnalyzer
from ..analysis import run_landmark_threshold_analysis
from ..io.viz_export import VizDataExporter

logger = get_logger(__name__)


class EnhancedCellPaintingProcessor:
    """
    Enhanced Cell Painting data processor with Z-score normalization
    """
    
    # In processor.py, modify the __init__ method:

    def __init__(self, input_file: Union[str, List[str]], metadata_file: Optional[str] = None, 
                config_file: Optional[str] = None, output_dir: str = "./processed_data"):
        """
        Initialize the processor

        Args:
            input_file: Path to main data file (Image.parquet) OR list of files
            metadata_file: Path to metadata CSV file
            config_file: Path to unified config YAML file
            output_dir: Output directory for results
        """
        # Handle both single file and list of files
        if isinstance(input_file, list):
            self.input_files = input_file
            # Fixed: Check if list is not empty before accessing index 0
            self.input_file = input_file[0] if input_file else None
        else:
            self.input_files = [input_file] if input_file else []
            self.input_file = input_file
            
        self.metadata_file = metadata_file
        self.config_file = config_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = load_config(config_file)
        
        # Initialize components - pass the list of files
        self.data_loader = DataLoader(self.input_files, metadata_file, config_file)

        
        # Initialize feature selector with config parameters
        qc_params = get_quality_control_params(self.config)
        self.feature_selector = FeatureSelector(
            missing_threshold=qc_params['missing_threshold'],
            correlation_threshold=qc_params['correlation_threshold'],
            high_variability_threshold=qc_params['high_variability_threshold'],
            low_variability_threshold=qc_params['low_variability_threshold']
        )
        
        self.normalizer = DataNormalizer()
        self.visualizer = None  # Initialized later when output_dir is finalized
        
        # Data containers
        self.raw_data = None
        self.processed_data = None
        self.normalized_data = None
        self.scaled_data = None
        self.well_aggregated_data = None
        
        # Track processing statistics
        self.removed_rows_count = 0

        self._logger = logger
        
        # Log configuration
        logger.info(f"Initialized processor with {len(self.input_files)} input files")
        for i, f in enumerate(self.input_files, 1):
            logger.info(f"  Input file {i}: {f}")
        logger.info(f"Quality control parameters: {qc_params}")
        logger.info(f"Default normalization method: Z-score (DMSO baseline)")

    
    def set_quality_thresholds(self, missing_threshold: float = None,
                              correlation_threshold: float = None,
                              high_variability_threshold: float = None,
                              low_variability_threshold: float = None):
        """
        Override quality control thresholds (takes precedence over config file)
        """
        logger.info("Overriding quality control thresholds from command line")
        
        if missing_threshold is not None:
            self.feature_selector.missing_threshold = missing_threshold
            
        if correlation_threshold is not None:
            self.feature_selector.correlation_threshold = correlation_threshold
            
        if high_variability_threshold is not None:
            self.feature_selector.high_variability_threshold = high_variability_threshold
            
        if low_variability_threshold is not None:
            self.feature_selector.low_variability_threshold = low_variability_threshold
        
        logger.info(f"Updated thresholds: missing={self.feature_selector.missing_threshold}, "
                   f"correlation={self.feature_selector.correlation_threshold}, "
                   f"high_var={self.feature_selector.high_variability_threshold}, "
                   f"low_var={self.feature_selector.low_variability_threshold}")
    
    def load_data(self) -> bool:
        logger.info("Loading and processing raw data")
        try:
            self.raw_data = self.data_loader.load_and_prepare_data()
            
            if self.raw_data is not None:
                # Filter out rows with missing perturbation_name early in pipeline
                self.raw_data = self._filter_missing_perturbations(self.raw_data)
                
            return self.raw_data is not None
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _filter_missing_perturbations(self, data):
        """
        Filter out rows where Metadata_perturbation_name is NaN/missing
        This should be done early to exclude these wells from all analyses
        
        Args:
            data: Input data (morar DataFrame or pandas DataFrame)
            
        Returns:
            Filtered data with same type as input
        """
        print("\n" + "="*80)
        print("FILTERING MISSING PERTURBATION DATA")
        print("="*80)
        print("Removing wells where Metadata_perturbation_name is NaN/missing")
        print("These wells will be excluded from all analyses (correlation, UMAP, etc.)")
        
        # Load metadata from data_loader if available for library analysis
        metadata_df = None
        if hasattr(self.data_loader, 'metadata_df') and self.data_loader.metadata_df is not None:
            metadata_df = self.data_loader.metadata_df
        elif self.metadata_file:
            print(" Loading metadata for library analysis...")
            try:
                from ..io.data_io import load_metadata_csv
                metadata_df = load_metadata_csv(self.metadata_file)
            except Exception as e:
                print(f"   Error loading metadata for library analysis: {e}")
                print("   Proceeding without library analysis...")
                metadata_df = None
        
        # Convert to pandas for filtering if needed
        if hasattr(data, 'to_pandas'):
            df = data.to_pandas()
            is_morar_input = True
        else:
            df = data
            is_morar_input = False
        
        original_count = len(df)
        print(f" Original data: {original_count:,} rows")
        
        # Check if perturbation column exists
        if 'Metadata_perturbation_name' not in df.columns:
            print("  ERROR: Metadata_perturbation_name column not found")
            print("  This is required for analysis - cannot proceed")
            raise ValueError("Missing required Metadata_perturbation_name column")
        
        # Count missing values
        missing_mask = df['Metadata_perturbation_name'].isna()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            print(" No missing perturbation_name values found - no filtering needed")
            # Still do library analysis even if no missing data
            if 'Metadata_library' in df.columns:
                print(f"\n LIBRARY ANALYSIS - DATA BEING PROCESSED:")
                library_stats = df.groupby('Metadata_library').agg({
                    'Metadata_perturbation_name': 'nunique',
                    'Metadata_plate_barcode': 'nunique'
                }).rename(columns={
                    'Metadata_perturbation_name': 'unique_perturbations',
                    'Metadata_plate_barcode': 'unique_plates'
                })
                
                if len(library_stats) > 0:
                    total_libraries_processed = len(library_stats)
                    total_perturbations = library_stats['unique_perturbations'].sum()
                    total_plates = library_stats['unique_plates'].sum()
                    
                    print(f" Processing data from {total_libraries_processed} libraries:")
                    print(f" Total: {total_perturbations} perturbations across {total_plates} plates")
                    
                    for library, stats in library_stats.iterrows():
                        print(f"   {library}: {stats['unique_perturbations']} perturbations, {stats['unique_plates']} plates")
                        
                        # Show sample perturbations from this library
                        library_perturbations = df[df['Metadata_library'] == library]['Metadata_perturbation_name'].unique()
                        sample_size = min(5, len(library_perturbations))
                        sample_perts = library_perturbations[:sample_size]
                        print(f"     Sample perturbations: {list(sample_perts)}")
                        if len(library_perturbations) > sample_size:
                            print(f"     ... and {len(library_perturbations) - sample_size} more")
            
            # Check unprocessed libraries
            if metadata_df is not None and 'Metadata_library' in metadata_df.columns:
                self._analyze_unprocessed_libraries(df, metadata_df)
            
            print("="*80)
            return data
        
        # Analyze missing data by library before filtering
        if 'Metadata_library' in df.columns and missing_count > 0:
            print(f"\n MISSING PERTURBATION DATA BY LIBRARY:")
            
            # Get missing data stats by library
            missing_by_library = df[missing_mask].groupby('Metadata_library').size()
            total_by_library = df.groupby('Metadata_library').size()
            
            for library in total_by_library.index:
                missing_lib_count = missing_by_library.get(library, 0)
                total_lib_count = total_by_library[library]
                missing_pct = (missing_lib_count / total_lib_count) * 100
                
                print(f"   {library}: {missing_lib_count:,} missing / {total_lib_count:,} total ({missing_pct:.1f}%)")
        
        # Filter out missing values
        df_filtered = df[~missing_mask].copy()
        filtered_count = len(df_filtered)
        removed_count = original_count - filtered_count
        
        print(f"\n OVERALL FILTERING RESULTS:")
        print(f" Removed: {removed_count:,} rows with missing perturbation_name ({removed_count/original_count*100:.1f}%)")
        print(f" Remaining: {filtered_count:,} rows with valid perturbation_name")
        
        # Check if we have any data left
        if filtered_count == 0:
            print("\n ERROR: No valid data remaining after filtering!")
            print(" All rows had missing perturbation_name values.")
            print(" Cannot proceed with analysis.")
            raise ValueError("No valid data remaining after filtering missing perturbations")
        
        # Show what perturbations we have
        unique_perturbations = df_filtered['Metadata_perturbation_name'].nunique()
        print(f" Unique perturbations remaining: {unique_perturbations}")
        
        # Show DMSO image count specifically
        dmso_count = (df_filtered['Metadata_perturbation_name'] == 'DMSO').sum()
        if dmso_count > 0:
            print(f" DMSO control images: {dmso_count:,}")
        else:
            print("WARNING: No DMSO control images found!")
        
        # Library analysis - what we're actually processing
        if 'Metadata_library' in df_filtered.columns:
            library_stats = df_filtered.groupby('Metadata_library').agg({
                'Metadata_perturbation_name': 'nunique',
                'Metadata_plate_barcode': 'nunique'
            }).rename(columns={
                'Metadata_perturbation_name': 'unique_perturbations',
                'Metadata_plate_barcode': 'unique_plates'
            })
            
            if len(library_stats) > 0:
                print(f"\n LIBRARY ANALYSIS - DATA BEING PROCESSED:")
                
                total_libraries_processed = len(library_stats)
                total_perturbations = library_stats['unique_perturbations'].sum()
                total_plates = library_stats['unique_plates'].sum()
                
                print(f" Processing data from {total_libraries_processed} libraries:")
                print(f" Total: {total_perturbations} perturbations across {total_plates} plates")
                
                for library, stats in library_stats.iterrows():
                    print(f"   {library}: {stats['unique_perturbations']} perturbations, {stats['unique_plates']} plates")
                    
                    # Show sample perturbations from this library
                    library_perturbations = df_filtered[df_filtered['Metadata_library'] == library]['Metadata_perturbation_name'].unique()
                    sample_size = min(5, len(library_perturbations))
                    sample_perts = library_perturbations[:sample_size]
                    print(f"     Sample perturbations: {list(sample_perts)}")
                    if len(library_perturbations) > sample_size:
                        print(f"     ... and {len(library_perturbations) - sample_size} more")
            else:
                print(" No library information found in processed data")

        # Check what's in the metadata CSV but not being processed
        if metadata_df is not None and 'Metadata_library' in metadata_df.columns:
            self._analyze_unprocessed_libraries(df_filtered, metadata_df)
        
        # Show top perturbations
        top_perturbations = df_filtered['Metadata_perturbation_name'].value_counts().head(5)
        print(f"\n Top 5 perturbations:")
        for pert, count in top_perturbations.items():
            print(f"   {pert}: {count:,} images")
        
        # Track removed count for summary
        self.removed_rows_count += removed_count
        
        # Convert back to original type if needed
        if is_morar_input:
            try:
                import morar
                filtered_data = morar.DataFrame(df_filtered)
                print(" Converted back to morar DataFrame")
            except Exception as e:
                print(f" ERROR: Could not convert back to morar DataFrame: {e}")
                raise RuntimeError("Failed to convert filtered data back to morar DataFrame")
        else:
            filtered_data = df_filtered
        
        print("="*80)
        return filtered_data

    def _analyze_unprocessed_libraries(self, df_filtered, metadata_df):
        """Helper method to analyze unprocessed libraries"""
        print(f"\n LIBRARY ANALYSIS - DATA AVAILABLE BUT NOT PROCESSED:")
        
        # Get libraries from metadata CSV
        csv_library_stats = metadata_df.groupby('Metadata_library').agg({
            'Metadata_perturbation_name': 'nunique',
            'Metadata_lib_plate_order': 'nunique'
        }).rename(columns={
            'Metadata_perturbation_name': 'unique_perturbations',
            'Metadata_lib_plate_order': 'unique_plates'
        })
        
        # Get processed libraries for comparison
        processed_libraries = set()
        if 'Metadata_library' in df_filtered.columns:
            processed_libraries = set(df_filtered['Metadata_library'].unique())
        
        # Find unprocessed libraries
        csv_libraries = set(csv_library_stats.index)
        unprocessed_libraries = csv_libraries - processed_libraries
        
        if unprocessed_libraries:
            print(f" Found {len(unprocessed_libraries)} libraries in metadata CSV but not processed:")
            print(f" (This occurs when plate barcodes are missing from config file)")
            
            for library in unprocessed_libraries:
                stats = csv_library_stats.loc[library]
                print(f"   {library}: {stats['unique_perturbations']} perturbations, {stats['unique_plates']} plates (NOT PROCESSED)")
                
                # Show sample perturbations from unprocessed library
                library_perturbations = metadata_df[metadata_df['Metadata_library'] == library]['Metadata_perturbation_name'].unique()
                sample_size = min(3, len(library_perturbations))
                sample_perts = library_perturbations[:sample_size]
                print(f"     Sample perturbations: {list(sample_perts)}")
        else:
            print(f" All libraries from metadata CSV are being processed")

    def process_features(self, control_compound: str = "DMSO") -> bool:
        """
        Run feature selection and cleaning pipeline with intermediate file saving
        """
        if self.raw_data is None:
            logger.error("No raw data available for feature processing")
            return False
        
        logger.info("=== FEATURE SELECTION AND CLEANING ===")
        
        try:
            # Initialize processed data
            self.processed_data = self.raw_data.copy()
            
            print("\n Running feature selection pipeline...")
            print("   This includes the expensive correlation computation")
            print("   The result will be saved for future fast testing")
            
            # Run feature selection pipeline
            self.processed_data = self.feature_selector.process_all_features(
                self.processed_data, control_compound
            )
            
            # Save feature-selected data immediately after processing
            print("\n Saving feature-selected data for future testing...")
            feature_save_success = self._save_feature_selected_data()
            
            if feature_save_success:
                print(" Feature-selected data saved successfully!")
                print("   This can be used to skip correlation computation in future runs")
            else:
                print("  Failed to save feature-selected data")
            
            # Store correlation matrix for visualization
            if hasattr(self.feature_selector, 'correlation_matrix'):
                self.visualizer.correlation_matrix = self.feature_selector.correlation_matrix
            
            logger.info(f"Feature processing completed. Final shape: {self.processed_data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error in feature processing: {e}")
            return False
    
    def _save_feature_selected_data(self) -> bool:
        """
        Save feature-selected data after correlation computation for future testing
        This is the optimal save point: after expensive feature selection, before normalization
        
        Returns:
            bool: Success status
        """
        try:
            # Create intermediate directory
            intermediate_dir = self.output_dir / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to pandas for saving
            if hasattr(self.processed_data, 'to_pandas'):
                save_data = self.processed_data.to_pandas()
            else:
                save_data = self.processed_data
            
            # Save as parquet for efficiency
            save_path = intermediate_dir / "feature_selected_data.parquet"
            save_data.to_parquet(save_path, index=False)
            
            # Also save a small sample CSV for inspection
            sample_path = intermediate_dir / "feature_selected_sample.csv"
            save_data.head(100).to_csv(sample_path, index=False)
            
            # Save info about the data
            info_path = intermediate_dir / "feature_selected_info.txt"
            with open(info_path, 'w') as f:
                f.write("FEATURE-SELECTED DATA INFO\n")
                f.write("="*50 + "\n")
                f.write(f"Data shape: {save_data.shape}\n")
                f.write(f"Feature columns: {len([col for col in save_data.columns if not col.startswith('Metadata_')])}\n")
                f.write(f"Metadata columns: {len([col for col in save_data.columns if col.startswith('Metadata_')])}\n")
                f.write(f"Memory usage: {save_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n")
                f.write(f"Generated from: {self.input_file}\n")
                f.write(f"Config used: {self.config_file}\n")
                f.write("\nThis file contains:\n")
                f.write("- All raw image-level data\n")
                f.write("- Clean features (after correlation removal)\n")
                f.write("- All metadata needed for normalization\n")
                f.write("- Ready for normalization testing\n")
                f.write("\nUse with rerun scripts to skip expensive correlation computation\n")
            
            print(f" Feature-selected data saved to: {save_path}")
            print(f" Sample CSV saved to: {sample_path}")
            print(f" Info file saved to: {info_path}")
            print(f" Data shape: {save_data.shape}")
            print(f" File size: {save_path.stat().st_size / 1024**2:.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving feature-selected data: {e}")
            return False
    
    def process_normalization(self, control_compound: str = "DMSO", 
                            use_all_conditions_baseline: bool = False) -> bool:
        """
        Run Z-score normalization pipeline with baseline choice
        
        Args:
            control_compound: Control compound name (default: "DMSO")
            use_all_conditions_baseline: If True, use all conditions as baseline. If False, use DMSO baseline.
        """
        if self.processed_data is None:
            logger.error("No processed data available for normalization")
            return False
        
        print("\n" + "="*80)
        print(" NORMALIZATION PIPELINE")
        print("="*80)
    
        if use_all_conditions_baseline:
            print(" Using Z-SCORE NORMALIZATION - ALL CONDITIONS BASELINE")
            print("   Formula: z_score = (value - all_conditions_mean_per_plate) / all_conditions_std_per_plate")
            print("   Expected: ALL conditions centered around 0")
        else:
            print(" Using Z-SCORE NORMALIZATION - DMSO BASELINE (RECOMMENDED)")
            print("   Formula: z_score = (value - DMSO_mean_per_plate) / DMSO_std_per_plate")
            print("   Expected: DMSO controls centered around 0, treatments show deviation from normal")
        
        try:
            # Convert to pandas DataFrame for normalization
            if hasattr(self.processed_data, 'to_pandas'):
                data_for_norm = self.processed_data.to_pandas()
            else:
                data_for_norm = self.processed_data
            
            # Get normalization parameters from config
            from ..io.config_loader import get_normalization_params
            norm_params = get_normalization_params(self.config)

            # Extract control column and compound from config
            config_control_column = norm_params['control_column']
            config_control_compound = norm_params['control_compound']
            
            # NEW: Get normalization type from config
            config_normalization_type = norm_params.get('normalization_type', 'control_based')
            
            # Determine which normalization to use
            # Command line parameter can override config
            if use_all_conditions_baseline:
                use_all_conditions = True
            else:
                use_all_conditions = (config_normalization_type == 'all_conditions')

            # Use config values (command line control_compound parameter can still override)
            actual_control_compound = control_compound if control_compound != "DMSO" else config_control_compound

            print(f" Using control column: {config_control_column}")
            print(f" Using control compound: {actual_control_compound}")
            print(f" Normalization type: {'all_conditions' if use_all_conditions else 'control_based'}")

            # Run normalization pipeline with method selection
            self.normalized_data, self.scaled_data, self.well_aggregated_data = \
                self.normalizer.process_normalization_pipeline(
                    data_for_norm, 
                    actual_control_compound,
                    config_control_column,
                    use_all_conditions_baseline=use_all_conditions  # Use the determined value
                )
            
            if self.well_aggregated_data is not None:
                baseline_type = "ALL CONDITIONS" if use_all_conditions_baseline else "DMSO"
                print(f"\n Z-score normalization pipeline ({baseline_type} baseline) completed successfully!")
                print(f" Well-level data shape: {self.well_aggregated_data.shape}")
                
                # Save intermediate normalized data for inspection
                self._save_intermediate_normalized_data(use_all_conditions_baseline)
                
                return True
            else:
                print(f"\n Normalization pipeline failed!")
                return False
                
        except Exception as e:
            logger.error(f"Error in normalization: {e}")
            return False
    
    def _save_intermediate_normalized_data(self, use_all_conditions_baseline: bool = False):
        """
        Save normalized data for inspection and validation
        """
        try:
            # Create intermediate directory
            intermediate_dir = self.output_dir / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            
            baseline_suffix = "_all_conditions" if use_all_conditions_baseline else "_dmso"
            
            # Save normalized data
            if self.normalized_data is not None:
                norm_path = intermediate_dir / f"normalized_data{baseline_suffix}.parquet"
                self.normalized_data.to_parquet(norm_path, index=False)
                print(f" Normalized data saved to: {norm_path}")
            
            # Save well-aggregated data
            if self.well_aggregated_data is not None:
                well_path = intermediate_dir / f"well_aggregated_data{baseline_suffix}.parquet"
                self.well_aggregated_data.to_parquet(well_path, index=False)
                print(f" Well-aggregated data saved to: {well_path}")
                
                # Save sample for inspection
                sample_path = intermediate_dir / f"well_aggregated_sample{baseline_suffix}.csv"
                self.well_aggregated_data.head(100).to_csv(sample_path, index=False)
                print(f" Well sample (100 rows) saved to: {sample_path}")
                
                # Save normalization info
                info_path = intermediate_dir / f"normalization_info{baseline_suffix}.txt"
                with open(info_path, 'w') as f:
                    f.write("NORMALIZATION RESULTS INFO\n")
                    f.write("="*50 + "\n")
                    baseline_type = "ALL CONDITIONS" if use_all_conditions_baseline else "DMSO"
                    f.write(f"Method used: Z-score ({baseline_type} baseline)\n")
                    f.write(f"Well-aggregated data shape: {self.well_aggregated_data.shape}\n")
                    
                    f.write(f"\nZ-Score Normalization ({baseline_type} baseline):\n")
                    if use_all_conditions_baseline:
                        f.write("- Formula: z_score = (value - all_conditions_mean_per_plate) / all_conditions_std_per_plate\n")
                        f.write("- ALL conditions should be centered around 0\n")
                        f.write("- Scaling based on overall plate variation\n")
                    else:
                        f.write("- Formula: z_score = (value - DMSO_mean_per_plate) / DMSO_std_per_plate\n")
                        f.write("- DMSO controls should be centered around 0\n")
                        f.write("- Treatments show deviation from normal cellular state\n")
                        f.write("- Biologically interpretable results\n")
                    f.write(f"\nGenerated from feature-selected data\n")
                    f.write(f"Ready for visualization and analysis\n")
                
                print(f" Normalization info saved to: {info_path}")
            
        except Exception as e:
            logger.warning(f"Could not save intermediate normalized data: {e}")
    
    def create_visualizations(self) -> bool:
        """
        Create visualizations and plots using optimized coordinate-based approach
        UPDATED: Now includes comprehensive histogram structure
        """
        logger.info("=== CREATING VISUALIZATIONS ===")
        
        try:
            # Use well-aggregated data for visualization
            viz_data = self.well_aggregated_data
            
            if viz_data is None:
                logger.error("No well-aggregated data available for visualization")
                return False
            
            logger.info(f"Creating visualizations from well-level data: {viz_data.shape}")
            
            # PCA variance analysis
            self.visualizer.create_pca_variance_plot(viz_data)
            
            # UPDATED: Create comprehensive histograms with full structure
            logger.info("Creating comprehensive histogram structure...")
            
            # Prepare raw data for histograms
            if self.processed_data is not None:
                # Convert morar DataFrame to pandas for histograms
                if hasattr(self.processed_data, 'to_pandas'):
                    raw_hist_data = self.processed_data.to_pandas()
                else:
                    raw_hist_data = self.processed_data
                
                logger.info("Creating comprehensive histograms (raw + normalized)...")
                self.visualizer.create_comprehensive_histograms(raw_hist_data, self.normalized_data)
            else:
                logger.warning("No raw data available for comprehensive histograms")
                # Fallback to normalized histograms only
                self.visualizer.create_split_histograms_dmso_vs_treatment(self.normalized_data)
            
            # Correlation heatmap
            if (hasattr(self.feature_selector, 'correlation_matrix') and
                self.feature_selector.correlation_matrix is not None):
                self.visualizer.create_correlation_heatmap(self.feature_selector.correlation_matrix)
            
            # Load visualization parameters from unified config
            umap_params, tsne_params = get_viz_parameters(self.config)
            
            # UMAP embeddings
            if umap_params:
                success = self.visualizer.compute_multiple_umaps(viz_data, umap_params)
                if not success:
                    logger.warning("UMAP computation failed")
            
            # t-SNE embeddings
            if tsne_params:
                success = self.visualizer.compute_multiple_tsnes(viz_data, tsne_params)
                if not success:
                    logger.warning("t-SNE computation failed")
            
            # Use coordinate-based plotting approach
            if self.visualizer.umap_results or self.visualizer.tsne_results:
                logger.info("Creating optimized interactive plots from coordinates...")
                # This will save coordinates and create plots from them
                self.visualizer.save_interactive_plots(viz_data)
            
            logger.info("Visualization creation completed using optimized coordinate-based approach")
            logger.info("Comprehensive histogram structure created in analysis/histograms/")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def save_datasets(self) -> bool:
        """
        Save all processed datasets
        """
        logger.info("=== SAVING DATASETS ===")
        
        try:
            # Prepare datasets dictionary
            datasets = {}
            
            if self.processed_data is not None:
                # Convert morar DataFrame to pandas for saving
                if hasattr(self.processed_data, 'to_pandas'):
                    datasets['processed_data'] = self.processed_data.to_pandas()
                else:
                    datasets['processed_data'] = self.processed_data
            
            if self.normalized_data is not None:
                datasets['normalized_data'] = self.normalized_data
            
            if self.scaled_data is not None:
                datasets['scaled_data'] = self.scaled_data
            
            if self.well_aggregated_data is not None:
                datasets['well_aggregated_data'] = self.well_aggregated_data
            
            # Save all datasets
            saved_files = save_all_datasets(datasets, self.output_dir)
            
            logger.info(f"Saved {len(saved_files)} dataset files")
            return True
            
        except Exception as e:
            logger.error(f"Error saving datasets: {e}")
            return False
    
    def generate_summary_report(self, use_all_conditions_baseline: bool = False) -> Path:
        """
        Generate comprehensive summary report with Z-score normalization info
        UPDATED: Now includes comprehensive histogram structure description
        """
        logger.info("Generating summary report")
        
        report = []
        report.append("=" * 80)
        report.append("ENHANCED CELL PAINTING DATA PROCESSING SUMMARY")
        report.append("=" * 80)
        report.append(f"Input file: {self.input_file}")
        report.append(f"Metadata file: {self.metadata_file}")
        report.append(f"Config file: {self.config_file}")
        
        # Normalization method info
        baseline_type = "ALL CONDITIONS" if use_all_conditions_baseline else "DMSO"
        report.append(f"Normalization method: Z-score ({baseline_type} baseline)")
        
        # Configuration summary
        if self.config:
            qc_params = get_quality_control_params(self.config)
            report.append("")
            report.append("Quality Control Parameters:")
            for param, value in qc_params.items():
                report.append(f"  {param}: {value}")
            
            # Visualization parameters
            umap_params, tsne_params = get_viz_parameters(self.config)
            report.append(f"  Visualization: {len(umap_params)} UMAP sets, {len(tsne_params)} t-SNE sets")
        
        report.append("")
        
        if self.raw_data is not None:
            report.append(f"Original shape: {self.raw_data.shape}")
        
        if self.removed_rows_count > 0:
            report.append(f"Rows removed (missing perturbation_name): {self.removed_rows_count:,}")
        
        if self.processed_data is not None:
            report.append(f"Final processed shape: {self.processed_data.shape}")
        
        # Dataset availability
        if self.normalized_data is not None:
            report.append(f"Normalized data shape: {self.normalized_data.shape}")
        if self.well_aggregated_data is not None:
            report.append(f"Well-aggregated data shape: {self.well_aggregated_data.shape}")

        report.append("")
        
        # Normalization details
        report.append("NORMALIZATION METHOD APPLIED:")
        report.append(f" Z-SCORE NORMALIZATION ({baseline_type} BASELINE)")
        if use_all_conditions_baseline:
            report.append("  Formula: z_score = (value - all_conditions_mean_per_plate) / all_conditions_std_per_plate")
            report.append("  ALL conditions centered around 0")
            report.append("  Scaling based on overall plate variation")
            report.append("  Useful for comparing relative positions within plate")
        else:
            report.append("  Formula: z_score = (value - DMSO_mean_per_plate) / DMSO_std_per_plate")
            report.append("  DMSO controls centered around 0")
            report.append("  Treatments show deviation from normal cellular state")
            report.append("  Biologically interpretable results")
            report.append("  Standard approach for Cell Painting")
        
        # Data composition summary
        if self.processed_data is not None:
            if hasattr(self.processed_data, 'to_pandas'):
                summary_data = self.processed_data.to_pandas()
            else:
                summary_data = self.processed_data
            
            summary = self.data_loader.get_metadata_summary(summary_data)
            report.append(f"Data composition: {summary['total_images']} images from {summary['n_plates']} plates")
            report.append(f"Total wells: {summary['n_wells']}")
            report.append(f"Fields per well (avg): {summary['fields_per_well']:.1f}")
            report.append(f"Perturbations: {summary['n_perturbations']} unique, {summary['n_dmso_images']} DMSO images")
        
        report.append("")
        
        # Feature removal summary
        removal_summary = self.feature_selector.get_removal_summary()
        report.append(f"Total features removed: {removal_summary['total_removed']}")
        report.append("")
        
        for step, count in removal_summary.items():
            if step != 'total_removed':
                step_name = step.replace('_', ' ').title()
                report.append(f"{step_name}: {count} features")
        report.append("")

        # PCA summary
        if self.visualizer.pca_model is not None:
            var_90 = np.argmax(np.cumsum(self.visualizer.pca_model.explained_variance_ratio_) >= 0.9) + 1
            total_var_50 = np.sum(self.visualizer.pca_model.explained_variance_ratio_[:50])
            report.append(f"PCA Analysis:")
            report.append(f"  Components for 90% variance: {var_90}")
            report.append(f"Variance in first 50 components: {total_var_50:.3f}")
            report.append("")
        
        # UPDATED: Enhanced output directory structure with comprehensive histograms
        baseline_suffix = "_all_conditions" if use_all_conditions_baseline else "_dmso"
        report.append("OUTPUT DIRECTORY STRUCTURE:")
        report.append("")
        report.append("intermediate/ [FOR TESTING - Skip expensive steps]")
        report.append("  |-- feature_selected_data.parquet [Skip correlation computation]")
        report.append("  |-- feature_selected_sample.csv [Inspection sample]")
        report.append("  |-- feature_selected_info.txt [Data information]")
        report.append(f"  |-- normalized_data{baseline_suffix}.parquet [After normalization]")
        report.append(f"  |-- well_aggregated_data{baseline_suffix}.parquet [Final analysis data]")
        report.append(f"  |-- well_aggregated_sample{baseline_suffix}.csv [Inspection sample]")
        report.append(f"  +-- normalization_info{baseline_suffix}.txt [Normalization details]")
        report.append("")
        report.append("data/ [FINAL DATASETS]")
        report.append("  |-- processed_image_data.parquet")
        report.append("  |-- processed_image_data_normalized.parquet")
        report.append("  |-- processed_image_data_standardscaler_scaled.parquet [same as normalized]")
        report.append("  +-- processed_image_data_well_level.parquet")
        report.append("")
        report.append("visualizations/ [OPTIMIZED PLOTS]")
        report.append("  |-- coordinates/embedding_coordinates.csv [All coordinates + metadata]")
        report.append("  |-- umap/interactive/ [Optimized UMAP plots - no legends]")
        report.append("  +-- tsne/interactive/ [Optimized t-SNE plots - no legends]")
        report.append("")
        report.append("analysis/ [ANALYSIS PLOTS]")
        report.append("  |-- pca/ [PCA variance analysis]")
        report.append("  |-- correlation/ [Correlation heatmaps]")
        report.append("  +-- histograms/ [COMPREHENSIVE FEATURE DISTRIBUTIONS]")
        report.append("       |-- raw/ [Raw data histograms]")
        report.append("       |    |-- all_conditions/ [All samples together]")
        report.append("       |    |-- all_conditions_log/ [All samples, log frequency]")
        report.append("       |    |-- dmso_only/ [DMSO controls only]")
        report.append("       |    |-- dmso_only_log/ [DMSO controls, log frequency]")
        report.append("       |    |-- treatments_only/ [Treatments only]")
        report.append("       |    +-- treatments_only_log/ [Treatments, log frequency]")
        report.append("       +-- normalized/ [Normalized data histograms]")
        report.append("            |-- all_conditions/ [All normalized samples]")
        report.append("            |-- all_conditions_log/ [All normalized, log frequency]")
        report.append("            |-- dmso_only/ [Normalized DMSO only]")
        report.append("            |-- dmso_only_log/ [Normalized DMSO, log frequency]")
        report.append("            |-- treatments_only/ [Normalized treatments only]")
        report.append("            +-- treatments_only_log/ [Normalized treatments, log frequency]")
        report.append("")
        report.append("hierarchical_clustering/ [CLUSTERING ANALYSIS]")
        report.append("  +-- hierarchical_cluster_map/ [Treatment clustering plots]")
        report.append("")
        
        # Usage recommendations
        report.append("USAGE RECOMMENDATIONS:")
        report.append("=" * 50)
        report.append(f" Current setup uses Z-score normalization ({baseline_type} baseline)")
        report.append("")
        report.append("For future fast testing:")
        report.append("  1. Use intermediate/feature_selected_data.parquet to skip correlation computation")
        report.append("  2. Create rerun scripts for different normalization methods")
        report.append("  3. Use visualizations/coordinates/ for plot recreation")
        report.append("")
        report.append("COMPREHENSIVE HISTOGRAM ANALYSIS:")
        report.append("  Raw histograms: Check feature distributions before normalization")
        report.append("  Normalized histograms: Verify Z-score normalization quality")
        report.append("  Log frequency plots: Better visualization of outliers and rare values")
        report.append("  Separate DMSO/treatments: Compare control vs perturbation distributions")
        report.append("")
        
        if use_all_conditions_baseline:
            report.append("Expected behavior with Z-score (ALL CONDITIONS baseline):")
            report.append("  ALL conditions (including DMSO) should be centered around 0")
            report.append("  Check normalized/all_conditions/ histograms for centering")
            report.append("  DMSO may not be centered at 0 (this is expected)")
            report.append("  Scaling based on overall plate variation")
            report.append("  Useful for understanding relative positions within plate")
        else:
            report.append("Expected DMSO control behavior with Z-score (DMSO baseline):")
            report.append("  DMSO feature means should be close to 0.0")
            report.append("  Check normalized/dmso_only/ histograms for centering")
            report.append("  Treatments show deviation from normal state in standard deviations")
            report.append("  Biologically interpretable results")
        
        report.append("")
        report.append("QUALITY CONTROL WITH DEBUG SCRIPT:")
        report.append("  Use the corrected debug script to inspect normalization quality:")
        report.append(f"    python inspect_normalized_data.py -i intermediate/well_aggregated_data{baseline_suffix}.parquet")
        report.append("  This will verify:")
        if use_all_conditions_baseline:
            report.append("    - All conditions properly centered around 0")
            report.append("    - No extreme variance features dominating PCA")
        else:
            report.append("    - DMSO controls properly centered around 0")
            report.append("    - Treatments showing appropriate deviation from controls")
        report.append("    - Blacklisted features properly removed")
        report.append("    - Feature variance distribution for PCA prediction")

        report.append("  |-- cp_for_viz_app.csv [Comprehensive viz export]")
        report.append("  |-- cp_for_viz_app.parquet [Compressed viz export]")
        
        # Save report
        report_path = self.output_dir / "comprehensive_summary.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        # Print to console
        print('\n'.join(report))
        
        return report_path
    

    def run_from_well_level(self, previous_run_dir=None):
        """Start pipeline from pre-computed well-level data"""
        logger.info("\n" + "="*80)
        logger.info("STARTING FROM WELL-LEVEL DATA")
        logger.info("="*80)
        
        # ADD THIS DEBUG LINE:
        logger.info(f"Config keys available: {self.config.keys() if self.config else 'No config'}")
        
        try:
            # Load config to get paths
            if 'skip_mode_paths' not in self.config:
                logger.error("Config missing 'skip_mode_paths' section")
                logger.error(f"Available config sections: {list(self.config.keys())}")  # ADD THIS
                return False
            
            skip_paths = self.config['skip_mode_paths']
            
            # Use previous_run_dir if provided, otherwise use from config
            if previous_run_dir:
                base_dir = Path(previous_run_dir)
            else:
                base_dir = Path(skip_paths.get('previous_run_base', ''))
            
            if not base_dir.exists():
                logger.error(f"Previous run directory not found: {base_dir}")
                return False
            
            # Load well-level data (unnormalized for viz-only mode)
            well_data_path = base_dir / skip_paths.get('well_level_parquet', 'data/processed_image_data_well_level.parquet')

            if not well_data_path.exists():
                logger.error(f"Well-level data not found: {well_data_path}")
                return False

            logger.info(f"Loading well-level data from: {well_data_path}")
            self.well_aggregated_data = pd.read_parquet(well_data_path)
            logger.info(f"Loaded {len(self.well_aggregated_data):,} wells")
            
            # Create new timestamped output directory in processed_data/
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Get the processed_data directory (parent of previous_run_base)
            processed_data_dir = base_dir.parent
            self.output_dir = processed_data_dir / f"{timestamp}_from_well_results"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {self.output_dir}")

            # Reinitialize visualizer with new output directory
            self.visualizer = DataVisualizer(self.output_dir)
                        
            # Save the well data to our new output dir so landmark analysis can find it
            data_dir = self.output_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as both CSV and Parquet for compatibility
            well_output_csv = data_dir / "processed_image_data_well_level.csv"
            well_output_parquet = data_dir / "processed_image_data_well_level.parquet"
            
            logger.info(f"Copying well data to new output directory...")
            self.well_aggregated_data.to_csv(well_output_csv, index=False)
            self.well_aggregated_data.to_parquet(well_output_parquet, index=False)
            logger.info(f"Well data saved to: {data_dir}")
            
            # Get analysis flags from config
            from ..io.config_loader import get_analysis_flags, get_visualization_flags
            analysis_flags = get_analysis_flags(self.config)
            viz_flags = get_visualization_flags(self.config)
            
            skip_embedding_generation = viz_flags.get('skip_embedding_generation', False)
            
            # Handle embedding generation or coordinate loading
            if not skip_embedding_generation:
                logger.info("\n" + "="*80)
                logger.info("GENERATING NEW EMBEDDINGS (UMAP/t-SNE)")
                logger.info("="*80)
                # We need to create visualizations on the well data
                # But we need to ensure PCA/embeddings are computed
                # This would require running create_visualizations() 
                # For now, we'll note this needs the visualization pipeline
                logger.warning("⚠️  Embedding generation in 'well' mode not yet fully implemented")
                logger.warning("   For now, use skip_embedding_generation=True in well mode")
                logger.warning("   Or run 'full' mode to generate new embeddings")
            else:
                logger.info("\n" + "="*80)
                logger.info("SKIPPING EMBEDDING GENERATION")
                logger.info("="*80)
                logger.info("Will use existing coordinates from previous run")
                
                # Find existing coordinates
                coords_path = self._get_embedding_coordinates_path(str(base_dir))
                
                if coords_path and coords_path.exists():
                    logger.info(f"✓ Found existing coordinates: {coords_path}")
                    logger.info("  Coordinates will be loaded during viz export")
                    
                    # Copy coordinates to new output directory for viz export to find
                    new_coords_dir = self.output_dir / "visualizations" / "coordinates"
                    new_coords_dir.mkdir(parents=True, exist_ok=True)
                    new_coords_path = new_coords_dir / "embedding_coordinates.csv"
                    
                    import shutil
                    shutil.copy2(coords_path, new_coords_path)
                    logger.info(f"  Copied coordinates to: {new_coords_path}")
                    
                    # Generate UMAP/t-SNE plots from the copied coordinates
                    logger.info("\n  Generating UMAP/t-SNE plots from existing coordinates...")
                    plot_success = self.visualizer.recreate_plots_from_coordinates(
                        use_pca_from_analysis=False  # No PCA model available in well mode
                    )
                    if plot_success:
                        logger.info("  ✓ UMAP/t-SNE plots generated successfully")
                    else:
                        logger.warning("  ⚠ Failed to generate UMAP/t-SNE plots from coordinates")
                else:
                    logger.error("✗ Could not find existing coordinates!")
                    logger.error("  Cannot proceed with skip_embedding_generation=True")
                    logger.error("  Either:")
                    logger.error("    1. Set skip_embedding_generation=False")
                    logger.error("    2. Run 'full' mode first to generate coordinates")
                    logger.error("    3. Check skip_mode_paths in config")
                    return False
            
            # Run landmark analysis if enabled
            landmark_success = False
            if analysis_flags.get('run_landmark_analysis', False):
                logger.info("\nRunning landmark analysis...")
                landmark_success = self.run_landmark_analysis()
                if not landmark_success:
                    logger.warning("Landmark analysis failed")
                    return False
            
            # Run threshold analysis if enabled (requires landmark analysis)
            if analysis_flags.get('run_landmark_threshold_analysis', False):
                if landmark_success:
                    logger.info("\n" + "="*80)
                    logger.info("LANDMARK THRESHOLD ANALYSIS")
                    logger.info("="*80)
                    
                    # Check if landmark file exists
                    landmark_file = self.output_dir / "landmark_analysis" / "reference_mad_and_dmso.csv"
                    
                    if landmark_file.exists():
                        # Get well-level data (already loaded in self.well_aggregated_data)
                        well_data = self.well_aggregated_data
                        
                        # Get feature columns
                        feature_cols = [col for col in well_data.columns if not col.startswith('Metadata_')]
                        logger.info(f"Using {len(feature_cols)} feature columns for distance calculations")
                        
                        # Import and run threshold analysis
                        from ..analysis import run_landmark_threshold_analysis
                        threshold_success = run_landmark_threshold_analysis(
                            well_data=well_data,
                            landmark_file=landmark_file,
                            feature_cols=feature_cols,
                            output_dir=self.output_dir,
                            config=self.config
                        )
                        
                        if threshold_success:
                            logger.info("✓ Landmark threshold analysis completed!")
                        else:
                            logger.warning("⚠ Landmark threshold analysis failed")
                    else:
                        logger.error(f"⚠ Landmark file not found: {landmark_file}")
                        logger.warning("Skipping threshold analysis")
                else:
                    logger.warning("Skipping threshold analysis - landmark analysis not run or failed")
            
            # Run hierarchical clustering if enabled (requires landmark analysis)
            if analysis_flags.get('run_hierarchical_clustering', False):
                if landmark_success:
                    logger.info("\nRunning hierarchical clustering...")
                    clustering_success = self.run_hierarchical_clustering()
                    if not clustering_success:
                        logger.warning("Hierarchical clustering failed")
                else:
                    logger.warning("Skipping hierarchical clustering - landmark analysis not run or failed")
            
            # Create visualization export
            logger.info("\nCreating visualization export...")
            viz_export_success = self.create_viz_export()
            
            # Generate summary
            self.generate_summary_report(use_all_conditions_baseline=False)
            
            logger.info(f"\nPipeline completed successfully from well-level!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to run from well-level: {e}", exc_info=True)
            return False

    def replot_only(self, previous_run_dir=None):
        """
        Regenerate UMAP/t-SNE visualization plots from existing coordinate files
        Does NOT regenerate landmark analysis or hierarchical clustering
        
        Args:
            previous_run_dir: Directory containing previous run outputs
            
        Returns:
            bool: Success status
        """
        logger.info("\n" + "="*80)
        logger.info("REPLOT MODE - Regenerating UMAP/t-SNE Plots Only")
        logger.info("="*80)
        logger.info("This mode only regenerates visualization plots from coordinates")
        logger.info("Landmark analysis and clustering are NOT regenerated")
        
        try:
            # Load config to get paths
            if 'skip_mode_paths' not in self.config:
                logger.error("Config missing 'skip_mode_paths' section")
                return False
            
            skip_paths = self.config['skip_mode_paths']
            
            # Use previous_run_dir if provided
            if previous_run_dir:
                base_dir = Path(previous_run_dir)
            else:
                base_dir = Path(skip_paths.get('previous_run_base', ''))
            
            if not base_dir.exists():
                logger.error(f"Previous run directory not found: {base_dir}")
                return False
            
            # Load embedding coordinates
            coords_path = base_dir / skip_paths.get('embedding_coordinates', 'visualizations/coordinates/embedding_coordinates.csv')
            
            if not coords_path.exists():
                logger.error(f"Embedding coordinates not found: {coords_path}")
                return False
            
            logger.info(f"Loading coordinates from: {coords_path}")
            
            # Create new output directory for plots in processed_data/
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Get the processed_data directory (parent of previous_run_base)
            processed_data_dir = base_dir.parent
            self.output_dir = processed_data_dir / f"{timestamp}_replot_results"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {self.output_dir}")

            # Reinitialize visualizer with new output directory
            self.visualizer = DataVisualizer(self.output_dir)
            
            # Initialize visualizer and recreate plots
            from .visualization import DataVisualizer
            
            visualizer = DataVisualizer(
                output_dir=self.output_dir,
                config=self.config
            )
            
            # Copy coordinates to new output directory
            viz_dir = self.output_dir / "visualizations" / "coordinates"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(coords_path, viz_dir / "embedding_coordinates.csv")
            
            # Recreate plots from coordinates
            logger.info("\nRegenerating plots from saved coordinates...")
            success = visualizer.recreate_plots_from_coordinates(str(viz_dir))
            
            if success:
                logger.info(f"\n✓ Plots regenerated successfully!")
                logger.info(f"Output directory: {self.output_dir}")
            else:
                logger.error("Failed to regenerate plots")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to replot: {e}", exc_info=True)
            return False
    
    def run_full_pipeline(self, use_all_conditions_baseline=False, 
                     start_from='full', previous_run_dir=None):
        """
        Run the complete Cell Painting processing pipeline with multiple entry points
        
        Args:
            use_all_conditions_baseline: Use all conditions for normalization (vs DMSO only)
            start_from: Entry point ('full', 'well', 'landmark', 'replot')
            previous_run_dir: Directory containing previous run outputs (for skip modes)
        
        Returns:
            bool: Success status
        """
        # ADD THIS DEBUG LINE:
        logger.info(f"DEBUG: run_full_pipeline called with start_from='{start_from}', previous_run_dir='{previous_run_dir}'")

        # Add routing logic at the beginning
        if start_from == 'well':
            logger.info(f"Starting pipeline from: well-level data")
            logger.info(f"Previous run directory: {previous_run_dir}")
            return self.run_from_well_level(previous_run_dir)
        elif start_from == 'replot':
            logger.info(f"Starting pipeline from: replot mode")
            logger.info(f"Previous run directory: {previous_run_dir}")
            return self.replot_only(previous_run_dir)
        
        
        # ========== FULL PIPELINE CODE STARTS HERE ==========
        baseline_type = "ALL CONDITIONS" if use_all_conditions_baseline else "DMSO"
        logger.info(f"Starting Enhanced Cell Painting processing pipeline with Z-score normalization ({baseline_type} baseline)")
        
        # Initialize visualizer now that output_dir is finalized (for full mode)
        if self.visualizer is None:
            self.visualizer = DataVisualizer(self.output_dir)
        
        # Get analysis flags from config
        from ..io.config_loader import get_analysis_flags
        analysis_flags = get_analysis_flags(self.config)
        viz_flags = get_visualization_flags(self.config)
        
        # Determine which analyses to run
        run_landmark_analysis = analysis_flags.get('run_landmark_analysis', False)
        run_hierarchical_clustering = analysis_flags.get('run_hierarchical_clustering', False)
        should_run_threshold_analysis = analysis_flags.get('run_landmark_threshold_analysis', False)
        skip_embedding_generation = viz_flags.get('skip_embedding_generation', False)
        
        # Validate flag dependencies
        if run_hierarchical_clustering and not run_landmark_analysis:
            logger.warning("="*80)
            logger.warning("⚠️  FLAG CONFLICT DETECTED")
            logger.warning("="*80)
            logger.warning("run_hierarchical_clustering=True requires run_landmark_analysis=True")
            logger.warning("Hierarchical clustering needs distance matrices from landmark analysis")
            logger.warning("Forcing run_landmark_analysis=True to satisfy dependency")
            logger.warning("="*80)
            run_landmark_analysis = True
        
        if run_landmark_threshold_analysis and not run_landmark_analysis:
            logger.warning("="*80)
            logger.warning("⚠️  FLAG CONFLICT DETECTED")
            logger.warning("="*80)
            logger.warning("run_landmark_threshold_analysis=True requires run_landmark_analysis=True")
        
        # Load data
        if not self.load_data():
            return False
        
        # Process features
        if not self.process_features():
            return False
        
        # Process normalization with baseline choice
        if not self.process_normalization(use_all_conditions_baseline=use_all_conditions_baseline):
            return False
        
        # Save datasets
        if not self.save_datasets():
            return False
        
        # Create visualizations (or load existing coordinates)
        if not skip_embedding_generation:
            logger.info("\n" + "="*80)
            logger.info("GENERATING NEW UMAP/t-SNE EMBEDDINGS")
            logger.info("="*80)
            if not self.create_visualizations():
                logger.warning("Visualization creation failed, but continuing...")
        else:
            logger.info("\n" + "="*80)
            logger.info("SKIPPING EMBEDDING GENERATION (REUSING EXISTING COORDINATES)")
            logger.info("="*80)
            logger.info("Flag: skip_embedding_generation=True")
            logger.info("Will load existing UMAP/t-SNE coordinates from previous run")
            logger.info("This saves ~20-30 minutes of computation time")
            logger.info("="*80)
            # Note: Actual coordinate loading happens in viz_export step
            # The visualizer will look for existing coordinate files
        
        # Run landmark analysis (required for hierarchical clustering)
        landmark_success = True
        if run_landmark_analysis or run_hierarchical_clustering:
            landmark_success = self.run_landmark_analysis()
            if not landmark_success:
                logger.warning("Landmark analysis failed")
        
        # LANDMARK THRESHOLD ANALYSIS (if enabled and landmark analysis was run)
        if run_landmark_analysis and should_run_threshold_analysis:
            print("\n" + "="*80)
            print("LANDMARK THRESHOLD ANALYSIS")
            print("="*80)
            # Check if landmark file exists
            landmark_file = self.output_dir / "landmark_analysis" / "reference_mad_and_dmso.csv"
            
            if landmark_file.exists():
                # Get well-level data
                well_data_file = self.output_dir / "data" / "processed_image_data_well_level.parquet"
                
                if well_data_file.exists():
                    print(f" Loading well-level data from: {well_data_file}")
                    well_data = pd.read_parquet(well_data_file)
                    
                    # Get feature columns
                    feature_cols = [col for col in well_data.columns if not col.startswith('Metadata_')]
                    print(f" Using {len(feature_cols)} feature columns for distance calculations")
                    
                    # Run threshold analysis
                    threshold_success = run_landmark_threshold_analysis(
                        well_data=well_data,
                        landmark_file=landmark_file,
                        feature_cols=feature_cols,
                        output_dir=self.output_dir,
                        config=self.config
                    )
                    
                    if threshold_success:
                        print(" ✓ Landmark threshold analysis completed!")
                    else:
                        print(" ⚠ Landmark threshold analysis failed")
                else:
                    print(f" ⚠ Well-level data not found: {well_data_file}")
            else:
                print(f" ⚠ Landmark file not found: {landmark_file}")
                print(" Skipping threshold analysis")
        
        # Run hierarchical clustering if enabled (requires landmark analysis outputs)
        if run_hierarchical_clustering:
            if landmark_success:
                logger.info("\nRunning hierarchical clustering (uses landmark analysis outputs)...")
                clustering_success = self.run_hierarchical_clustering()
                if not clustering_success:
                    logger.warning("Hierarchical clustering failed")
            else:
                logger.warning("Hierarchical clustering requested but landmark analysis failed/not run")
                logger.warning("  Hierarchical clustering requires landmark analysis distance matrices")
        
        # Create visualization export file (combines everything for viz app)
        logger.info("\n✓ Creating comprehensive visualization export file...")
        viz_export_success = self.create_viz_export()
        if not viz_export_success:
            logger.warning("Visualization export creation failed - continuing anyway")
        
        # Generate summary report
        self.generate_summary_report(use_all_conditions_baseline)
        
        logger.info(f"Enhanced processing pipeline completed successfully!")
        return True
    

    def run_landmark_analysis(self) -> bool:
        """
        Run landmark analysis on well-level data
        
        Requires:
            - Config with 'plate_dict' section where each plate has a 'type' field
            - Well-level data already generated
        
        Returns:
            bool: Success status
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING LANDMARK ANALYSIS")
        logger.info("="*80)
        
        # Validate config has plate_dict
        if 'plate_dict' not in self.config:
            logger.error("Config missing 'plate_dict' section for landmark analysis")
            logger.error("Please add plate_dict to your config YAML file")
            return False
        
        # Validate that plates have 'type' field
        plate_dict = self.config['plate_dict']
        sample_plate = next(iter(plate_dict.values()))
        if 'type' not in sample_plate:
            logger.error("Plates in plate_dict must have 'type' field for landmark analysis")
            logger.error("Please add 'type': 'reference' or 'type': 'test' to each plate")
            return False
        
        # Get well-level data path
        well_data_path = self.output_dir / "data" / "processed_image_data_well_level.csv"
        
        if not well_data_path.exists():
            logger.error(f"Well-level data not found: {well_data_path}")
            logger.error("Landmark analysis requires well-level aggregated data")
            return False
        
        try:
            # Create analyzer
            analyzer = LandmarkAnalyzer(
                well_data_path=str(well_data_path),
                metadata_path=self.metadata_file if self.metadata_file else None,
                config=self.config,
                output_dir=str(self.output_dir / "landmark_analysis")
            )
            
            # Run analysis
            success = analyzer.run_full_analysis()
            
            if success:
                logger.info("\nLandmark analysis completed successfully!")
                logger.info(f"Results saved to: {self.output_dir / 'landmark_analysis'}")
            
            return success
            
        except Exception as e:
            logger.error(f"Landmark analysis failed: {e}", exc_info=True)
            return False
        
        
    def run_hierarchical_clustering(self) -> bool:
        """
        Run hierarchical clustering analysis using pre-computed distance matrices
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING HIERARCHICAL CLUSTERING ANALYSIS")
        logger.info("="*80)
        
        # Check if distance matrix exists from landmark analysis
        landmark_dir = self.output_dir / "landmark_analysis"
        distance_matrix_path = landmark_dir / "cosine_distance_matrix_for_clustering.parquet"
        similarity_matrix_path = landmark_dir / "cosine_similarity_matrix_for_clustering.parquet"
        treatment_metadata_path = landmark_dir / "treatment_metadata_for_clustering.csv"
        reference_mad_path = landmark_dir / "reference_mad_and_dmso.csv"
        test_landmark_path = landmark_dir / "test_to_landmark_distances.csv"
        
        # Check if required files exist
        required_files = [
            distance_matrix_path, similarity_matrix_path, treatment_metadata_path,
            reference_mad_path, test_landmark_path
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            logger.error(f"Missing required files for hierarchical clustering:")
            for f in missing_files:
                logger.error(f"  - {f}")
            logger.error("Please run landmark analysis first to generate these files")
            return False
        
        try:
            from ..analysis.hierarchical_clustering import HierarchicalClusteringAnalyzer
            
            # Create analyzer
            clustering_dir = self.output_dir / "hierarchical_clustering"
            analyzer = HierarchicalClusteringAnalyzer(clustering_dir)
            
            # AFTER (CORRECT - Includes config parameter):
            # Run analysis
            success = analyzer.run_full_analysis(
                distance_matrix_path=distance_matrix_path,
                similarity_matrix_path=similarity_matrix_path,
                treatment_metadata_path=treatment_metadata_path,
                reference_mad_path=reference_mad_path,
                test_landmark_path=test_landmark_path,
                config=self.config,  # ← ADD THIS LINE
                chunk_size=200
            )
            
            if success:
                logger.info(f"\nHierarchical clustering completed successfully!")
                logger.info(f"Results saved to: {clustering_dir}")
            
            return success
            
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {e}", exc_info=True)
            return False
        
    def create_viz_export(self) -> bool:
        """
        Create comprehensive visualization export file (cp_for_viz_app.csv)
        
        This combines:
        - Well-level metadata (no features)
        - UMAP/t-SNE coordinates
        - Landmark analysis results
        
        Returns:
            bool: Success status
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING VISUALIZATION EXPORT FILE")
        logger.info("="*80)
        
        try:
            exporter = VizDataExporter(self.output_dir)
            success = exporter.create_viz_export()
            
            if success:
                logger.info("✓ Visualization export file created successfully!")
                logger.info(f"  Location: {self.output_dir / 'data' / 'cp_for_viz_app.csv'}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to create visualization export: {e}", exc_info=True)
            return False
    
    def _get_embedding_coordinates_path(self, previous_run_dir: Optional[str] = None) -> Optional[Path]:
        """
        Get path to existing embedding coordinates from previous run
        
        Priority:
        1. previous_run_dir argument (if provided)
        2. skip_mode_paths in config
        3. Current output directory (if coordinates already exist there)
        4. None (coordinates not found)
        
        Args:
            previous_run_dir: Optional override path to previous run directory
        
        Returns:
            Path to embedding_coordinates.csv or None if not found
        """
        # Try previous_run_dir argument first
        if previous_run_dir:
            coords_path = Path(previous_run_dir) / "visualizations" / "coordinates" / "embedding_coordinates.csv"
            if coords_path.exists():
                logger.info(f"Found coordinates via previous_run_dir: {coords_path}")
                return coords_path
            else:
                logger.warning(f"Coordinates not found at: {coords_path}")
        
        # Try config skip_mode_paths
        if self.config and 'skip_mode_paths' in self.config:
            skip_paths = self.config['skip_mode_paths']
            base_dir = skip_paths.get('previous_run_base')
            rel_path = skip_paths.get('embedding_coordinates', 
                                      'visualizations/coordinates/embedding_coordinates.csv')
            if base_dir:
                coords_path = Path(base_dir) / rel_path
                if coords_path.exists():
                    logger.info(f"Found coordinates via config skip_mode_paths: {coords_path}")
                    return coords_path
                else:
                    logger.warning(f"Coordinates not found at config path: {coords_path}")
        
        # Try current output directory (in case we're re-running viz export)
        coords_path = self.output_dir / "visualizations" / "coordinates" / "embedding_coordinates.csv"
        if coords_path.exists():
            logger.info(f"Found coordinates in current output directory: {coords_path}")
            return coords_path
        
        logger.error("Could not locate existing embedding coordinates")
        logger.error("Tried:")
        if previous_run_dir:
            logger.error(f"  1. {previous_run_dir}/visualizations/coordinates/embedding_coordinates.csv")
        if self.config and 'skip_mode_paths' in self.config:
            logger.error(f"  2. Config skip_mode_paths")
        logger.error(f"  3. {self.output_dir}/visualizations/coordinates/embedding_coordinates.csv")
        
        return None