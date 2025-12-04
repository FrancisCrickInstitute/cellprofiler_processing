"""
Normalization and scaling utilities for Cell Painting data
UPDATED: Now uses Z-score normalization (standardization) with your new morar functions
INCLUDES: Both DMSO-baseline and full-plate baseline Z-score options.
For full-plate baseline Z-scoring, this method assumes that the majority of the plate is inactive (ie, this should not be used for secondary screening or dose response testing).
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    import morar
    from morar.normalise import z_score_normalize_dmso, z_score_normalize_all_conditions
    from morar.normalise import robust_normalise  # Keep as backup option
except ImportError:
    morar = None
    z_score_normalize_dmso = None
    z_score_normalize_all_conditions = None
    robust_normalise = None

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataNormalizer:
    """Handles data normalization and scaling operations with Z-score normalization as default"""
    
    def __init__(self):
        """Initialize data normalizer"""
        pass
    
    def z_score_normalize_to_dmso(self, data: pd.DataFrame, 
                             control_compound: str = "DMSO",
                             control_column: str = "Metadata_perturbation_name") -> Optional[pd.DataFrame]:
        """
        Z-score normalization using DMSO controls as baseline (RECOMMENDED)
        
        Formula: z_score = (feature_value - DMSO_mean_per_plate) / DMSO_std_per_plate
        
        Expected results:
        - DMSO controls centered around 0
        - Treatments show deviation from normal cellular state (in standard deviations)
        - Biologically interpretable: +1.0 = 1 std dev higher than DMSO, -1.0 = 1 std dev lower
        
        Args:
            data: Input DataFrame (should be morar.DataFrame or convertible)
            control_compound: Control compound name (default: "DMSO")
            
        Returns:
            pd.DataFrame: Z-score normalized data or None if normalization fails
        """
        print("\n" + "="*80)
        print(" Z-SCORE NORMALIZATION - DMSO BASELINE (RECOMMENDED)")
        print("="*80)
        print(f"Using morar.normalise.z_score_normalize_dmso with {control_compound} baseline")
        print("Formula: z_score = (value - DMSO_mean_per_plate) / DMSO_std_per_plate")
        print("Expected result: DMSO controls centered around 0, treatments in standard deviations from normal")
        
        try:
            # Check if morar and new functions are available
            if morar is None or z_score_normalize_dmso is None:
                logger.error(" morar package or z_score_normalize_dmso function not available")
                logger.error("Make sure you've added the new functions to morar and reinstalled: pip install .")
                return None
            
            print(" morar package and z_score_normalize_dmso function available")
            
            # Check required columns
            if 'Metadata_plate_barcode' not in data.columns:
                logger.error(" Metadata_plate_barcode column not found - cannot normalize per plate")
                return None
            
            print(" Metadata_plate_barcode column found")
            
            # Use the standard perturbation column
            perturbation_col = control_column
            
            if perturbation_col not in data.columns:
                logger.error(f" {perturbation_col} column not found - required for normalization")
                available_cols = [col for col in data.columns if 'perturbation' in col.lower() or 'compound' in col.lower()]
                if available_cols:
                    logger.error(f"Available compound/perturbation columns: {available_cols}")
                return None
            
            print(f" Using perturbation column: {perturbation_col}")
            
            # Check if we have control compound data
            control_data = data[data[perturbation_col] == control_compound]
            if len(control_data) == 0:
                logger.error(f" No {control_compound} controls found in {perturbation_col} column")
                unique_compounds = data[perturbation_col].unique()[:10]
                logger.error(f"Available compounds (first 10): {unique_compounds}")
                return None
            
            plates_with_controls = control_data['Metadata_plate_barcode'].nunique()
            total_plates = data['Metadata_plate_barcode'].nunique()
            
            print(f" Found {len(control_data):,} {control_compound} control images")
            print(f" Controls present on {plates_with_controls}/{total_plates} plates")
            
            if plates_with_controls < total_plates:
                missing_plates = total_plates - plates_with_controls
                logger.warning(f"  WARNING: {missing_plates} plates have no {control_compound} controls!")
            
            # Convert to morar DataFrame if needed
            if not hasattr(data, 'featurecols'):
                print(" Converting pandas DataFrame to morar DataFrame...")
                if morar is None:
                    logger.error(" Cannot convert to morar DataFrame - morar not available")
                    return None
                morar_data = morar.DataFrame(data)
            else:
                morar_data = data
            
            print(f" Input data shape: {morar_data.shape}")
            print(f" Feature columns: {len(morar_data.featurecols):,}")
            
            # BEFORE normalization - check DMSO values
            print(f"\n BEFORE NORMALIZATION - {control_compound} Control Analysis:")
            feature_cols = [col for col in data.columns if not col.startswith('Metadata_')]
            control_before = control_data[feature_cols]
            
            if len(control_before) > 0:
                control_means_before = control_before.mean()
                control_stds_before = control_before.std()
                overall_mean_before = control_means_before.mean()
                overall_std_before = control_stds_before.mean()
                
                print(f"   {control_compound} feature means - Average: {overall_mean_before:.6f}")
                print(f"   {control_compound} feature stds - Average: {overall_std_before:.6f}")
                print(f"   Range of feature means: {control_means_before.min():.3f} to {control_means_before.max():.3f}")
            
            # Apply Z-score normalization using your new morar function
            print(f"\n Applying Z-score normalization using morar...")
            print("   1. Calculate DMSO mean and standard deviation for each plate")
            print("   2. Handle zero std with small epsilon (1e-8)")
            print("   3. Apply formula: z_score = (value - DMSO_mean) / DMSO_std")
            print("   4. Result: DMSO controls centered around 0, treatments in standard deviations")
            
            normalized_data = z_score_normalize_dmso(
                data=morar_data,
                plate_id='Metadata_plate_barcode',
                compound=perturbation_col,
                neg_compound=control_compound,
            )
            
            # Convert back to pandas DataFrame for consistency with rest of pipeline
            if hasattr(normalized_data, 'to_pandas'):
                result_df = normalized_data.to_pandas()
            else:
                result_df = normalized_data
            
            print(f"\n Z-SCORE NORMALIZATION COMPLETED")
            print(f" Output data shape: {result_df.shape}")

            # ENHANCED VALIDATION: Check that DMSO controls are centered around 0
            print(f"\n VALIDATION - {control_compound} Control Analysis AFTER Z-Score Normalization:")
            control_data_normalized = result_df[result_df[perturbation_col] == control_compound]
            
            if len(control_data_normalized) > 0:
                feature_cols = [col for col in result_df.columns if not col.startswith('Metadata_')]
                control_means_after = control_data_normalized[feature_cols].mean()
                control_stds_after = control_data_normalized[feature_cols].std()
                overall_mean_after = control_means_after.mean()
                overall_std_after = control_stds_after.mean()
                
                print(f"    {control_compound} feature means - Average: {overall_mean_after:.8f}")
                print(f"    {control_compound} feature stds - Average: {overall_std_after:.6f}")
                print(f"    Range of feature means: {control_means_after.min():.6f} to {control_means_after.max():.6f}")
                
                # Check how well centered the data is
                if abs(overall_mean_after) < 0.01:
                    print(f"    EXCELLENT: {control_compound} controls well-centered around 0 (mean={overall_mean_after:.8f})")
                elif abs(overall_mean_after) < 0.1:
                    print(f"    GOOD: {control_compound} controls reasonably centered (mean={overall_mean_after:.6f})")
                else:
                    print(f"   WARNING: {control_compound} controls not well-centered (mean={overall_mean_after:.3f})")
                
                # Check if standard deviations are reasonable (should be around 1.0 for z-scores)
                if 0.8 <= overall_std_after <= 1.2:
                    print(f"    EXCELLENT: {control_compound} control std devs near 1.0 (std={overall_std_after:.3f})")
                elif 0.5 <= overall_std_after <= 1.5:
                    print(f"    GOOD: {control_compound} control std devs reasonable (std={overall_std_after:.3f})")
                else:
                    print(f"   WARNING: {control_compound} control std devs unusual (std={overall_std_after:.3f})")
                
                # Show improvement from before normalization
                if 'overall_mean_before' in locals():
                    improvement = abs(overall_mean_before) - abs(overall_mean_after)
                    print(f"    Centering improvement: {improvement:.6f} (closer to 0 is better)")
                
                # Count total problematic values in the dataset
                total_nan = result_df[feature_cols].isnull().sum().sum()
                total_inf = np.isinf(result_df[feature_cols]).sum().sum()
                
                print(f"\n    OVERALL DATA QUALITY AFTER NORMALIZATION:")
                print(f"      Total NaN values: {total_nan:,}")
                print(f"      Total infinite values: {total_inf:,}")
                print(f"      Data shape: {result_df.shape}")
                
                if total_nan == 0 and total_inf == 0:
                    print(f"     SUCCESS: No NaN or infinite values after normalization!")
                else:
                    print(f"     WARNING: Still have {total_nan + total_inf:,} problematic values")
            
            print("="*80)
            return result_df
            
        except Exception as e:
            logger.error(f" Error during Z-score normalization (DMSO baseline): {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def z_score_normalize_all_conditions(self, data: pd.DataFrame, 
                                    control_compound: str = "DMSO",
                                    control_column: str = "Metadata_perturbation_name") -> Optional[pd.DataFrame]:
        """
        Z-score normalization using ALL CONDITIONS as baseline (ALTERNATIVE)
        
        Formula: z_score = (feature_value - all_conditions_mean_per_plate) / all_conditions_std_per_plate
        
        Expected results:
        - ALL conditions (including DMSO) centered around 0
        - Scaling based on overall plate variation, not just DMSO variation
        - All treatments show relative position within plate distribution
        
        Args:
            data: Input DataFrame (should be morar.DataFrame or convertible)
            control_compound: Control compound name (used for validation only)
            
        Returns:
            pd.DataFrame: Z-score normalized data using all conditions baseline or None if normalization fails
        """
        print("\n" + "="*80)
        print(" Z-SCORE NORMALIZATION - ALL CONDITIONS BASELINE (ALTERNATIVE)")
        print("="*80)
        print(f"Using ALL CONDITIONS as baseline (not just {control_compound})")
        print("Formula: z_score = (value - all_conditions_mean_per_plate) / all_conditions_std_per_plate")
        print("Expected result: ALL conditions (including DMSO) centered around 0")
        
        try:
            # Check if morar and new functions are available
            if morar is None or z_score_normalize_all_conditions is None:
                logger.error(" morar package or z_score_normalize_all_conditions function not available")
                logger.error("Make sure you've added the new functions to morar and reinstalled: pip install .")
                return None
            
            print(" morar package and z_score_normalize_all_conditions function available")
            
            # Check required columns (same validation as before)
            if 'Metadata_plate_barcode' not in data.columns:
                logger.error(" Metadata_plate_barcode column not found - cannot normalize per plate")
                return None
            
            perturbation_col = control_column
            
            if perturbation_col not in data.columns:
                logger.error(f" {perturbation_col} column not found - required for normalization")
                return None
            
            # Check if we have control compound data (for validation)
            control_data = data[data[perturbation_col] == control_compound]
            if len(control_data) == 0:
                logger.warning(f" No {control_compound} controls found - proceeding with all-conditions normalization anyway")
            else:
                print(f" Found {len(control_data):,} {control_compound} control images for validation")
            
            total_plates = data['Metadata_plate_barcode'].nunique()
            total_conditions = data[perturbation_col].nunique()
            
            print(f" Total plates: {total_plates}")
            print(f" Total conditions per plate: ~{total_conditions}")
            print(f" Using ALL {total_conditions} conditions as normalization baseline")
            
            # Convert to morar DataFrame if needed
            if not hasattr(data, 'featurecols'):
                print(" Converting pandas DataFrame to morar DataFrame...")
                morar_data = morar.DataFrame(data)
            else:
                morar_data = data
            
            print(f" Input data shape: {morar_data.shape}")
            print(f" Feature columns: {len(morar_data.featurecols):,}")
            
            # BEFORE normalization - show overall statistics
            print(f"\n BEFORE NORMALIZATION - Overall Data Analysis:")
            feature_cols = [col for col in data.columns if not col.startswith('Metadata_')]
            all_data = data[feature_cols]
            
            if len(all_data) > 0:
                overall_mean_before = all_data.mean().mean()
                overall_std_before = all_data.mean().std()
                
                print(f"   Overall feature means - Average: {overall_mean_before:.6f}")
                print(f"   Overall feature means - Std Dev: {overall_std_before:.6f}")
                
                # Show DMSO stats if available
                if len(control_data) > 0:
                    control_before = control_data[feature_cols]
                    dmso_mean_before = control_before.mean().mean()
                    print(f"   {control_compound} feature means - Average: {dmso_mean_before:.6f}")
            
            # Apply Z-score normalization with ALL CONDITIONS baseline
            print(f"\n Applying Z-score normalization with ALL CONDITIONS baseline...")
            print("   1. Calculate ALL CONDITIONS mean and std for each plate")
            print("   2. Handle zero std with small epsilon (1e-8)")
            print("   3. Apply formula: z_score = (value - all_conditions_mean) / all_conditions_std")
            print("   4. Result: ALL conditions centered around 0, scaled by overall plate variation")
            
            normalized_data = z_score_normalize_all_conditions(
                data=morar_data,
                plate_id='Metadata_plate_barcode',
                compound=perturbation_col,
                neg_compound=control_compound,  # Used for validation only
            )
            
            # Convert back to pandas DataFrame
            if hasattr(normalized_data, 'to_pandas'):
                result_df = normalized_data.to_pandas()
            else:
                result_df = normalized_data
            
            print(f"\n Z-SCORE NORMALIZATION (ALL CONDITIONS) COMPLETED")
            print(f" Output data shape: {result_df.shape}")
            
            # VALIDATION: Check that ALL conditions are centered around 0
            print(f"\n VALIDATION - ALL CONDITIONS Analysis AFTER Normalization:")
            all_data_normalized = result_df[feature_cols]
            
            if len(all_data_normalized) > 0:
                overall_mean_after = all_data_normalized.mean().mean()
                overall_std_after = all_data_normalized.mean().std()
                
                print(f"    ALL CONDITIONS feature means - Average: {overall_mean_after:.8f}")
                print(f"    ALL CONDITIONS feature means - Std Dev: {overall_std_after:.6f}")
                
                # Check centering
                if abs(overall_mean_after) < 0.01:
                    print(f"    EXCELLENT: ALL conditions well-centered around 0 (mean={overall_mean_after:.8f})")
                elif abs(overall_mean_after) < 0.1:
                    print(f"    GOOD: ALL conditions reasonably centered (mean={overall_mean_after:.6f})")
                else:
                    print(f"   WARNING: ALL conditions not well-centered (mean={overall_mean_after:.3f})")
                
                # Show DMSO stats after normalization
                if len(control_data) > 0:
                    control_data_normalized = result_df[result_df[perturbation_col] == control_compound]
                    dmso_mean_after = control_data_normalized[feature_cols].mean().mean()
                    print(f"    {control_compound} feature means - Average: {dmso_mean_after:.6f}")
                    print(f"    Note: {control_compound} is no longer centered at 0 (expected with all-conditions baseline)")
                
                # Count total problematic values in the dataset
                total_nan = result_df[feature_cols].isnull().sum().sum()
                total_inf = np.isinf(result_df[feature_cols]).sum().sum()
                
                print(f"\n    OVERALL DATA QUALITY AFTER NORMALIZATION:")
                print(f"      Total NaN values: {total_nan:,}")
                print(f"      Total infinite values: {total_inf:,}")
                print(f"      Data shape: {result_df.shape}")
                
                if total_nan == 0 and total_inf == 0:
                    print(f"     SUCCESS: No NaN or infinite values after normalization!")
                else:
                    print(f"     WARNING: Still have {total_nan + total_inf:,} problematic values")
            
            print("="*80)
            return result_df
            
        except Exception as e:
            logger.error(f" Error during Z-score normalization (all conditions): {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def aggregate_to_well_level(self, scaled_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Aggregate data by well (mean across fields)
        
        Args:
            scaled_data: Normalized data (from Z-score normalization)
            
        Returns:
            pd.DataFrame: Well-aggregated data or None if aggregation fails
        """
        print("\n WELL-LEVEL AGGREGATION")
        print("Aggregating data to well level (mean across fields per well)")
        
        try:
            if scaled_data is None:
                logger.error("No normalized data available for aggregation")
                return None
            
            # Get feature and metadata columns
            feature_cols = [col for col in scaled_data.columns if not col.startswith('Metadata_')]
            metadata_cols = [col for col in scaled_data.columns if col.startswith('Metadata_')]
            
            # Define grouping columns (plate + well)
            groupby_cols = ['Metadata_plate_barcode', 'Metadata_well']
            
            # Check if grouping columns exist
            missing_cols = [col for col in groupby_cols if col not in scaled_data.columns]
            if missing_cols:
                logger.error(f"Missing grouping columns: {missing_cols}")
                return None
            
            print(f" Aggregating {len(feature_cols):,} features by {groupby_cols}")
            
            # Aggregate features (mean) and metadata (first value)
            agg_dict = {}
            
            # Features: take mean
            for col in feature_cols:
                agg_dict[col] = 'mean'
            
            # Metadata: take first value (should be same within each well)
            for col in metadata_cols:
                if col not in groupby_cols:  # Don't aggregate grouping columns
                    agg_dict[col] = 'first'
            
            # Perform aggregation
            well_aggregated_data = scaled_data.groupby(groupby_cols).agg(agg_dict).reset_index()
            
            print(f" WELL AGGREGATION COMPLETED")
            print(f" Aggregated data shape: {well_aggregated_data.shape}")
            print(f" Reduced from {len(scaled_data):,} images to {len(well_aggregated_data):,} wells")
            
            return well_aggregated_data
            
        except Exception as e:
            logger.error(f"Error during well-level aggregation: {e}")
            return None
        

    def aggregate_to_treatment_level(self, well_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Aggregate well-level data to treatment level (median across wells replicates of the same treatment).
        For cols with 'Metdata_' prefix, take first instance.
        
        Args:
            well_data: Well-level data
            
        Returns:
            pd.DataFrame: Treatment-aggregated data
        """
        print("\n TREATMENT-LEVEL AGGREGATION")
        print("Aggregating data to treatment level (median across wells per treatment)")
        
        try:
            if well_data is None:
                logger.error("No well-level data available for aggregation")
                return None
            
            feature_cols = [col for col in well_data.columns if not col.startswith('Metadata_')]
            metadata_cols = [col for col in well_data.columns if col.startswith('Metadata_')]
            
            agg_dict = {}
            for col in metadata_cols:
                if col != 'Metadata_treatment':
                    agg_dict[col] = 'first'
            for col in feature_cols:
                agg_dict[col] = 'median'
            
            treatment_data = well_data.groupby('Metadata_treatment').agg(agg_dict).reset_index()
            
            print(f" TREATMENT AGGREGATION COMPLETED")
            print(f" Aggregated data shape: {treatment_data.shape}")
            print(f" Reduced from {len(well_data):,} wells to {len(treatment_data):,} treatments")
            
            return treatment_data
            
        except Exception as e:
            logger.error(f"Error during treatment-level aggregation: {e}")
            return None
    
    
    
    def process_normalization_pipeline(self, data: pd.DataFrame, 
                                     control_compound: str = "DMSO",
                                     control_column: str = "Metadata_perturbation_name",
                                     use_all_conditions_baseline: bool = False) -> tuple:
        """
        Run Z-score normalization pipeline with choice of baseline
        
        Args:
            data: Input data
            control_compound: Control compound name
            use_all_conditions_baseline: If True, use all conditions as baseline. If False, use DMSO-only baseline.
        """
        print("\n" + "="*80)
        if use_all_conditions_baseline:
            normalized_data = self.z_score_normalize_all_conditions(data, control_compound, control_column)
        else:
            normalized_data = self.z_score_normalize_to_dmso(data, control_compound, control_column)
        
        scaled_data = normalized_data  # Same thing for Z-score
        
        # Aggregate to well level
        well_aggregated_data = None
        if scaled_data is not None:
            well_aggregated_data = self.aggregate_to_well_level(scaled_data)
        
        # Aggregate to treatment level
        treatment_aggregated_data = None
        if well_aggregated_data is not None:
            treatment_aggregated_data = self.aggregate_to_treatment_level(well_aggregated_data)
        
        return normalized_data, well_aggregated_data, treatment_aggregated_data