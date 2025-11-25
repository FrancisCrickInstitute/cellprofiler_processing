"""
Feature selection and cleaning utilities for Cell Painting data
UPDATED: Now includes zero MAD feature filtering to prevent normalization failures
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional

try:
    import morar
    from morar import feature_selection
except ImportError:
    morar = None
    feature_selection = None

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Handles feature selection and cleaning operations"""
    
    def __init__(self, missing_threshold: float = 0.05, 
                 correlation_threshold: float = 0.95,
                 high_variability_threshold: float = 15,
                 low_variability_threshold: float = 0.01):
        """
        Initialize feature selector with quality control thresholds
        UPDATED: Added max_zero_mad_ratio parameter
        
        Args:
            missing_threshold: Threshold for missing/zero values (default: 0.05)
            correlation_threshold: Correlation threshold (default: 0.95)
            high_variability_threshold: High variability threshold in SDs (default: 15)
            low_variability_threshold: Low variability threshold (default: 0.01)
        """
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold
        self.high_variability_threshold = high_variability_threshold
        self.low_variability_threshold = low_variability_threshold
        
        # Track removed features - UPDATED to include unwanted metadata
        self.removed_features = {
            'unwanted_metadata_columns': [],  # NEW
            'missing_features': [],
            'blacklisted_features': [],
            'correlated_features': [],
            'high_variability_features': [],
            'low_variability_features': []
        }
        
        # Store correlation matrix for later use
        self.correlation_matrix = None

    def remove_unwanted_metadata_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unwanted execution time and file path metadata columns
        This should be run BEFORE any feature selection steps
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: Data with unwanted metadata columns removed
        """
        logger.info("Removing unwanted execution time and file path metadata columns")
        
        # EXACT list of metadata columns to remove
        unwanted_metadata_columns = [
            'Metadata_ExecutionTime_08EnhanceOrSuppressFeatures',
            'Metadata_ExecutionTime_09MaskImage',
            'Metadata_ExecutionTime_10IdentifyPrimaryObjects',
            'Metadata_ExecutionTime_11EnhanceOrSuppressFeatures',
            'Metadata_ExecutionTime_12MaskImage',
            'Metadata_ExecutionTime_13IdentifyPrimaryObjects',
            'Metadata_ExecutionTime_20MeasureObjectNeighbors',
            'Metadata_ExecutionTime_21MeasureObjectNeighbors',
            'Metadata_ExecutionTime_25MeasureObjectIntensity',
            'Metadata_ExecutionTime_26MeasureObjectIntensity',
            'Metadata_ExecutionTime_30MeasureGranularity',
            'Metadata_ExecutionTime_34MeasureTexture',
            'Metadata_FileName_AGP',
            'Metadata_FileName_DNA',
            'Metadata_FileName_Mito',
            'Metadata_FileName_RNAandER',
            'Metadata_Group_Index',
            'Metadata_Group_Length',
            'Metadata_Group_Number',
            'Metadata_Height_AGP',
            'Metadata_Height_DNA',
            'Metadata_Height_Mito',
            'Metadata_MD5Digest_AGP',
            'Metadata_MD5Digest_DNA',
            'Metadata_MD5Digest_Mito',
            'Metadata_MD5Digest_RNAandER',
            'Metadata_PathName_AGP',
            'Metadata_PathName_DNA',
            'Metadata_PathName_Mito',
            'Metadata_PathName_RNAandER',
            'Metadata_Scaling_AGP',
            'Metadata_Scaling_DNA',
            'Metadata_Scaling_Mito',
            'Metadata_Scaling_RNAandER',
            'Metadata_URL_AGP',
            'Metadata_URL_DNA',
            'Metadata_URL_Mito',
            'Metadata_Width_AGP',
            'Metadata_Width_DNA',
            'Metadata_Width_Mito',
            'Metadata_Width_RNAandER',
            'Metadata_field',
            'Metadata_lib_plate_order',
            'Metadata_replicate'
        ]
        
        # Find which columns actually exist in the data
        columns_to_remove = [col for col in unwanted_metadata_columns if col in data.columns]
        
        if columns_to_remove:
            logger.info(f"Removing {len(columns_to_remove)} unwanted metadata columns:")
            for col in columns_to_remove:
                logger.info(f"  - {col}")
            
            # Track what we're removing
            self.removed_features['unwanted_metadata_columns'] = columns_to_remove
            
            data_clean = data.drop(columns=columns_to_remove)
            logger.info(f"Data shape after removal: {data_clean.shape}")
            return data_clean
        else:
            logger.info("No unwanted metadata columns found to remove")
            return data
    
    def remove_missing_features(self, data: 'morar.DataFrame') -> 'morar.DataFrame':
        """
        Step 1: Remove features with >threshold% missing or zero values
        
        Args:
            data: morar DataFrame
            
        Returns:
            morar.DataFrame: Data with missing features removed
        """
        logger.info(f"Step 1: Removing features with >{self.missing_threshold*100}% missing/zero values")
        
        # Calculate missing values percentage
        missing_pct = data.isnull().sum() / len(data)
        
        # Calculate zero values percentage for feature columns only
        feature_data = data.featuredata
        zero_pct = (feature_data == 0).sum() / len(feature_data)
        
        # Combine missing and zero percentages
        combined_pct = missing_pct.add(zero_pct, fill_value=0)
        
        # Find features to remove
        features_to_remove = combined_pct[combined_pct > self.missing_threshold].index.tolist()

        # Only remove feature columns, not metadata
        features_to_remove = [f for f in features_to_remove if f in data.featurecols]
        
        self.removed_features['missing_features'] = features_to_remove
        
        if features_to_remove:
            logger.info(f"Removing features: {features_to_remove[:10]}{'...' if len(features_to_remove) > 10 else ''}")
            data = data.drop(columns=features_to_remove)
            logger.info(f"Removed {len(features_to_remove)} features with high missing/zero values")
        else:
            logger.info("No features removed for missing/zero values")
        
        return data
    
    def remove_blacklisted_features(self, data: 'morar.DataFrame') -> 'morar.DataFrame':
        """
        Step 2: Remove blacklisted features using morar's find_unwanted_cols
        
        Args:
            data: morar DataFrame
            
        Returns:
            morar.DataFrame: Data with blacklisted features removed
        """
        logger.info("Step 2: Removing blacklisted features")
        
        if morar is None or feature_selection is None:
            logger.warning("morar package not available, skipping blacklist removal")
            return data
        
        try:
            # Use morar's feature selection to find unwanted columns
            # We have modified morar's unwanted columns function (using version 0.4, which has removes other problematic columns)
            unwanted_features = feature_selection.find_unwanted(data)
            self.removed_features['blacklisted_features'] = unwanted_features
            
            if unwanted_features:
                logger.info(f"Blacklisted features: {unwanted_features[:10]}{'...' if len(unwanted_features) > 10 else ''}")
                data = data.drop(columns=unwanted_features)
                logger.info(f"Removed {len(unwanted_features)} blacklisted features")
            else:
                logger.info("No blacklisted features found")
                
        except Exception as e:
            logger.warning(f"Could not remove blacklisted features: {e}")
            logger.info("Continuing without blacklist removal")
        
        return data
    
    def remove_correlated_features(self, data: 'morar.DataFrame') -> 'morar.DataFrame':
        """
        Step 3: Remove features with >threshold% correlation
        
        Args:
            data: morar DataFrame
            
        Returns:
            morar.DataFrame: Data with correlated features removed
        """
        logger.info(f"Step 3: Removing features with >{self.correlation_threshold*100}% correlation")
        
        try:
            feature_cols = data.featurecols
            n_features = len(feature_cols)
            logger.info(f"Computing correlation matrix for {n_features} features...")
            
            # Add progress tracking
            start_time = time.time()
            
            # Compute correlation matrix once and reuse
            feature_data = data.featuredata
            self.correlation_matrix = feature_data.corr()
            
            if morar is not None and feature_selection is not None:
                # Use morar's function
                correlated_features = feature_selection.find_correlation(
                    data, threshold=self.correlation_threshold
                )
            else:
                # Fallback implementation
                correlated_features = self._find_correlated_features_fallback(
                    self.correlation_matrix, self.correlation_threshold
                )
            
            elapsed = time.time() - start_time
            logger.info(f"Correlation computation completed in {elapsed:.1f} seconds")
            
            self.removed_features['correlated_features'] = correlated_features
            
            if correlated_features:
                logger.info(f"Highly correlated features being removed: {correlated_features[:10]}{'...' if len(correlated_features) > 10 else ''}")
                # Log which features are kept vs removed for transparency
                remaining_features = [f for f in feature_cols if f not in correlated_features]
                logger.info(f"Keeping {len(remaining_features)} features, removing {len(correlated_features)} correlated features")
                
                data = data.drop(columns=correlated_features)
                logger.info(f"Removed {len(correlated_features)} highly correlated features")
            else:
                logger.info("No highly correlated features found")
                
        except Exception as e:
            logger.warning(f"Error removing correlated features: {e}")
        
        return data
    
    def _find_correlated_features_fallback(self, corr_matrix: pd.DataFrame, 
                                         threshold: float) -> List[str]:
        """
        Fallback method to find correlated features when morar is not available
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold
            
        Returns:
            List of features to remove
        """
        # Find pairs of features with correlation > threshold
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Remove one feature from each correlated pair
        features_to_remove = set()
        for feat1, feat2 in corr_pairs:
            if feat1 not in features_to_remove:
                features_to_remove.add(feat2)
        
        return list(features_to_remove)
    
    
    def remove_high_variability_features(self, data: 'morar.DataFrame', 
                                       control_compound: str = "DMSO") -> 'morar.DataFrame':
        """
        Step 4: Remove highly variable features (>threshold SD in control compounds)
        
        Args:
            data: morar DataFrame
            control_compound: Control compound name (default: "DMSO")
            
        Returns:
            morar.DataFrame: Data with high variability features removed
        """
        logger.info(f"Step 4: Removing features with >{self.high_variability_threshold}SD in {control_compound} controls")
        
        try:
            # Find control wells
            if 'Metadata_perturbation_name' in data.columns:
                control_data = data[data['Metadata_perturbation_name'] == control_compound]
            else:
                logger.warning(f"No {control_compound} controls found, skipping high variability filter")
                return data
            
            logger.info(f"Found {len(control_data)} {control_compound} control wells")
            
            # Calculate standard deviations for feature columns
            feature_stds = control_data.featuredata.std()
            mean_stds = control_data.featuredata.mean()
            
            # Find features with high variability (>threshold * SD)
            high_var_features = []
            for feature in feature_stds.index:
                if feature_stds[feature] > self.high_variability_threshold * abs(mean_stds[feature]):
                    high_var_features.append(feature)
            
            self.removed_features['high_variability_features'] = high_var_features
            
            if high_var_features:
                logger.info(f"High variability features: {high_var_features[:10]}{'...' if len(high_var_features) > 10 else ''}")
                data = data.drop(columns=high_var_features)
                logger.info(f"Removed {len(high_var_features)} highly variable features")
            else:
                logger.info("No highly variable features found")
                
        except Exception as e:
            logger.warning(f"Error removing high variability features: {e}")
        
        return data
    
    def remove_low_variability_features(self, data: 'morar.DataFrame') -> 'morar.DataFrame':
        """
        Step 5: Remove low variability features
        
        Args:
            data: morar DataFrame
            
        Returns:
            morar.DataFrame: Data with low variability features removed
        """
        logger.info(f"Step 5: Removing features with SD < {self.low_variability_threshold}")
        
        try:
            # Calculate standard deviations for all feature columns
            feature_stds = data.featuredata.std()
            
            # Find features with low variability
            low_var_features = feature_stds[feature_stds < self.low_variability_threshold].index.tolist()
            self.removed_features['low_variability_features'] = low_var_features
            
            if low_var_features:
                logger.info(f"Low variability features: {low_var_features[:10]}{'...' if len(low_var_features) > 10 else ''}")
                data = data.drop(columns=low_var_features)
                logger.info(f"Removed {len(low_var_features)} low variability features")
            else:
                logger.info("No low variability features found")
                
        except Exception as e:
            logger.warning(f"Error removing low variability features: {e}")
        
        return data
    
    def remove_rows_with_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with missing values in feature columns (not zeros)
        
        Args:
            data: DataFrame (can be morar or pandas)
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info("Removing rows with missing values in feature columns (after feature cleaning)")
        
        # Convert to pandas for easier manipulation
        df = data.copy()
        original_rows = len(df)
        
        # Get feature columns
        if hasattr(data, 'featurecols'):
            feature_cols = data.featurecols
        else:
            feature_cols = [col for col in df.columns if not col.startswith('Metadata_')]
        
        logger.info(f"Checking {len(feature_cols)} cleaned feature columns for missing values")
        
        # Remove rows with any missing values in feature columns (NOT zeros)
        df_clean = df.dropna(subset=feature_cols)
        rows_after_na = len(df_clean)
        rows_removed_na = original_rows - rows_after_na
        
        logger.info(f"Removed {rows_removed_na:,} rows with missing values ({rows_removed_na/original_rows*100:.2f}%)")
        logger.info(f"Remaining rows: {rows_after_na:,}")
        
        return df_clean
    
    def process_all_features(self, data: 'morar.DataFrame', 
                       control_compound: str = "DMSO") -> 'morar.DataFrame':
        """
        Run all feature selection steps in sequence
        UPDATED: Now removes unwanted metadata columns first
        """
        logger.info("=== FEATURE SELECTION AND CLEANING ===")
        
        # NEW STEP 0: Remove unwanted metadata columns FIRST
        data = self.remove_unwanted_metadata_columns(data)
        
        # Step 1: Remove missing features
        data = self.remove_missing_features(data)
        
        # Step 2: Remove blacklisted features
        data = self.remove_blacklisted_features(data)
        
        # Step 3: Remove correlated features
        data = self.remove_correlated_features(data)
        
        # Step 4: Remove high variability features
        data = self.remove_high_variability_features(data, control_compound)
        
        # Step 5: Remove low variability features
        data = self.remove_low_variability_features(data)
        
        # Step 6: Remove rows with missing values
        if morar is not None:
            cleaned_data = self.remove_rows_with_missing_values(data)
            data = morar.DataFrame(cleaned_data)
        else:
            data = self.remove_rows_with_missing_values(data)
        
        return data
    
    def get_removal_summary(self) -> Dict[str, int]:
        """
        Get summary of features removed in each step
        
        Returns:
            Dictionary with removal counts for each step
        """
        summary = {}
        for step, features in self.removed_features.items():
            summary[step] = len(features)
        summary['total_removed'] = sum(summary.values())
        return summary
    


    