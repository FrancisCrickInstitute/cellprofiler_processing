"""
Metadata extraction and mapping utilities - UPDATED with Metadata_treatment creation
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional, Dict, Any

from .logging_utils import get_logger

logger = get_logger(__name__)


def add_metadata_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Metadata_ prefix to CellProfiler technical columns
    
    Args:
        df: Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with renamed columns
    """
    logger.info("Adding Metadata_ prefix to technical columns")
    
    # Define technical metadata patterns that should get Metadata_ prefix
    technical_patterns = [
        'ExecutionTime_', 'FileName_', 'Group_', 'Height_', 'ImageNumber',
        'MD5Digest_', 'PathName_', 'Scaling_', 'URL_', 'Width_'
    ]
    
    # Track renamed columns
    renamed_columns = {}
    
    for col in df.columns:
        # Check if column matches any technical pattern
        for pattern in technical_patterns:
            if col.startswith(pattern) or col == 'ImageNumber':
                new_name = f'Metadata_{col}'
                renamed_columns[col] = new_name
                break
    
    # Rename the columns
    if renamed_columns:
        df = df.rename(columns=renamed_columns)
        logger.info(f"Renamed {len(renamed_columns)} technical columns with Metadata_ prefix")
        
        # Show some examples
        for old_name, new_name in list(renamed_columns.items())[:5]:
            logger.info(f"  {old_name} -> {new_name}")
        if len(renamed_columns) > 5:
            logger.info(f"  ... and {len(renamed_columns) - 5} more")
    else:
        logger.info("No technical columns found to rename")
        
    return df


def extract_plate_well_field_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract plate barcode, well, and field information from file paths
    
    Args:
        df: Input DataFrame with URL columns
    
    Returns:
        pd.DataFrame: DataFrame with extracted metadata columns
    """
    logger.info("Extracting plate, well, and field information from file paths")
    
    # Use the specific URL column for extraction
    filename_col = 'Metadata_URL_DNA'
    if filename_col not in df.columns:
        logger.error(f"{filename_col} column not found in data")
        return df

    logger.info(f"Using {filename_col} to extract plate/well/field info")
    
    def extract_info(filepath: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Extract plate, well, field from filepath"""
        if pd.isna(filepath):
            return None, None, None
        
        # Extract plate barcode (directory name before filename)
        path_parts = str(filepath).split('/')
        plate_barcode = None
        well = None
        field = None
        
        # Find the plate barcode (5-digit number in path)
        for part in path_parts:
            if part.isdigit() and len(part) >= 5:
                plate_barcode = part
                break
        
        # Extract well and field from filename (format: WELL_FIELD_channel.extension)
        filename = path_parts[-1]
        # Updated regex to capture well and field
        well_field_match = re.match(r'^([A-P][0-9]{1,2})_([0-9]{1,2})_', filename)
        if well_field_match:
            well = well_field_match.group(1)
            field = int(well_field_match.group(2))

        return plate_barcode, well, field
    
    # Apply extraction
    extraction_results = df[filename_col].apply(extract_info)
    df['Metadata_plate_barcode'] = [result[0] for result in extraction_results]
    df['Metadata_well'] = [result[1] for result in extraction_results]
    df['Metadata_field'] = [result[2] for result in extraction_results]
    
    # Log results
    unique_plates = df['Metadata_plate_barcode'].nunique()
    unique_wells = df['Metadata_well'].nunique()
    unique_fields = df['Metadata_field'].nunique()
    logger.info(f"Extracted {unique_plates} unique plates, {unique_wells} unique wells, {unique_fields} unique fields")
    
    # Show some examples
    sample_data = df[['Metadata_plate_barcode', 'Metadata_well', 'Metadata_field', filename_col]].head()
    logger.info(f"Sample extracted data:\n{sample_data}")
    
    return df


def map_plate_metadata(df: pd.DataFrame, plate_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Map plate barcodes to metadata using plate dictionary
    
    Args:
        df: Input DataFrame
        plate_dict: Dictionary mapping plate barcodes to metadata
    
    Returns:
        pd.DataFrame: DataFrame with mapped plate metadata
    """
    if not plate_dict:
        logger.info("No plate dictionary available, skipping plate metadata mapping")
        return df
    
    logger.info("Mapping plate barcodes to experimental metadata")
    
    # Create mapping DataFrame from plate dictionary
    plate_mapping = []
    for barcode, info in plate_dict.items():
        plate_info = {'Metadata_plate_barcode': barcode}
        # ADD METADATA_ PREFIX TO ALL KEYS
        for key, value in info.items():
            plate_info[f'Metadata_{key}'] = value
        plate_mapping.append(plate_info)
    
    plate_mapping_df = pd.DataFrame(plate_mapping)
    logger.info(f"Created plate mapping with columns: {list(plate_mapping_df.columns)}")
    
    # Merge with main data
    df = df.merge(plate_mapping_df, on='Metadata_plate_barcode', how='left')
    
    # Check for unmatched plates
    unmatched = df[df['Metadata_lib_plate_order'].isna()]['Metadata_plate_barcode'].unique()
    if len(unmatched) > 0:
        logger.warning(f"Found {len(unmatched)} unmatched plate barcodes: {unmatched}")
    
    return df


def merge_perturbation_metadata(df: pd.DataFrame, metadata_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge with perturbation metadata - FIXED to include ALL columns from metadata CSV
    
    Args:
        df: Main DataFrame
        metadata_df: Metadata DataFrame with perturbation information
    
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    if metadata_df is None:
        logger.info("No metadata available, skipping perturbation metadata merge")
        return df
    
    logger.info("Merging with perturbation metadata - INCLUDING ALL COLUMNS FROM CSV")
    
    # Keep the existing merge keys (don't change merge logic)
    merge_keys = ['Metadata_lib_plate_order', 'Metadata_well']
    
    # Check if merge keys exist
    missing_keys = [key for key in merge_keys if key not in metadata_df.columns]
    if missing_keys:
        logger.error(f"Missing required merge keys in metadata: {missing_keys}")
        return df
    
    # CRITICAL FIX: Get ALL columns from metadata CSV file
    all_metadata_csv_cols = list(metadata_df.columns)
    
    logger.info(f"METADATA CSV FILE ANALYSIS:")
    logger.info(f"Total columns in metadata CSV: {len(all_metadata_csv_cols)}")
    logger.info(f"ALL columns from CSV: {all_metadata_csv_cols}")
    
    # Use ALL columns from the metadata CSV (no filtering, no hardcoded lists)
    metadata_subset = metadata_df.copy()  # Use the entire metadata dataframe
    
    # Check which columns already exist in main dataframe
    existing_cols = [col for col in all_metadata_csv_cols if col in df.columns]
    new_cols = [col for col in all_metadata_csv_cols if col not in df.columns]
    
    logger.info(f"Columns already in main data: {len(existing_cols)} - {existing_cols}")
    logger.info(f"NEW columns to be added from CSV: {len(new_cols)} - {new_cols}")
    
    # Merge ALL columns from metadata CSV
    original_shape = df.shape
    df = df.merge(metadata_subset, on=merge_keys, how='left', suffixes=('', '_from_csv'))
    
    logger.info(f"Merge completed. Shape: {original_shape} -> {df.shape}")
    
    # Verify the merge worked
    if 'Metadata_perturbation_name' in df.columns:
        unmatched = df[df['Metadata_perturbation_name'].isna()].shape[0]
        matched = len(df) - unmatched
        logger.info(f"Successfully matched: {matched:,} rows")
        logger.info(f"Unmatched rows: {unmatched:,}")
    
    # VERIFICATION: Check what metadata columns are now in the final dataframe
    final_metadata_cols = [col for col in df.columns if col.startswith('Metadata_')]
    csv_metadata_cols = [col for col in final_metadata_cols if col in all_metadata_csv_cols]
    
    logger.info(f"VERIFICATION - Metadata columns from CSV now in dataframe:")
    logger.info(f"CSV metadata columns included: {len(csv_metadata_cols)} - {csv_metadata_cols}")

    return df


def create_treatment_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    NEW: Create Metadata_treatment column combining perturbation_name and compound_uM
    
    Args:
        df: DataFrame with Metadata_perturbation_name and Metadata_compound_uM columns
    
    Returns:
        pd.DataFrame: DataFrame with new Metadata_treatment column
    """
    logger.info("Creating Metadata_treatment column...")
    
    # Check if required columns exist
    required_cols = ['Metadata_perturbation_name', 'Metadata_compound_uM']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns for treatment creation: {missing_cols}")
        return df
    
    # Create treatment column
    def create_treatment(row):
        perturbation = row['Metadata_perturbation_name']
        concentration = row['Metadata_compound_uM']
        
        # Handle missing perturbation names
        if pd.isna(perturbation) or perturbation == '' or str(perturbation).lower() == 'nan':
            return 'nan@0.0'
        else:
            # Convert concentration to string, handling potential NaN values
            if pd.isna(concentration):
                conc_str = '0.0'
            else:
                conc_str = str(float(concentration))
            return f"{perturbation}@{conc_str}"
    
    # Apply the function to create treatment column
    df['Metadata_treatment'] = df.apply(create_treatment, axis=1)
    
    # Debug information
    unique_treatments = df['Metadata_treatment'].nunique()
    sample_treatments = df['Metadata_treatment'].unique()[:10]
    
    logger.info(f"✓ CREATED Metadata_treatment column successfully!")
    logger.info(f"  Total unique treatments: {unique_treatments}")
    logger.info(f"  Sample treatments: {list(sample_treatments)}")
    
    # Show some examples with the source columns
    sample_df = df[['Metadata_perturbation_name', 'Metadata_compound_uM', 'Metadata_treatment']].head(10)
    logger.info(f"Sample treatment combinations:")
    for idx, row in sample_df.iterrows():
        logger.info(f"  {row['Metadata_perturbation_name']} + {row['Metadata_compound_uM']} -> {row['Metadata_treatment']}")
        if idx >= 4:  # Show first 5 examples
            break
    
    return df

## Create PP_ID_uM
def create_pp_id_um_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Metadata_PP_ID_uM column combining PP_ID and compound_uM
    
    Args:
        df: DataFrame with Metadata_PP_ID and Metadata_compound_uM columns
    
    Returns:
        pd.DataFrame: DataFrame with new Metadata_PP_ID_uM column
    """
    logger.info("Creating Metadata_PP_ID_uM column...")
    
    # Check if required columns exist
    required_cols = ['Metadata_PP_ID', 'Metadata_compound_uM']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing required columns for PP_ID_uM creation: {missing_cols}")
        logger.warning("Skipping PP_ID_uM column creation")
        return df
    
    # Create PP_ID_uM column
    def create_pp_id_um(row):
        pp_id = row['Metadata_PP_ID']
        concentration = row['Metadata_compound_uM']
        
        # Handle missing PP_ID
        if pd.isna(pp_id) or pp_id == '' or str(pp_id).lower() == 'nan':
            return 'Unknown@0.0'
        else:
            # Convert concentration to string, handling potential NaN values
            if pd.isna(concentration):
                conc_str = '0.0'
            else:
                conc_str = str(float(concentration))
            return f"{pp_id}@{conc_str}"
    
    # Apply the function to create PP_ID_uM column
    df['Metadata_PP_ID_uM'] = df.apply(create_pp_id_um, axis=1)
    
    # Debug information
    unique_pp_id_um = df['Metadata_PP_ID_uM'].nunique()
    sample_pp_id_um = df['Metadata_PP_ID_uM'].unique()[:10]
    
    logger.info(f"✓ CREATED Metadata_PP_ID_uM column successfully!")
    logger.info(f"  Total unique PP_ID_uM values: {unique_pp_id_um}")
    logger.info(f"  Sample PP_ID_uM values: {list(sample_pp_id_um)}")
    
    # Show some examples with the source columns
    sample_df = df[['Metadata_PP_ID', 'Metadata_compound_uM', 'Metadata_PP_ID_uM']].head(10)
    logger.info(f"Sample PP_ID_uM combinations:")
    for idx, row in sample_df.iterrows():
        logger.info(f"  {row['Metadata_PP_ID']} + {row['Metadata_compound_uM']} -> {row['Metadata_PP_ID_uM']}")
        if idx >= 4:  # Show first 5 examples
            break
    
    return df