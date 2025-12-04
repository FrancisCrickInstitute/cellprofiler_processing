"""
Data input/output utilities - Updated with organized directory structure
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_parquet_or_csv(file_path: str) -> pd.DataFrame:
    """
    Load data from parquet or CSV file, auto-detecting format
    
    Args:
        file_path: Path to data file
    
    Returns:
        pd.DataFrame: Loaded data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not supported
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading data from {file_path}")
    
    try:
        if file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            # Try parquet first, then CSV
            try:
                df = pd.read_parquet(file_path)
                logger.info("Successfully loaded as parquet format")
            except:
                df = pd.read_csv(file_path)
                logger.info("Successfully loaded as CSV format")
        
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def load_metadata_csv(metadata_file: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Load metadata CSV file with validation
    
    Args:
        metadata_file: Path to metadata CSV file
    
    Returns:
        pd.DataFrame: Metadata or None if file not found/invalid
    """
    if not metadata_file or not Path(metadata_file).exists():
        logger.warning("No metadata file provided or file doesn't exist. Proceeding without metadata.")
        return None
    
    try:
        metadata_df = pd.read_csv(metadata_file)
        logger.info(f"Loaded metadata with {metadata_df.shape[0]} rows and {metadata_df.shape[1]} columns")
        
        # Check for required columns
        required_cols = ['Metadata_lib_plate_order', 'Metadata_well']
        missing_cols = [col for col in required_cols if col not in metadata_df.columns]
        if missing_cols:
            logger.error(f"Missing required columns in metadata: {missing_cols}")
            return None
        
        logger.info(f"Available metadata columns: {list(metadata_df.columns)}")
        return metadata_df
        
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return None


def save_sample_data(data: pd.DataFrame, output_dir: Path, filename_prefix: str, n_rows: int = 10000) -> Path:
    """
    Save first n_rows of data for inspection as CSV
    
    Args:
        data: DataFrame to sample
        output_dir: Output directory
        filename_prefix: Prefix for filename
        n_rows: Number of rows to save
    
    Returns:
        Path: Path to saved file
    """
    sample_data = data.head(n_rows)
    
    csv_path = output_dir / f"{filename_prefix}_sample_{n_rows}.csv"
    sample_data.to_csv(csv_path, index=False)
    logger.info(f"Saved sample data ({n_rows} rows) to: {csv_path}")
    
    return csv_path


def save_dataset(data: pd.DataFrame, output_dir: Path, base_name: str, save_sample: bool = True) -> tuple:
    """
    Save dataset as both parquet and sample CSV to organized directories
    
    Args:
        data: DataFrame to save
        output_dir: Base output directory (will create data/ and samples/ subdirs)
        base_name: Base filename
        save_sample: Whether to save sample CSV
    
    Returns:
        tuple: (parquet_path, sample_csv_path)
    """
    # Create organized directory structure
    data_dir = output_dir / "data"
    samples_dir = output_dir / "samples"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parquet to data directory
    parquet_path = data_dir / f"{base_name}.parquet"
    data.to_parquet(parquet_path, index=False)
    logger.info(f"Saved data as parquet: {parquet_path}")
    
    # Save sample CSV to samples directory
    sample_csv_path = None
    if save_sample:
        sample_csv_path = save_sample_data(data, samples_dir, base_name)
    
    return parquet_path, sample_csv_path


def save_all_datasets(datasets: dict, output_dir: Path) -> dict:
    """
    Save all datasets with consistent naming to organized directory structure
    
    Args:
        datasets: Dictionary of {name: dataframe} pairs
        output_dir: Base output directory
    
    Returns:
        dict: Dictionary of saved file paths
    """
    logger.info("Saving all datasets to organized directory structure...")
    
    saved_files = {}
    
    # Create organized directory structure
    data_dir = output_dir / "data"
    samples_dir = output_dir / "samples"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset name mappings
    name_mappings = {
        'processed_data': 'processed_image_data',
        'normalized_data': 'processed_image_data_normalized',
        'well_aggregated_data': 'processed_image_data_well_level',
        'treatment_aggregated_data': 'processed_image_data_treatment_level'
    }
    
    for key, data in datasets.items():
        if data is not None:
            base_name = name_mappings.get(key, key)
            
            # Save parquet to data directory
            parquet_path = data_dir / f"{base_name}.parquet"
            data.to_parquet(parquet_path, index=False)
            logger.info(f"Saved data as parquet: {parquet_path}")
            saved_files[f'{key}_parquet'] = parquet_path
            
            # Save sample to samples directory
            sample_path = save_sample_data(data, samples_dir, base_name)
            saved_files[f'{key}_sample'] = sample_path
            
            # Well-level and treatment-level data also get full CSV in data directory
            if key in ['well_aggregated_data', 'treatment_aggregated_data']:
                csv_path = data_dir / f"{base_name}.csv"
                data.to_csv(csv_path, index=False)
                logger.info(f"Well-level data saved as CSV: {csv_path}")
                saved_files[f'{key}_csv'] = csv_path
    
    logger.info(f"All datasets saved to organized structure under {output_dir}")
    return saved_files