"""
Data loading and preparation module - MODIFIED FOR MULTI-FILE SUPPORT
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Union, List
from tqdm import tqdm

try:
    import morar
except ImportError:
    morar = None

from ..io.data_io import load_parquet_or_csv, load_metadata_csv
from ..io.config_loader import load_config, get_plate_dict
from ..utils.metadata_utils import (
        add_metadata_prefix, 
        extract_plate_well_field_info,
        map_plate_metadata,
        merge_perturbation_metadata,
        create_treatment_column,
        create_pp_id_um_column
    )

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Handles loading and initial preparation of Cell Painting data"""
    
    def __init__(self, 
                 input_file: Union[str, List[str]], 
                 metadata_file: Optional[str] = None, 
                 config_file: Optional[str] = None):
        """
        Initialize data loader
        
        Args:
            input_file: Path to main data file (Image.parquet) OR list of paths for multiple files
            metadata_file: Path to metadata CSV file
            config_file: Path to config YAML file
        """
        # MODIFIED: Accept single file or list of files
        if isinstance(input_file, str):
            self.input_files = [input_file]
        elif isinstance(input_file, list):
            self.input_files = input_file
        else:
            raise TypeError(f"input_file must be str or list, got {type(input_file)}")
        
        self.metadata_file = metadata_file
        self.config_file = config_file
        
        # Will be populated during loading
        self.config = None
        self.metadata_df = None
        self.plate_dict = {}
    
    def load_config_and_metadata(self) -> None:
        """Load configuration and metadata files"""
        print(" Loading configuration and metadata...")
        
        # Load config
        self.config = load_config(self.config_file)
        self.plate_dict = get_plate_dict(self.config)
        
        # NEW: Get metadata file from config if not provided directly
        if not self.metadata_file and self.config:
            from ..io.config_loader import get_metadata_file
            self.metadata_file = get_metadata_file(self.config)
            print(f"    Retrieved metadata file from config: {self.metadata_file}")
        
        # Load metadata
        self.metadata_df = load_metadata_csv(self.metadata_file)
        
        # DEBUG: Print metadata info
        print(f"    Metadata file: {self.metadata_file}")
        if self.metadata_df is not None:
            print(f"    Metadata shape: {self.metadata_df.shape}")
            print(f"    Metadata columns: {list(self.metadata_df.columns)}")
            if len(self.metadata_df) > 0:
                print(f"    First few rows of metadata:")
                print(self.metadata_df.head(3))
        else:
            print("    WARNING: No metadata loaded!")
        
        if self.plate_dict:
            print(f"    Loaded plate mappings for {len(self.plate_dict)} plates")
        if self.metadata_df is not None:
            print(f"    Loaded metadata with {len(self.metadata_df)} rows")
    
    def validate_multiple_inputs(self, dfs: List[pd.DataFrame]) -> bool:
        """
        NEW FUNCTION: Validate that multiple CellProfiler parquets are compatible
        
        Args:
            dfs: List of DataFrames to validate
        
        Returns:
            bool: True if compatible, False otherwise
        """
        if len(dfs) <= 1:
            return True
        
        logger.info(f"Validating {len(dfs)} input files for compatibility...")
        
        # Check 1: Same columns
        ref_cols = set(dfs[0].columns)
        for i, df in enumerate(dfs[1:], 2):
            curr_cols = set(df.columns)
            if curr_cols != ref_cols:
                missing = ref_cols - curr_cols
                extra = curr_cols - ref_cols
                logger.error(f"File {i} column mismatch!")
                if missing:
                    logger.error(f"  Missing columns: {list(missing)[:10]}")
                if extra:
                    logger.error(f"  Extra columns: {list(extra)[:10]}")
                return False
        
        logger.info(f"✓ All {len(dfs)} files have compatible columns ({len(ref_cols)} columns)")
        
        # Check 2: Verify we have URL columns for metadata extraction
        url_cols = [col for col in ref_cols if 'URL' in col or 'FileName' in col]
        if not url_cols:
            logger.warning("⚠ No URL/FileName columns found - metadata extraction may fail")
        else:
            logger.info(f"✓ Found {len(url_cols)} URL/FileName columns for metadata extraction")
        
        # Check 3: Report row counts
        for i, df in enumerate(dfs, 1):
            logger.info(f"  File {i}: {df.shape[0]:,} rows")
        
        return True
    
    def load_well_level_data_only(self) -> pd.DataFrame:
        """
        Load existing well-level data for visualization only
        MODIFIED: Now handles multiple input files
        
        Returns:
            pd.DataFrame: Well-level data
        """
        print(f" Loading well-level data from {len(self.input_files)} file(s)...")
        
        dfs = []
        for i, file_path in enumerate(self.input_files, 1):
            print(f"   Loading file {i}/{len(self.input_files)}: {Path(file_path).name}")
            with tqdm(desc=f"Loading file {i}", unit="MB") as pbar:
                df = load_parquet_or_csv(file_path)
                pbar.update(1)
            dfs.append(df)
            print(f"    Loaded {df.shape[0]:,} rows")
        
        # Merge if multiple files
        if len(dfs) > 1:
            print(f" Merging {len(dfs)} datasets...")
            df = pd.concat(dfs, axis=0, ignore_index=True)
            print(f"    Merged shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        else:
            df = dfs[0]
        
        # Identify feature columns
        feature_cols = [col for col in df.columns if not col.startswith('Metadata_')]
        metadata_cols = [col for col in df.columns if col.startswith('Metadata_')]
        logger.info(f"Feature columns: {len(feature_cols)}, Metadata columns: {len(metadata_cols)}")
        
        return df
    
    def load_and_prepare_data(self) -> 'morar.DataFrame':
        """
        Load Image.parquet file(s) and convert to morar DataFrame with all metadata
        MODIFIED: Now handles multiple input files with validation
        
        Returns:
            morar.DataFrame: Prepared data with metadata
        
        Raises:
            ImportError: If morar package not available
            ValueError: If files are incompatible for merging
            Exception: If data loading fails
        """
        if morar is None:
            raise ImportError("morar package not found. Please install it first.")
        
        print(" Starting data loading and preparation...")
        print(f" Number of input files: {len(self.input_files)}")
        for i, file_path in enumerate(self.input_files, 1):
            print(f"   File {i}: {file_path}")
        
        # Step 1: Load all parquet files
        print(" Loading parquet file(s)...")
        dfs = []
        for i, file_path in enumerate(self.input_files, 1):
            print(f" Loading file {i}/{len(self.input_files)}: {Path(file_path).name}")
            with tqdm(desc=f"Reading parquet {i}", unit="MB", ncols=80) as pbar:
                df = load_parquet_or_csv(file_path)
                pbar.update(1)
                pbar.set_description(f" File {i} loaded")
            
            print(f"    Loaded {df.shape[0]:,} rows and {df.shape[1]:,} columns")
            dfs.append(df)
        
        # Step 2: Validate compatibility if multiple files
        if len(dfs) > 1:
            if not self.validate_multiple_inputs(dfs):
                raise ValueError("Input files are not compatible for merging!")
        
        # Step 3: Merge if multiple files
        if len(dfs) > 1:
            print(f" Merging {len(dfs)} datasets...")
            with tqdm(desc="Merging dataframes", ncols=80) as pbar:
                df = pd.concat(dfs, axis=0, ignore_index=True)
                pbar.update(1)
                pbar.set_description(" Merge complete")
            print(f"    Merged shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        else:
            df = dfs[0]
        
        # Step 4: Load config and metadata
        self.load_config_and_metadata()
        
        # Step 5: Process metadata step by step with progress
        print(" Processing metadata...")
        df = self._process_metadata(df)
        
        # Step 6: Convert to morar DataFrame
        print(" Converting to morar DataFrame...")
        with tqdm(desc="Creating morar DataFrame", ncols=80) as pbar:
            morar_df = morar.DataFrame(df)
            pbar.update(1)
            pbar.set_description(" Morar conversion complete")
        
        print(f"    Final data shape: {morar_df.shape}")
        print(f"    Feature columns: {len(morar_df.featurecols)}")
        print(f"    Metadata columns: {len(morar_df.metacols)}")
        print(" Data loading completed successfully!")
        
        return morar_df
    
    def _process_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all metadata additions and mappings
        
        Args:
            df: Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with processed metadata
        """
        steps = [
            ("Adding metadata prefixes", lambda d: add_metadata_prefix(d)),
            ("Extracting plate/well/field info", lambda d: extract_plate_well_field_info(d)),
            ("Mapping plate metadata", lambda d: map_plate_metadata(d, self.plate_dict)),
            ("Merging perturbation metadata", lambda d: merge_perturbation_metadata(d, self.metadata_df)),
            ("Creating treatment column", lambda d: create_treatment_column(d)),
            ("Creating PP_ID_uM column", lambda d: create_pp_id_um_column(d))
        ]
        
        with tqdm(steps, desc="Processing metadata", ncols=80) as pbar:
            for step_name, step_func in pbar:
                pbar.set_description(f"   {step_name}")
                df = step_func(df)
                pbar.update(1)
        
        return df
    
    def get_metadata_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics about the loaded data
        
        Args:
            df: DataFrame to summarize
        
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_images': len(df),
            'n_plates': df['Metadata_plate_barcode'].nunique() if 'Metadata_plate_barcode' in df.columns else 0,
            'n_wells': df['Metadata_well'].nunique() if 'Metadata_well' in df.columns else 0,
            'n_fields': df['Metadata_field'].nunique() if 'Metadata_field' in df.columns else 0,
            'n_perturbations': 0,
            'n_dmso_images': 0
        }
        
        if 'Metadata_perturbation_name' in df.columns:
            summary['n_perturbations'] = df['Metadata_perturbation_name'].nunique()
            summary['n_dmso_images'] = len(df[df['Metadata_perturbation_name'] == 'DMSO'])
        
        # Calculate fields per well
        if summary['n_wells'] > 0:
            summary['fields_per_well'] = summary['total_images'] / summary['n_wells']
        else:
            summary['fields_per_well'] = 0
        
        return summary