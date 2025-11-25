"""
Visualization Export Module
Creates comprehensive well-level CSV for visualization applications (NO FEATURES)

This module merges:
- Well-aggregated data (metadata only, no features)
- UMAP/t-SNE coordinates
- Landmark analysis results (with ONE-to-MANY propagation)

FIXED:
- Paths constructed on-demand (prevents rogue folder creation)
- Derived columns created BEFORE filtering (ensures they're included)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class VizDataExporter:
    """Handles creation of comprehensive visualization export file"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize exporter - paths are constructed on-demand, not pre-created
        
        Args:
            output_dir: Base output directory containing all analysis results
        """
        self.output_dir = Path(output_dir)
        # DON'T create subdirectory paths here - they'll be constructed when needed
        # This prevents creating folders in wrong locations
    
    def create_viz_export(self) -> bool:
        """
        Create comprehensive well-level export file for visualization (NO FEATURES)
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("="*80)
            logger.info("CREATING VISUALIZATION EXPORT FILE (cp_for_viz_app.csv)")
            logger.info("="*80)
            logger.info("This file contains METADATA + COORDINATES + LANDMARK INFO only")
            logger.info("Features are excluded to keep file size reasonable")
            
            # Step 1: Load well-aggregated data (metadata only)
            logger.info("\nStep 1: Loading well-aggregated data (metadata only)...")
            well_data = self._load_well_metadata()
            if well_data is None:
                return False
            logger.info(f"  Loaded {len(well_data):,} wells with {len(well_data.columns)} metadata columns")

            # Step 1b: Add derived well columns
            well_data = self._add_derived_well_columns(well_data)

            # Step 1c: Add cell count
            well_data = self._add_cell_count(well_data)

            logger.info(f"  Data now has {len(well_data.columns)} columns")
            
            # Step 2: Add UMAP/t-SNE coordinates
            logger.info("\nStep 2: Adding UMAP/t-SNE coordinates...")
            well_data = self._add_embedding_coordinates(well_data)
            if well_data is None:
                return False
            logger.info(f"  Data now has {len(well_data.columns)} columns")
            
            # Step 3: Add landmark analysis results (with propagation)
            logger.info("\nStep 3: Adding landmark analysis results...")
            well_data = self._add_landmark_results(well_data)
            if well_data is None:
                return False
            logger.info(f"  Data now has {len(well_data.columns)} columns")
            
            # Step 4: Save output files
            logger.info("\nStep 4: Saving visualization export files...")
            self._save_viz_export(well_data)
            
            logger.info("\n" + "="*80)
            logger.info("✓ VISUALIZATION EXPORT COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create visualization export: {e}", exc_info=True)
            return False
    
    def _load_well_metadata(self) -> Optional[pd.DataFrame]:
        """Load well-aggregated data and keep ONLY metadata columns"""
        # Construct path on-demand
        data_dir = self.output_dir / "data"
        
        # Try both possible filenames
        possible_files = [
            data_dir / "processed_image_data_well_level.parquet"
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                logger.info(f"  Found: {file_path.name}")
                well_data = pd.read_parquet(file_path)
                
                # Verify required columns
                required_cols = ['Metadata_plate_barcode', 'Metadata_well', 'Metadata_treatment']
                missing = [col for col in required_cols if col not in well_data.columns]
                if missing:
                    logger.error(f"  Missing required columns: {missing}")
                    continue
                
                # KEEP ONLY METADATA COLUMNS (remove all features)
                metadata_cols = [col for col in well_data.columns if col.startswith('Metadata_')]
                well_data_metadata = well_data[metadata_cols].copy()
                
                n_removed = len(well_data.columns) - len(metadata_cols)
                logger.info(f"  Removed {n_removed:,} feature columns")
                logger.info(f"  Kept {len(metadata_cols)} metadata columns")
                
                return well_data_metadata
        
        logger.error(f"Could not find well-aggregated data in {data_dir}")
        logger.error(f"Tried: {[f.name for f in possible_files]}")
        return None
    
    def _add_derived_well_columns(self, well_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived columns from well metadata
        
        Adds:
        - well_row: Letter part of well (A-P)
        - well_column: Number part of well (1-24)
        
        Args:
            well_data: DataFrame with Metadata_well column
        
        Returns:
            pd.DataFrame: DataFrame with added columns
        """
        logger.info("  Adding derived well columns (well_row, well_column)...")
        
        if 'Metadata_well' not in well_data.columns:
            logger.warning("  Metadata_well column not found - skipping well_row/well_column")
            return well_data
        
        # Extract well_row (letter) and well_column (number)
        well_data['well_row'] = well_data['Metadata_well'].str.extract(r'^([A-P])', expand=False)
        well_data['well_column'] = well_data['Metadata_well'].str.extract(r'([0-9]{1,2})$', expand=False).astype('Int64')
        
        logger.info(f"    ✓ Added well_row and well_column")
        
        return well_data
    
    def _add_cell_count(self, well_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add cell count from raw image-level data
        
        Loads Count_Cytoplasm from raw data and matches to wells
        
        Args:
            well_data: DataFrame with well metadata
        
        Returns:
            pd.DataFrame: DataFrame with cell_count column added
        """
        logger.info("  Adding cell count from raw data...")
        
        # Construct path on-demand
        data_dir = self.output_dir / "data"
        raw_data_file = data_dir / "processed_image_data.parquet"
        
        if not raw_data_file.exists():
            logger.warning(f"  Raw data file not found: {raw_data_file}")
            logger.warning("  Skipping cell_count - will be NaN")
            well_data['cell_count'] = np.nan
            return well_data
        
        try:
            # Load only required columns from raw data
            logger.info(f"    Loading cell counts from: {raw_data_file.name}")
            raw_data = pd.read_parquet(
                raw_data_file,
                columns=['Metadata_plate_barcode', 'Metadata_well', 'Count_Cytoplasm']
            )
            
            # Convert plate_barcode to string for merging
            raw_data['Metadata_plate_barcode'] = raw_data['Metadata_plate_barcode'].astype(str)
            raw_data['Metadata_well'] = raw_data['Metadata_well'].astype(str)
            
            # Aggregate to well level (sum of cells across all images in well)
            well_cell_counts = raw_data.groupby(
                ['Metadata_plate_barcode', 'Metadata_well']
            )['Count_Cytoplasm'].sum().reset_index()
            
            well_cell_counts.rename(columns={'Count_Cytoplasm': 'cell_count'}, inplace=True)
            
            logger.info(f"    Calculated cell counts for {len(well_cell_counts):,} wells")
            
            # Merge with well data
            well_data = well_data.merge(
                well_cell_counts,
                on=['Metadata_plate_barcode', 'Metadata_well'],
                how='left'
            )
            
            n_matched = well_data['cell_count'].notna().sum()
            logger.info(f"    ✓ Matched cell counts for {n_matched:,} / {len(well_data):,} wells")
            
            return well_data
            
        except Exception as e:
            logger.warning(f"  Error loading cell counts: {e}")
            well_data['cell_count'] = np.nan
            return well_data
    
    def _add_embedding_coordinates(self, well_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Add UMAP/t-SNE coordinates from embedding_coordinates.csv"""
        # Construct path on-demand
        viz_dir = self.output_dir / "visualizations"
        coords_file = viz_dir / "coordinates" / "embedding_coordinates.csv"
        
        if not coords_file.exists():
            logger.error(f"Embedding coordinates file not found: {coords_file}")
            return None
        
        logger.info(f"  Loading coordinates from: {coords_file.name}")
        coords_data = pd.read_csv(coords_file)
        logger.info(f"  Loaded coordinates for {len(coords_data):,} wells")
        
        # Identify coordinate columns (exclude PCA as requested)
        umap_cols = [col for col in coords_data.columns if col.startswith('umap_')]
        tsne_cols = [col for col in coords_data.columns if col.startswith('tsne_')]
        
        logger.info(f"  Found {len(umap_cols)} UMAP coordinate columns")
        logger.info(f"  Found {len(tsne_cols)} t-SNE coordinate columns")
        logger.info(f"  Excluding PCA components (not needed for viz)")

        # Select columns to merge
        merge_cols = ['Metadata_plate_barcode', 'Metadata_well']
        cols_to_add = umap_cols + tsne_cols

        # Create merge dataframe
        coords_subset = coords_data[merge_cols + cols_to_add].copy()

        # Ensure consistent data types for merge
        logger.info("  Ensuring consistent data types for merge keys...")

        # Convert plate_barcode to string in both dataframes
        if 'Metadata_plate_barcode' in well_data.columns:
            well_data['Metadata_plate_barcode'] = well_data['Metadata_plate_barcode'].astype(str)

        if 'Metadata_plate_barcode' in coords_subset.columns:
            coords_subset['Metadata_plate_barcode'] = coords_subset['Metadata_plate_barcode'].astype(str)

        # Convert well to string in both dataframes  
        if 'Metadata_well' in well_data.columns:
            well_data['Metadata_well'] = well_data['Metadata_well'].astype(str)

        if 'Metadata_well' in coords_subset.columns:
            coords_subset['Metadata_well'] = coords_subset['Metadata_well'].astype(str)

        # Merge with well data
        logger.info(f"  Merging coordinates with well data...")
        merged_data = well_data.merge(
            coords_subset,
            on=merge_cols,
            how='left',
            suffixes=('', '_coord')
        )
        
        # Check for unmatched wells
        n_matched = merged_data[umap_cols[0]].notna().sum() if umap_cols else 0
        n_total = len(merged_data)
        logger.info(f"  Matched coordinates for {n_matched:,} / {n_total:,} wells ({100*n_matched/n_total:.1f}%)")
        
        if n_matched < n_total * 0.95:
            logger.warning(f"  Only {100*n_matched/n_total:.1f}% of wells have coordinates!")
        
        return merged_data
    
    def _add_landmark_results(self, well_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Add landmark analysis results with ONE-to-MANY propagation
        
        Merges treatment-level landmark data to well-level data.
        Includes UPDATED columns: PP_ID_uM, SMILES, compound_type, manual_annotation,
        plate_barcode, and well for all 3 landmarks.
        
        FIXED: Creates derived _compound_uM columns BEFORE filtering
        """
        # Construct path on-demand
        landmark_dir = self.output_dir / "landmark_analysis"
        
        # Load test and reference landmark results
        test_landmark_file = landmark_dir / "test_to_landmark_distances.csv"
        ref_landmark_file = landmark_dir / "reference_to_landmark_distances.csv"
        
        if not test_landmark_file.exists() and not ref_landmark_file.exists():
            logger.warning("No landmark analysis files found - skipping landmark data")
            logger.warning(f"  Looked for: {test_landmark_file.name}")
            logger.warning(f"  Looked for: {ref_landmark_file.name}")
            return well_data
        
        # Load landmark files
        landmark_dfs = []
        
        if test_landmark_file.exists():
            logger.info(f"  Loading test landmark results: {test_landmark_file.name}")
            test_landmarks = pd.read_csv(test_landmark_file)
            logger.info(f"    Loaded {len(test_landmarks):,} test treatments")
            # Add is_landmark = False for test treatments
            test_landmarks['is_landmark'] = False
            landmark_dfs.append(test_landmarks)
        
        if ref_landmark_file.exists():
            logger.info(f"  Loading reference landmark results: {ref_landmark_file.name}")
            ref_landmarks = pd.read_csv(ref_landmark_file)
            logger.info(f"    Loaded {len(ref_landmarks):,} reference treatments")
            
            # Load is_landmark from reference_mad_and_dmso.csv
            reference_mad_file = landmark_dir / "reference_mad_and_dmso.csv"
            if reference_mad_file.exists():
                logger.info(f"    Loading is_landmark status from: {reference_mad_file.name}")
                ref_mad = pd.read_csv(reference_mad_file, usecols=['treatment', 'is_landmark'])
                ref_landmarks = ref_landmarks.merge(
                    ref_mad[['treatment', 'is_landmark']],
                    on='treatment',
                    how='left'
                )
                logger.info(f"    ✓ Added is_landmark column to reference treatments")
            else:
                logger.warning(f"    ⚠ reference_mad_and_dmso.csv not found - is_landmark will be missing")
                ref_landmarks['is_landmark'] = False
            
            landmark_dfs.append(ref_landmarks)
        
        # Combine test and reference
        if len(landmark_dfs) == 0:
            logger.warning("  No landmark data loaded")
            return well_data
        
        all_landmarks = pd.concat(landmark_dfs, ignore_index=True)
        logger.info(f"  Combined landmark data: {len(all_landmarks):,} total treatments")
        
        # Define columns to keep from landmark analysis
        landmark_cols_to_keep = [
            'treatment',  # merge key
            'query_mad',
            'query_dmso_distance',
            'is_landmark',  # if present (reference only)
            'valid_for_phenotypic_makeup',  # if present (test only)
            # Query treatment's own truncated columns:
            'Metadata_annotated_target_first',
            'Metadata_annotated_target_truncated_10',
            'Metadata_annotated_target_description_truncated_10',
            'Metadata_annotated_target_first_compound_uM',  # Will be created below
        ]

        # Landmark metadata columns (for all 3 landmarks)
        landmark_prefixes = ['closest_landmark_', 'second_closest_landmark_', 'third_closest_landmark_']
        landmark_metadata_suffixes = [
            'treatment',
            'distance',
            'Metadata_plate_barcode',
            'Metadata_well',
            'Metadata_perturbation_name',
            'Metadata_chemical_name',
            'Metadata_PP_ID',
            'Metadata_PP_ID_uM',
            'Metadata_SMILES',
            'Metadata_compound_type',
            'Metadata_compound_uM',  # Needed for creating derived column
            'Metadata_manual_annotation',
            'Metadata_annotated_target',
            'Metadata_annotated_target_description',
            'Metadata_annotated_target_first',
            'Metadata_annotated_target_truncated_10',
            'Metadata_annotated_target_description_truncated_10',
            'Metadata_annotated_target_first_compound_uM',  # Will be created below
            'library'
        ]
        
        # Build full list of columns to keep
        for prefix in landmark_prefixes:
            for suffix in landmark_metadata_suffixes:
                col_name = f'{prefix}{suffix}'
                landmark_cols_to_keep.append(col_name)
        
        # ========================================================================
        # CRITICAL FIX: CREATE DERIVED COLUMNS BEFORE FILTERING
        # ========================================================================
        logger.info("  Creating derived compound_uM columns on-the-fly...")
        
        # For query treatment
        if 'Metadata_annotated_target_first' in all_landmarks.columns and 'Metadata_compound_uM' in all_landmarks.columns:
            all_landmarks['Metadata_annotated_target_first_compound_uM'] = all_landmarks.apply(
                lambda row: f"{row['Metadata_annotated_target_first']}@{row['Metadata_compound_uM']}" 
                if pd.notna(row['Metadata_annotated_target_first']) and pd.notna(row['Metadata_compound_uM'])
                else None,
                axis=1
            )
            logger.info(f"    ✓ Created Metadata_annotated_target_first_compound_uM for query treatments")
        else:
            missing = []
            if 'Metadata_annotated_target_first' not in all_landmarks.columns:
                missing.append('Metadata_annotated_target_first')
            if 'Metadata_compound_uM' not in all_landmarks.columns:
                missing.append('Metadata_compound_uM')
            logger.warning(f"    ⚠ Cannot create query column - missing: {missing}")
        
        # For each of the 3 closest landmarks
        for rank in ['closest', 'second_closest', 'third_closest']:
            target_col = f'{rank}_landmark_Metadata_annotated_target_first'
            conc_col = f'{rank}_landmark_Metadata_compound_uM'
            output_col = f'{rank}_landmark_Metadata_annotated_target_first_compound_uM'
            
            if target_col in all_landmarks.columns and conc_col in all_landmarks.columns:
                all_landmarks[output_col] = all_landmarks.apply(
                    lambda row: f"{row[target_col]}@{row[conc_col]}"
                    if pd.notna(row[target_col]) and pd.notna(row[conc_col])
                    else None,
                    axis=1
                )
                logger.info(f"    ✓ Created {output_col}")
            else:
                logger.warning(f"    ⚠ Cannot create {output_col} - missing source columns")
        
        # NOW filter to only columns that exist (AFTER creating new ones!)
        existing_landmark_cols = [col for col in landmark_cols_to_keep if col in all_landmarks.columns]
        missing_landmark_cols = [col for col in landmark_cols_to_keep if col not in all_landmarks.columns]
        
        if missing_landmark_cols:
            logger.warning(f"  Missing landmark columns: {missing_landmark_cols[:5]}")
            if len(missing_landmark_cols) > 5:
                logger.warning(f"    ... and {len(missing_landmark_cols) - 5} more")
        
        logger.info(f"  Keeping {len(existing_landmark_cols)} landmark columns (including 4 derived _compound_uM columns)")

        # Verify the derived columns are in the final list
        derived_cols = [
            'Metadata_annotated_target_first_compound_uM',
            'closest_landmark_Metadata_annotated_target_first_compound_uM',
            'second_closest_landmark_Metadata_annotated_target_first_compound_uM',
            'third_closest_landmark_Metadata_annotated_target_first_compound_uM'
        ]
        included_derived = [col for col in derived_cols if col in existing_landmark_cols]
        logger.info(f"  ✓ Derived columns included: {len(included_derived)}/4")
        for col in included_derived:
            logger.info(f"    ✓ {col}")

        # Subset landmark data
        all_landmarks_subset = all_landmarks[existing_landmark_cols].copy()
        
        # Merge with well data (ONE-to-MANY propagation)
        logger.info(f"  Merging landmark data with well data...")
        logger.info(f"  This is a ONE-to-MANY merge: treatment-level → well-level")
        
        merged_data = well_data.merge(
            all_landmarks_subset,
            left_on='Metadata_treatment',
            right_on='treatment',
            how='left',
            suffixes=('', '_landmark')
        )
        
        # Drop the redundant 'treatment' column from landmark data
        if 'treatment' in merged_data.columns:
            merged_data = merged_data.drop(columns=['treatment'])
        
        # Check merge success
        n_matched = merged_data['query_mad'].notna().sum() if 'query_mad' in merged_data.columns else 0
        n_total = len(merged_data)
        logger.info(f"  Matched landmark data for {n_matched:,} / {n_total:,} wells ({100*n_matched/n_total:.1f}%)")
        
        # Expected: DMSO wells won't have landmark data (they're the baseline)
        n_dmso = (well_data['Metadata_treatment'].str.contains('DMSO', case=False, na=False)).sum()
        n_expected_match = n_total - n_dmso
        if n_matched < n_expected_match * 0.95:
            logger.warning(f"  Expected ~{n_expected_match:,} matches (excluding DMSO), got {n_matched:,}")
        
        return merged_data
    
    def _save_viz_export(self, well_data: pd.DataFrame) -> None:
        """Save visualization export files"""
        # Construct path on-demand
        data_dir = self.output_dir / "data"
        
        # Create output filenames
        csv_path = data_dir / "cp_for_viz_app.csv"
        parquet_path = data_dir / "cp_for_viz_app.parquet"
        
        # Save CSV
        logger.info(f"  Saving CSV: {csv_path.name}")
        well_data.to_csv(csv_path, index=False)
        csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
        logger.info(f"    CSV file size: {csv_size_mb:.1f} MB")
        
        # Save Parquet
        logger.info(f"  Saving Parquet: {parquet_path.name}")
        well_data.to_parquet(parquet_path, index=False)
        parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        logger.info(f"    Parquet file size: {parquet_size_mb:.1f} MB")
        
        logger.info(f"\n  ✓ Saved {len(well_data):,} wells × {len(well_data.columns)} columns")
        logger.info(f"  ✓ Files saved to: {data_dir}")
        logger.info(f"     - {csv_path.name} ({csv_size_mb:.1f} MB)")
        logger.info(f"     - {parquet_path.name} ({parquet_size_mb:.1f} MB)")
        
        # Log column categories
        metadata_cols = [col for col in well_data.columns if col.startswith('Metadata_')]
        umap_cols = [col for col in well_data.columns if col.startswith('umap_')]
        tsne_cols = [col for col in well_data.columns if col.startswith('tsne_')]
        landmark_cols = [col for col in well_data.columns 
                        if col.startswith('query_') or col.startswith('closest_') 
                        or col.startswith('second_') or col.startswith('third_')
                        or col in ['is_landmark', 'valid_for_phenotypic_makeup']]
        
        logger.info(f"\n  Column breakdown:")
        logger.info(f"    - Metadata columns: {len(metadata_cols)}")
        logger.info(f"    - UMAP coordinates: {len(umap_cols)}")
        logger.info(f"    - t-SNE coordinates: {len(tsne_cols)}")
        logger.info(f"    - Landmark info: {len(landmark_cols)}")
        logger.info(f"    - Total: {len(well_data.columns)}")