"""
Landmark-based compound profiling analysis module

This module identifies high-quality "landmark" compounds from a reference set
and measures the similarity of all compounds to these landmarks using morphological features.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_distances

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class LandmarkAnalyzer:
    """
    Performs landmark-based compound profiling analysis on CellProfiler data
    
    Identifies high-quality reference compounds (landmarks) and measures
    similarity of all compounds to these landmarks using cosine distance.
    """
    
    def __init__(self, 
                 well_data_path: str,
                 metadata_path: str,
                 config: Dict[str, Any],
                 output_dir: str):
        """
        Initialize landmark analyzer
        
        Args:
            well_data_path: Path to well-level CellProfiler data
            metadata_path: Path to metadata CSV file
            config: Configuration dictionary (should contain plate_dict)
            output_dir: Output directory for results
        """
        self.well_data_path = Path(well_data_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.config = config
        self.output_dir = Path(output_dir)
        
        # Get thresholds from config or use defaults
        landmark_config = config.get('landmark_analysis', {})
        self.mad_threshold = landmark_config.get('mad_threshold', 0.20)
        self.dmso_percentile = landmark_config.get('dmso_percentile', 80)
        self.min_replicates = landmark_config.get('min_replicates', 2)
        
        # Data containers (populated during analysis)
        self.df = None
        self.meta_df = None
        self.feature_cols = []
        self.dmso_medianoid = None
        self.dmso_threshold = None
        
        # Results containers
        self.reference_centroids = None
        self.test_centroids = None
        self.reference_mad = None
        self.test_mad = None
        self.reference_dmso_dist = None
        self.test_dmso_dist = None
        self.landmarks = None
        self.reference_landmark_results = None
        self.test_landmark_results = None
    
    def run_full_analysis(self) -> bool:
        """
        Run complete landmark analysis pipeline
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("="*80)
            logger.info("CELLPROFILER LANDMARK ANALYSIS")
            logger.info("="*80)
            
            # Run analysis steps
            self.load_data()
            self.add_plate_metadata()
            self.merge_metadata_file()
            self.filter_and_split_data()
            self.create_treatment_centroids()
            self.calculate_dmso_medianoid()
            self.calculate_mad_for_all()
            self.calculate_dmso_distances_for_all()
            self.identify_landmarks()
            self.find_nearest_landmarks_for_all()
            self.save_all_results()
            
            logger.info("="*80)
            logger.info("LANDMARK ANALYSIS COMPLETE!")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"Landmark analysis failed: {e}", exc_info=True)
            return False
    
    def run_post_landmark_full_dist(self, landmark_dir: Path) -> bool:
        """
        Run post-landmark full distance matrix generation (Step 13 only)
        
        This method is designed to resume from a previous landmark analysis run
        where Steps 1-12 completed successfully but Step 13 failed.
        
        Requires:
            - cellprofiler_landmarks.csv (landmarks already identified)
            - reference_centroids.parquet (saved centroids with features)
            - test_centroids.parquet (saved centroids with features)
            - reference_mad_and_dmso.csv (MAD + DMSO metrics for reference)
            - test_mad_and_dmso.csv (MAD + DMSO metrics for test)
        
        Creates:
            - reference_full_landmark_distances.parquet
            - test_full_landmark_distances.parquet
        
        Args:
            landmark_dir: Directory containing landmark analysis outputs
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("="*80)
            logger.info("POST-LANDMARK FULL DISTANCE MATRIX GENERATION")
            logger.info("="*80)
            logger.info("This mode creates full treatment × landmark distance matrices")
            logger.info("from existing landmark analysis outputs (Steps 1-12 must be complete)")
            logger.info("="*80)
            
            self.output_dir = Path(landmark_dir)
            
            # Check required files
            landmarks_file = self.output_dir / 'cellprofiler_landmarks.csv'
            ref_centroids_file = self.output_dir / 'reference_centroids.parquet'
            test_centroids_file = self.output_dir / 'test_centroids.parquet'
            ref_mad_file = self.output_dir / 'reference_mad_and_dmso.csv'
            test_mad_file = self.output_dir / 'test_mad_and_dmso.csv'
            
            required_files = {
                'landmarks': landmarks_file,
                'reference_centroids': ref_centroids_file,
                'test_centroids': test_centroids_file,
                'reference_mad_and_dmso': ref_mad_file,
                'test_mad_and_dmso': test_mad_file
            }
            
            missing_files = []
            for name, filepath in required_files.items():
                if not filepath.exists():
                    missing_files.append(f"{name}: {filepath}")
            
            if missing_files:
                logger.error("Missing required files:")
                for missing in missing_files:
                    logger.error(f"  - {missing}")
                logger.error("")
                logger.error("This mode requires a previous landmark analysis run to have")
                logger.error("completed Steps 1-12 successfully, including:")
                logger.error("  - Saved centroids (Step 5)")
                logger.error("  - MAD + DMSO distance metrics (Steps 7-8)")
                return False
            
            logger.info("✓ All required files found")
            logger.info("")
            
            # Load landmarks
            logger.info("Loading landmarks...")
            self.landmarks = pd.read_csv(landmarks_file)
            logger.info(f"  Loaded {len(self.landmarks):,} landmarks")
            
            # Load centroids (which include all features)
            logger.info("\nLoading centroids...")
            self.reference_centroids = pd.read_parquet(ref_centroids_file)
            self.test_centroids = pd.read_parquet(test_centroids_file)
            
            logger.info(f"  Loaded {len(self.reference_centroids):,} reference centroids")
            logger.info(f"  Loaded {len(self.test_centroids):,} test centroids")
            
            # ========================================================================
            # CRITICAL FIX: Load MAD + DMSO distance dataframes
            # These are needed to populate query_mad and query_dmso_distance columns
            # ========================================================================
            logger.info("\nLoading MAD + DMSO distance metrics...")
            
            self.reference_mad = pd.read_csv(ref_mad_file)
            logger.info(f"  ✓ Loaded reference MAD + DMSO: {len(self.reference_mad):,} treatments")
            
            # Verify DMSO distance column exists
            if 'cosine_distance_from_dmso' in self.reference_mad.columns:
                logger.info(f"    ✓ Contains query_dmso_distance (cosine_distance_from_dmso)")
            else:
                logger.warning(f"    ⚠ Missing cosine_distance_from_dmso column!")
            
            self.test_mad = pd.read_csv(test_mad_file)
            logger.info(f"  ✓ Loaded test MAD + DMSO: {len(self.test_mad):,} treatments")
            
            # Verify DMSO distance column exists
            if 'cosine_distance_from_dmso' in self.test_mad.columns:
                logger.info(f"    ✓ Contains query_dmso_distance (cosine_distance_from_dmso)")
            else:
                logger.warning(f"    ⚠ Missing cosine_distance_from_dmso column!")
            
            # Identify feature columns
            self.feature_cols = [col for col in self.reference_centroids.columns 
                                if not col.startswith('Metadata_') and col != 'library']
            logger.info(f"\nFound {len(self.feature_cols)} feature columns")
            
            # Run Step 13 only
            logger.info("\n" + "="*80)
            logger.info("RUNNING STEP 13: CREATING FULL LANDMARK DISTANCE MATRICES")
            logger.info("="*80)
            logger.info("Each matrix will include:")
            logger.info("  - query_mad: MAD for query treatment")
            logger.info("  - query_dmso_distance: Distance to DMSO medianoid")
            logger.info("  - Distance to ALL landmarks")
            logger.info("="*80)
            
            self._create_full_landmark_distance_matrix()
            
            logger.info("\n" + "="*80)
            logger.info("✓ POST-LANDMARK FULL DISTANCE MATRIX GENERATION COMPLETE!")
            logger.info("="*80)
            logger.info(f"Outputs saved to: {self.output_dir}")
            logger.info("  - reference_full_landmark_distances.parquet (with query_dmso_distance)")
            logger.info("  - test_full_landmark_distances.parquet (with query_dmso_distance)")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"Post-landmark full distance matrix generation failed: {e}", exc_info=True)
            return False
    
    def load_data(self):
        """Load well-level CellProfiler data and metadata"""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*80)
        
        # Load CellProfiler data
        logger.info(f"Loading CellProfiler data from: {self.well_data_path}")
        self.df = pd.read_csv(self.well_data_path, low_memory=False)
        logger.info(f"Loaded {len(self.df):,} wells with {len(self.df.columns)} columns")
        
        # Load metadata if provided
        if self.metadata_path and self.metadata_path.exists():
            logger.info(f"Loading metadata from: {self.metadata_path}")
            self.meta_df = pd.read_csv(self.metadata_path)
            logger.info(f"Loaded metadata with {len(self.meta_df)} rows")
        else:
            logger.warning("No metadata file provided or file doesn't exist")
            self.meta_df = None
        
        # Identify feature columns (exclude metadata)
        exclude_patterns = ['Metadata_', 'ExecutionTime']
        self.feature_cols = [col for col in self.df.columns 
                            if not any(pattern in col for pattern in exclude_patterns)]
        logger.info(f"Using {len(self.feature_cols)} CellProfiler feature columns")
    
    def add_plate_metadata(self):
        """Check plate metadata - already present from main pipeline"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: CHECKING PLATE METADATA")
        logger.info("="*80)
        
        # Data already has library and type from main pipeline
        # Just create non-Metadata versions for landmark analysis compatibility
        
        if 'Metadata_library' in self.df.columns:
            self.df['library'] = self.df['Metadata_library']
            logger.info("✓ Using existing Metadata_library")
        else:
            logger.warning("⚠️  Metadata_library not found")
            self.df['library'] = 'Unknown'
        
        if 'Metadata_type' in self.df.columns:
            self.df['dataset_type'] = self.df['Metadata_type']
            logger.info("✓ Using existing Metadata_type")
        else:
            logger.warning("⚠️  Metadata_type not found")
            self.df['dataset_type'] = 'Unknown'
        
        # Summary
        logger.info(f"\nDataset type breakdown:")
        for dtype in self.df['dataset_type'].unique():
            count = len(self.df[self.df['dataset_type'] == dtype])
            logger.info(f"  {dtype}: {count:,} wells")
        
        logger.info(f"\nLibrary breakdown:")
        for lib in self.df['library'].unique():
            count = len(self.df[self.df['library'] == lib])
            logger.info(f"  {lib}: {count:,} wells")
    
    def merge_metadata_file(self):
        """Check metadata - merge skipped as data already contains all metadata"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: CHECKING METADATA")
        logger.info("="*80)
        
        if self.meta_df is None:
            logger.warning("No metadata file provided")
        else:
            logger.info(f"Metadata file loaded: {len(self.meta_df):,} rows")
        
        # Check if data already has required metadata columns
        logger.info("\nChecking for required metadata columns in well-level data...")
        
        required_cols = [
            'Metadata_perturbation_name',
            'Metadata_is_control', 
            'Metadata_PP_ID',
            'Metadata_PP_ID_uM',
            'Metadata_library',
            'Metadata_SMILES'
        ]
        
        present = []
        missing = []
        
        for col in required_cols:
            if col in self.df.columns:
                present.append(col)
                non_null = self.df[col].notna().sum()
                logger.info(f"  ✓ {col}: {non_null:,}/{len(self.df):,} non-null values")
            else:
                missing.append(col)
                logger.warning(f"  ✗ {col}: NOT FOUND")
        
        if missing:
            logger.error(f"\nMissing required columns: {missing}")
            logger.error("Pipeline may fail in later steps!")
        else:
            logger.info(f"\n✓ All {len(required_cols)} required metadata columns present")
            logger.info("Skipping redundant metadata merge")
        
        logger.info(f"\nFinal data shape: {self.df.shape}")
    
    def filter_and_split_data(self):
        """Filter out DMSO controls and split into reference/test sets"""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: FILTERING AND SPLITTING DATA")
        logger.info("="*80)
        
        original_count = len(self.df)
        
        # Mark DMSO controls
        self.df['is_dmso'] = self.df['Metadata_is_control'] == True
        
        # Split into DMSO and non-DMSO
        non_dmso_df = self.df[~self.df['is_dmso']].copy()
        
        logger.info(f"Total wells: {original_count:,}")
        logger.info(f"DMSO wells: {self.df['is_dmso'].sum():,}")
        logger.info(f"Non-DMSO wells: {len(non_dmso_df):,}")
        
        # Split non-DMSO into reference and test
        self.reference_df = non_dmso_df[non_dmso_df['dataset_type'] == 'reference'].copy()
        self.test_df = non_dmso_df[non_dmso_df['dataset_type'] == 'test'].copy()
        
        logger.info(f"\nReference set: {len(self.reference_df):,} wells")
        logger.info(f"Test set: {len(self.test_df):,} wells")
    
    def create_treatment_centroids(self):
        """
        Aggregate wells to treatment level using median
        
        NEW: Saves centroids to disk for resume capability
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 5: CREATING TREATMENT CENTROIDS")
        logger.info("="*80)
        
        # Check if centroids already exist
        ref_centroid_path = self.output_dir / 'reference_centroids.parquet'
        test_centroid_path = self.output_dir / 'test_centroids.parquet'
        
        if ref_centroid_path.exists() and test_centroid_path.exists():
            logger.info("✓ Centroids already exist, loading from disk...")
            self.reference_centroids = pd.read_parquet(ref_centroid_path)
            self.test_centroids = pd.read_parquet(test_centroid_path)
            logger.info(f"  Loaded {len(self.reference_centroids):,} reference centroids")
            logger.info(f"  Loaded {len(self.test_centroids):,} test centroids")
            return
        
        # Create centroids from scratch
        self.reference_centroids = self._create_centroids(self.reference_df, "REFERENCE")
        self.test_centroids = self._create_centroids(self.test_df, "TEST")
        
        # Save centroids for future resume
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("\nSaving centroids for resume capability...")
        self.reference_centroids.to_parquet(ref_centroid_path, index=False)
        self.test_centroids.to_parquet(test_centroid_path, index=False)
        
        ref_size_mb = ref_centroid_path.stat().st_size / (1024 * 1024)
        test_size_mb = test_centroid_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"  ✓ Saved reference_centroids.parquet ({ref_size_mb:.1f} MB)")
        logger.info(f"  ✓ Saved test_centroids.parquet ({test_size_mb:.1f} MB)")
        logger.info(f"  These files allow fast resume if analysis is interrupted")
    
    def _create_centroids(self, data: pd.DataFrame, label: str) -> pd.DataFrame:
        """Helper to create treatment centroids"""
        logger.info(f"\nCreating {label} treatment centroids...")
        
        # Define aggregation
        agg_dict = {}
        
        # Metadata: take first (should be same for all replicates)
        metadata_cols = [col for col in data.columns if col.startswith('Metadata_') or col == 'library']
        for col in metadata_cols:
            if col != 'Metadata_treatment':
                agg_dict[col] = 'first'
        
        # Features: take median
        for col in self.feature_cols:
            if col in data.columns:
                agg_dict[col] = 'median'
        
        # Aggregate
        centroids = data.groupby('Metadata_treatment').agg(agg_dict).reset_index()
        
        # Add replicate count
        replicate_counts = data.groupby('Metadata_treatment').size().reset_index(name='n_replicates')
        centroids = centroids.merge(replicate_counts, on='Metadata_treatment')
        
        logger.info(f"Created {len(centroids):,} {label} treatment centroids")
        
        # NEW: Create truncated and derived columns
        logger.info(f"Creating truncated metadata columns for {label}...")
        
        # 1. Metadata_annotated_target_description_truncated_10 (first 10 WORDS)
        if 'Metadata_annotated_target_description' in centroids.columns:
            centroids['Metadata_annotated_target_description_truncated_10'] = centroids['Metadata_annotated_target_description'].apply(
                lambda x: ' '.join(str(x).split()[:10]) if pd.notna(x) and len(str(x).split()) > 10 else x
            )
            logger.info(f"   Created Metadata_annotated_target_description_truncated_10 (first 10 words)")
        
        # 2. Metadata_annotated_target_first (FIRST TARGET ONLY)
        if 'Metadata_annotated_target' in centroids.columns:
            centroids['Metadata_annotated_target_first'] = centroids['Metadata_annotated_target'].apply(
                lambda x: str(x).split(',')[0].strip() if pd.notna(x) else x
            )
            logger.info(f"   Created Metadata_annotated_target_first (first target only)")
        
        # 3. Metadata_annotated_target_truncated_10 (FIRST 10 TARGETS)
        if 'Metadata_annotated_target' in centroids.columns:
            centroids['Metadata_annotated_target_truncated_10'] = centroids['Metadata_annotated_target'].apply(
                lambda x: ', '.join(str(x).split(',')[:10]) if pd.notna(x) and len(str(x).split(',')) > 10 else x
            )
            logger.info(f"   Created Metadata_annotated_target_truncated_10 (first 10 targets)")
        
       # Show library breakdown
        if 'library' in centroids.columns:
            logger.info(f"{label} library breakdown:")
            for lib in centroids['library'].unique():
                count = len(centroids[centroids['library'] == lib])
                logger.info(f"  {lib}: {count}")
        
        # NEW: Add truncated metadata columns for EACH centroid row
        logger.info(f"Adding truncated/derived columns for each {label} centroid...")
        centroids_data_list = centroids.to_dict('records')
        updated_centroids_data = []
        
        for centroid_data in centroids_data_list:
            # Add truncated columns for this centroid
            centroid_data = self._add_truncated_metadata_columns(centroid_data, prefix='')
            updated_centroids_data.append(centroid_data)
        
        # Rebuild dataframe with updated data
        centroids = pd.DataFrame(updated_centroids_data)
        logger.info(f"✓ Added truncated/derived columns for {len(centroids)} {label} centroids")
        
        return centroids
    
    def _add_truncated_metadata_columns(self, metadata_dict: dict, prefix: str = '') -> dict:
        """
        Helper function to add truncated/derived metadata columns to a dictionary
        
        Args:
            metadata_dict: Dictionary of metadata key-value pairs
            prefix: Optional prefix for column names (e.g., 'closest_landmark_')
        
        Returns:
            dict: Metadata dictionary with truncated columns added
        """
        # 1. Truncate target_description to first 10 WORDS
        target_desc_key = f'{prefix}Metadata_annotated_target_description'
        if target_desc_key in metadata_dict:
            value = metadata_dict[target_desc_key]
            if pd.notna(value):
                words = str(value).split()
                metadata_dict[f'{prefix}Metadata_annotated_target_description_truncated_10'] = (
                    ' '.join(words[:10]) if len(words) > 10 else value
                )
        
        # 2. Extract FIRST TARGET only
        target_key = f'{prefix}Metadata_annotated_target'
        if target_key in metadata_dict:
            value = metadata_dict[target_key]
            if pd.notna(value):
                targets = str(value).split(',')
                metadata_dict[f'{prefix}Metadata_annotated_target_first'] = targets[0].strip()
        
        # 3. Truncate target to first 10 TARGETS
        if target_key in metadata_dict:
            value = metadata_dict[target_key]
            if pd.notna(value):
                targets = str(value).split(',')
                metadata_dict[f'{prefix}Metadata_annotated_target_truncated_10'] = (
                    ', '.join(targets[:10]) if len(targets) > 10 else value
                )
        
        # 4. Ensure PP_ID_uM is included (should already exist from pipeline)
        pp_id_um_key = f'{prefix}Metadata_PP_ID_uM'
        if pp_id_um_key not in metadata_dict:
            # Fallback: construct from PP_ID and compound_uM if available
            pp_id_key = f'{prefix}Metadata_PP_ID'
            compound_um_key = f'{prefix}Metadata_compound_uM'
            
            if pp_id_key in metadata_dict and compound_um_key in metadata_dict:
                pp_id = metadata_dict[pp_id_key]
                conc = metadata_dict[compound_um_key]
                
                if pd.notna(pp_id):
                    conc_str = '0.0' if pd.isna(conc) else str(float(conc))
                    metadata_dict[pp_id_um_key] = f"{pp_id}@{conc_str}"
        
        # 5. NEW: Create annotated_target_first + compound_uM combination
        target_first_key = f'{prefix}Metadata_annotated_target_first'
        compound_um_key = f'{prefix}Metadata_compound_uM'
        target_first_um_key = f'{prefix}Metadata_annotated_target_first_compound_uM'
        
        if target_first_key in metadata_dict and compound_um_key in metadata_dict:
            target_first = metadata_dict[target_first_key]
            conc = metadata_dict[compound_um_key]
            
            if pd.notna(target_first):
                conc_str = '0.0' if pd.isna(conc) else str(float(conc))
                metadata_dict[target_first_um_key] = f"{target_first}@{conc_str}"
            else:
                metadata_dict[target_first_um_key] = f"Unknown@0.0"

        return metadata_dict
    
    def calculate_dmso_medianoid(self):
        """Calculate DMSO medianoid from reference set"""
        logger.info("\n" + "="*80)
        logger.info("STEP 6: CALCULATING REFERENCE DMSO MEDIANOID")
        logger.info("="*80)
        
        # Get DMSO samples from REFERENCE set only
        reference_dmso = self.df[(self.df['dataset_type'] == 'reference') & (self.df['is_dmso'] == True)]
        logger.info(f"Found {len(reference_dmso):,} DMSO wells in reference set")
        
        if len(reference_dmso) == 0:
            raise ValueError("No DMSO samples found in reference set!")
        
        # Calculate DMSO medianoid (median of all features across DMSO wells)
        self.dmso_medianoid = reference_dmso[self.feature_cols].median().values
        logger.info(f"Calculated DMSO medianoid from {len(reference_dmso)} reference DMSO wells")
        
        # Calculate DMSO distance threshold
        logger.info(f"\nCalculating DMSO distance threshold from reference DMSO samples...")
        dmso_distances = []
        for _, row in reference_dmso.iterrows():
            sample_vector = row[self.feature_cols].values
            dist = cosine(self.dmso_medianoid, sample_vector)
            dmso_distances.append(dist)
        
        self.dmso_threshold = np.percentile(dmso_distances, self.dmso_percentile)
        logger.info(f"DMSO {self.dmso_percentile}th percentile threshold: {self.dmso_threshold:.4f}")
        logger.info(f"  Mean DMSO-to-medianoid distance: {np.mean(dmso_distances):.4f}")
        logger.info(f"  Median DMSO-to-medianoid distance: {np.median(dmso_distances):.4f}")
        logger.info(f"  Min: {np.min(dmso_distances):.4f}, Max: {np.max(dmso_distances):.4f}")
    
    def calculate_mad_for_all(self):
        """Calculate MAD for both reference and test treatments"""
        logger.info("\n" + "="*80)
        logger.info("STEP 7: CALCULATING MAD (MEDIAN ABSOLUTE DEVIATION)")
        logger.info("="*80)
        
        self.reference_mad = self._calculate_mad(self.reference_df, "REFERENCE")
        self.test_mad = self._calculate_mad(self.test_df, "TEST")
    
    def _calculate_mad(self, data: pd.DataFrame, label: str) -> pd.DataFrame:
        """Helper to calculate MAD for treatments"""
        logger.info(f"\nCalculating MAD for {label} treatments...")
        
        treatments = data['Metadata_treatment'].unique()
        mad_results = []
        
        for treatment in tqdm(treatments, desc=f"Computing {label} MAD"):
            if pd.isna(treatment) or treatment == '':
                continue
            
            treatment_data = data[data['Metadata_treatment'] == treatment]
            
            # Need at least min_replicates
            if len(treatment_data) < self.min_replicates:
                continue
            
            # Create plate_well identifier
            treatment_data = treatment_data.copy()
            treatment_data['plate_well'] = (treatment_data['Metadata_plate_barcode'].astype(str) + '_' + 
                                           treatment_data['Metadata_well'].astype(str))
            
            plate_wells = treatment_data['plate_well'].unique()
            
            if len(plate_wells) < self.min_replicates:
                continue
            
            # Calculate centroids for each plate_well
            well_centroids = {}
            for pw in plate_wells:
                pw_data = treatment_data[treatment_data['plate_well'] == pw]
                if len(pw_data) > 0:
                    well_centroids[pw] = pw_data[self.feature_cols].mean().values
            
            # Calculate pairwise distances
            distances = []
            plate_well_list = list(well_centroids.keys())
            for i in range(len(plate_well_list)):
                for j in range(i+1, len(plate_well_list)):
                    dist = cosine(well_centroids[plate_well_list[i]], 
                                well_centroids[plate_well_list[j]])
                    distances.append(dist)
            
            if not distances:
                continue
            
            # Calculate MAD
            median_dist = np.median(distances)
            abs_deviations = np.abs(distances - median_dist)
            mad = np.median(abs_deviations)
            
            # Get metadata from first row (ALL metadata columns preserved)
            first_row = treatment_data.iloc[0]
            result = {
                'treatment': treatment,
                'mad_cosine': mad,
                'median_distance': median_dist,
                'well_count': len(plate_wells),
                'sample_count': len(treatment_data),
                'is_reference': (label == "REFERENCE")
            }
            
            # Add ALL metadata columns (not just hardcoded list)
            metadata_cols = [col for col in treatment_data.columns if col.startswith('Metadata_') or col == 'library']
            for col in metadata_cols:
                if col not in ['Metadata_treatment']:
                    result[col] = first_row.get(col)
            
            mad_results.append(result)
        
        mad_df = pd.DataFrame(mad_results)
        logger.info(f"Calculated MAD for {len(mad_df)} {label} treatments")
        logger.info(f"  MAD range: {mad_df['mad_cosine'].min():.4f} to {mad_df['mad_cosine'].max():.4f}")
        logger.info(f"  Mean MAD: {mad_df['mad_cosine'].mean():.4f}")
        
        return mad_df
    
    def calculate_dmso_distances_for_all(self):
        """Calculate DMSO distances for both reference and test"""
        logger.info("\n" + "="*80)
        logger.info("STEP 8: CALCULATING DMSO DISTANCES")
        logger.info("="*80)
        logger.info("NOTE: Both reference and test compounds are compared to REFERENCE DMSO medianoid")
        
        self.reference_dmso_dist = self._calculate_dmso_distances(self.reference_centroids, "REFERENCE")
        self.test_dmso_dist = self._calculate_dmso_distances(self.test_centroids, "TEST")
        
        # IMPROVEMENT: Merge MAD and DMSO distance for each set
        logger.info("\nMerging MAD and DMSO distance metrics...")
        self.reference_mad = self.reference_mad.merge(
            self.reference_dmso_dist[['treatment', 'cosine_distance_from_dmso', 'exceeds_threshold']], 
            on='treatment', 
            how='outer'
        )
        self.test_mad = self.test_mad.merge(
            self.test_dmso_dist[['treatment', 'cosine_distance_from_dmso', 'exceeds_threshold']], 
            on='treatment', 
            how='outer'
        )
        logger.info("Successfully merged MAD + DMSO distance into single dataframes")
    
    def _calculate_dmso_distances(self, centroids: pd.DataFrame, label: str) -> pd.DataFrame:
        """Helper to calculate DMSO distances"""
        logger.info(f"\n{label} set:")
        
        dmso_dist_results = []
        
        for _, row in tqdm(centroids.iterrows(), total=len(centroids), 
                          desc=f"Computing {label} DMSO distances"):
            treatment = row['Metadata_treatment']
            
            if pd.isna(treatment):
                continue
            
            # Get centroid vector
            centroid_vector = row[self.feature_cols].values
            
            # Calculate distance to reference DMSO medianoid
            dist = cosine(self.dmso_medianoid, centroid_vector)
            
            result = {
                'treatment': treatment,
                'cosine_distance_from_dmso': dist,
                'exceeds_threshold': dist > self.dmso_threshold,
                'sample_count': row.get('n_replicates', 0)
            }
            
            # Add ALL metadata columns (not just hardcoded list)
            metadata_cols = [col for col in row.index if col.startswith('Metadata_') or col == 'library']
            for col in metadata_cols:
                if col not in ['Metadata_treatment']:
                    result[col] = row.get(col)
            
            dmso_dist_results.append(result)
        
        dmso_dist_df = pd.DataFrame(dmso_dist_results).sort_values('cosine_distance_from_dmso', ascending=False)
        
        logger.info(f"  Calculated distances for {len(dmso_dist_df)} treatments")
        logger.info(f"  Distance range: {dmso_dist_df['cosine_distance_from_dmso'].min():.4f} to {dmso_dist_df['cosine_distance_from_dmso'].max():.4f}")
        logger.info(f"  Treatments exceeding threshold: {dmso_dist_df['exceeds_threshold'].sum()}")
        
        return dmso_dist_df
    
    def identify_landmarks(self):
        """Identify landmarks from reference set"""
        logger.info("\n" + "="*80)
        logger.info("STEP 9: IDENTIFYING LANDMARKS FROM REFERENCE SET")
        logger.info("="*80)
        
        logger.info(f"Landmark candidate criteria:")
        logger.info(f"  - MAD threshold: <= {self.mad_threshold}")
        logger.info(f"  - DMSO distance: > {self.dmso_threshold:.4f} (reference {self.dmso_percentile}th percentile)")
        
        # NEW: Add is_landmark boolean column to reference_mad
        self.reference_mad['is_landmark'] = (
            (self.reference_mad['mad_cosine'] <= self.mad_threshold) &
            (self.reference_mad['exceeds_threshold'] == True)
        )
        
        # Apply filters (reference_mad now has both MAD and DMSO distance)
        self.landmarks = self.reference_mad[self.reference_mad['is_landmark'] == True].copy()
        
        logger.info(f"\nIdentified {len(self.landmarks)} landmarks from {len(self.reference_mad)} reference treatments")
        logger.info(f"   Added 'is_landmark' boolean column to reference_mad_and_dmso.csv")
        
        if len(self.landmarks) > 0:
            # Show library breakdown
            if 'library' in self.landmarks.columns:
                logger.info("\nLandmark library breakdown:")
                for lib in self.landmarks['library'].unique():
                    count = len(self.landmarks[self.landmarks['library'] == lib])
                    logger.info(f"  {lib}: {count}")
            
            logger.info(f"\nLandmark statistics:")
            logger.info(f"  MAD range: {self.landmarks['mad_cosine'].min():.4f} to {self.landmarks['mad_cosine'].max():.4f}")
            logger.info(f"  DMSO distance range: {self.landmarks['cosine_distance_from_dmso'].min():.4f} to {self.landmarks['cosine_distance_from_dmso'].max():.4f}")
    
    def find_nearest_landmarks_for_all(self):
        """Find top 3 nearest landmarks for all treatments"""
        logger.info("\n" + "="*80)
        logger.info("STEP 10: FINDING TOP 3 LANDMARKS")
        logger.info("="*80)
        
        logger.info("NOTE: Excluding self-matches for reference treatments")
        
        self.reference_landmark_results = self._find_top3_landmarks(
            self.reference_centroids, exclude_self=True, is_reference=True
        )
        
        self.test_landmark_results = self._find_top3_landmarks(
            self.test_centroids, exclude_self=False, is_reference=False
        )
        
        logger.info(f"\nGenerated landmark distances for {len(self.reference_landmark_results)} reference treatments")
        logger.info(f"Generated landmark distances for {len(self.test_landmark_results)} test treatments")
        
        # NEW: Log truncated columns
        logger.info(f"\n   Added truncated/derived metadata columns to all output files:")
        logger.info(f"    - Metadata_annotated_target_description_truncated_10 (first 10 words)")
        logger.info(f"    - Metadata_annotated_target_first (first target only)")
        logger.info(f"    - Metadata_annotated_target_truncated_10 (first 10 targets)")
        logger.info(f"  These columns added for:")
        logger.info(f"    - Query treatments (reference and test)")
        logger.info(f"    - Each of the 3 closest landmarks (with prefixes)")
        
        # NEW: Log valid_for_phenotypic_makeup statistics
        if 'valid_for_phenotypic_makeup' in self.test_landmark_results.columns:
            n_valid = self.test_landmark_results['valid_for_phenotypic_makeup'].sum()
            n_total = len(self.test_landmark_results)
            logger.info(f"\n   Added 'valid_for_phenotypic_makeup' boolean column to test_to_landmark_distances.csv")
            logger.info(f"  Valid test treatments (distance < 0.2): {n_valid} / {n_total} ({100*n_valid/n_total:.1f}%)")
    
    # Modified script to implement a vectorised approach for landmark searching - much faster.

    def _find_top3_landmarks(self, query_centroids: pd.DataFrame, 
                            exclude_self: bool, is_reference: bool) -> pd.DataFrame:
        """Helper to find top 3 nearest landmarks for each query treatment (VECTORIZED)"""
        label = "REFERENCE" if is_reference else "TEST"
        logger.info(f"\nProcessing {len(query_centroids)} {label} treatments...")
        
        # Extract landmark embeddings and metadata
        landmark_embeddings = {}
        landmark_metadata = {}
        
        for _, lm_row in self.landmarks.iterrows():
            treatment = lm_row['treatment']
            
            # Get corresponding centroid from reference
            centroid_row = self.reference_centroids[
                self.reference_centroids['Metadata_treatment'] == treatment
            ]
            
            if len(centroid_row) == 0:
                continue
            
            landmark_embeddings[treatment] = centroid_row[self.feature_cols].values[0]
            landmark_metadata[treatment] = lm_row.to_dict()
        
        logger.info(f"Using {len(landmark_embeddings)} landmarks")
        
        # ========================================
        # VECTORIZED DISTANCE COMPUTATION
        # ========================================
        
        # Pre-compute landmark matrix (ONCE for all queries)
        landmark_names = list(landmark_embeddings.keys())
        landmark_matrix = np.array([landmark_embeddings[name] for name in landmark_names])
        logger.info(f"  Landmark matrix shape: {landmark_matrix.shape}")
        
        # Extract all query vectors at once
        query_matrix = query_centroids[self.feature_cols].values
        query_treatments = query_centroids['Metadata_treatment'].values
        logger.info(f"  Query matrix shape: {query_matrix.shape}")
        
        # Compute ALL distances at once (VECTORIZED!)
        logger.info("  Computing all pairwise distances...")
        all_distances = cosine_distances(query_matrix, landmark_matrix)
        logger.info(f"  Distance matrix shape: {all_distances.shape}")
        logger.info(f"  Computed {all_distances.size:,} distances in vectorized operation!")
        
        # ========================================
        # BUILD RESULTS DATAFRAME
        # ========================================
        
        results = []
        self_matches_excluded = 0
        
        logger.info("  Building results dataframe with top 3 landmarks...")
        for idx, query_row in tqdm(query_centroids.iterrows(), total=len(query_centroids),
                                desc=f"Processing {label} results"):
            query_idx = list(query_centroids.index).index(idx)
            query_treatment = query_row['Metadata_treatment']
            
            # Get pre-computed distances for this query
            query_distances = all_distances[query_idx]
            
            # Create list of (landmark_name, distance) tuples
            landmark_distances = list(zip(landmark_names, query_distances))
            
            # Handle self-matching
            if exclude_self:
                original_len = len(landmark_distances)
                landmark_distances = [(name, dist) for name, dist in landmark_distances 
                                    if name != query_treatment]
                if len(landmark_distances) < original_len:
                    self_matches_excluded += 1
            
            if not landmark_distances:
                continue
            
            # Sort by distance and get top 3
            landmark_distances.sort(key=lambda x: x[1])
            top3 = landmark_distances[:3]
            
            # Build result
            result = {
                'treatment': query_treatment,
                'is_reference': is_reference
            }
            
            # IMPROVEMENT: Add ALL query metadata (not just hardcoded subset)
            query_metadata_cols = [col for col in query_row.index 
                                if (col.startswith('Metadata_') or col == 'library') 
                                and col != 'Metadata_treatment']
            for col in query_metadata_cols:
                if col in query_row:
                    result[col] = query_row[col]
            
            # NEW: Add truncated/derived columns for query treatment metadata
            result = self._add_truncated_metadata_columns(result, prefix='')
            
            # Also add n_replicates
            if 'n_replicates' in query_row:
                result['n_replicates'] = query_row['n_replicates']
            
            # IMPROVEMENT: Add query's own MAD and DMSO distance
            if is_reference and self.reference_mad is not None:
                query_metrics = self.reference_mad[self.reference_mad['treatment'] == query_treatment]
                if len(query_metrics) > 0:
                    result['query_mad'] = query_metrics.iloc[0].get('mad_cosine')
                    result['query_dmso_distance'] = query_metrics.iloc[0].get('cosine_distance_from_dmso')
            elif not is_reference and self.test_mad is not None:
                query_metrics = self.test_mad[self.test_mad['treatment'] == query_treatment]
                if len(query_metrics) > 0:
                    result['query_mad'] = query_metrics.iloc[0].get('mad_cosine')
                    result['query_dmso_distance'] = query_metrics.iloc[0].get('cosine_distance_from_dmso')
            
            # Add top 3 landmarks with full metadata
            rank_names = ['closest', 'second_closest', 'third_closest']
            for i, (lm_treatment, lm_dist) in enumerate(top3):
                prefix = f'{rank_names[i]}_landmark_'
                
                result[f'{prefix}treatment'] = lm_treatment
                result[f'{prefix}distance'] = lm_dist
                
                # Add landmark metadata
                if lm_treatment in landmark_metadata:
                    lm_meta = landmark_metadata[lm_treatment]
                    # Add ALL landmark metadata with prefix
                    for key, value in lm_meta.items():
                        if key != 'treatment':
                            result[f'{prefix}{key}'] = value
                    
                    # NEW: Add truncated/derived columns for this landmark
                    result = self._add_truncated_metadata_columns(result, prefix=prefix)
            
            # NEW: Add valid_for_phenotypic_makeup boolean column for TEST treatments
            # Based on closest_landmark_distance < 0.2 threshold
            if not is_reference and len(top3) > 0:
                result['valid_for_phenotypic_makeup'] = top3[0][1] < 0.2
            
            results.append(result)
        
        if exclude_self:
            logger.info(f"  Excluded {self_matches_excluded} self-matches")
        
        return pd.DataFrame(results)
        
    def save_all_results(self):
        """
        Save all output files with resume capability
        
        NEW: Checks if outputs exist before creating them
        NEW: Handles Step 13 failure gracefully
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 11: SAVING OUTPUTS")
        logger.info("="*80)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save landmarks (has both MAD and DMSO distance)
        landmarks_path = self.output_dir / 'cellprofiler_landmarks.csv'
        if not landmarks_path.exists():
            self.landmarks.to_csv(landmarks_path, index=False)
            logger.info(f"1. Saved {len(self.landmarks)} landmarks to: {landmarks_path}")
        else:
            logger.info(f"1. ✓ Landmarks file already exists, skipping: {landmarks_path.name}")
        
        # 2. Save reference MAD + DMSO distance (merged) WITH is_landmark column
        ref_mad_path = self.output_dir / 'reference_mad_and_dmso.csv'
        if not ref_mad_path.exists():
            self.reference_mad.to_csv(ref_mad_path, index=False)
            logger.info(f"2. Saved reference MAD + DMSO distances to: {ref_mad_path}")
            logger.info(f"    Includes 'is_landmark' boolean column")
        else:
            logger.info(f"2. ✓ Reference MAD file already exists, skipping: {ref_mad_path.name}")
        
        # 3. Save test MAD + DMSO distance (merged)
        test_mad_path = self.output_dir / 'test_mad_and_dmso.csv'
        if not test_mad_path.exists():
            self.test_mad.to_csv(test_mad_path, index=False)
            logger.info(f"3. Saved test MAD + DMSO distances to: {test_mad_path}")
        else:
            logger.info(f"3. ✓ Test MAD file already exists, skipping: {test_mad_path.name}")
        
        # 4. Save reference landmark distances (includes query metrics)
        ref_landmark_path = self.output_dir / 'reference_to_landmark_distances.csv'
        if not ref_landmark_path.exists():
            self.reference_landmark_results.to_csv(ref_landmark_path, index=False)
            logger.info(f"4. Saved reference landmark distances to: {ref_landmark_path}")
        else:
            logger.info(f"4. ✓ Reference landmark distances already exist, skipping: {ref_landmark_path.name}")
        
        # 5. Save test landmark distances (includes query metrics) WITH valid_for_phenotypic_makeup column
        test_landmark_path = self.output_dir / 'test_to_landmark_distances.csv'
        if not test_landmark_path.exists():
            self.test_landmark_results.to_csv(test_landmark_path, index=False)
            logger.info(f"5. Saved test landmark distances to: {test_landmark_path}")
            if 'valid_for_phenotypic_makeup' in self.test_landmark_results.columns:
                logger.info(f"    Includes 'valid_for_phenotypic_makeup' boolean column (threshold < 0.2)")
        else:
            logger.info(f"5. ✓ Test landmark distances already exist, skipping: {test_landmark_path.name}")
            if 'valid_for_phenotypic_makeup' in self.test_landmark_results.columns:
                logger.info(f"    (File includes 'valid_for_phenotypic_makeup' boolean column)")

        # 6. Compute and save cosine distance matrix for hierarchical clustering
        distance_matrix_path = self.output_dir / 'cosine_distance_matrix_for_clustering.parquet'
        similarity_matrix_path = self.output_dir / 'cosine_similarity_matrix_for_clustering.parquet'
        metadata_path = self.output_dir / 'treatment_metadata_for_clustering.csv'
        
        if not distance_matrix_path.exists() or not similarity_matrix_path.exists() or not metadata_path.exists():
            logger.info("\n6. Computing cosine distance/similarity matrices...")
            self._compute_and_save_distance_matrix()
        else:
            logger.info(f"\n6. ✓ Distance matrices already exist, skipping:")
            logger.info(f"    - {distance_matrix_path.name}")
            logger.info(f"    - {similarity_matrix_path.name}")
            logger.info(f"    - {metadata_path.name}")

        # 7. Create full landmark distance matrix (ALL treatments × ALL landmarks)
        # This is Step 13 - the one that was failing
        ref_full_matrix_path = self.output_dir / 'reference_full_landmark_distances.parquet'
        test_full_matrix_path = self.output_dir / 'test_full_landmark_distances.parquet'
        
        if not ref_full_matrix_path.exists() or not test_full_matrix_path.exists():
            logger.info("\n7. Creating full landmark distance matrices (Step 13)...")
            try:
                self._create_full_landmark_distance_matrix()
                logger.info("    ✓ Full landmark distance matrices created successfully!")
            except Exception as e:
                logger.error(f"    ✗ Step 13 failed (full landmark matrices): {e}", exc_info=True)
                logger.warning("")
                logger.warning("="*80)
                logger.warning("⚠️  STEP 13 FAILED BUT OTHER OUTPUTS ARE VALID")
                logger.warning("="*80)
                logger.warning("You can re-run landmark mode to retry Step 13.")
                logger.warning("All previous steps (1-12) will be skipped automatically.")
                logger.warning("="*80)
                logger.warning("")
        else:
            logger.info(f"\n7. ✓ Full landmark matrices already exist, skipping:")
            logger.info(f"    - {ref_full_matrix_path.name}")
            logger.info(f"    - {test_full_matrix_path.name}")

        # Summary statistics
        self._print_summary_statistics()
    
    def _compute_and_save_distance_matrix(self):
        """
        Compute cosine distance matrix for ALL treatments (reference + test)
        This matrix can be used directly by the hierarchical clustering script
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 12: COMPUTING COSINE DISTANCE MATRIX FOR HIERARCHICAL CLUSTERING")
        logger.info("="*80)
        
        # Combine reference and test centroids
        all_centroids = pd.concat([self.reference_centroids, self.test_centroids], ignore_index=True)
        logger.info(f"Total treatments for distance matrix: {len(all_centroids)}")
        
        # Filter out DMSO treatments
        all_centroids_no_dmso = all_centroids[
            ~all_centroids['Metadata_treatment'].str.contains('DMSO', case=False, na=False)
        ]
        dmso_filtered = len(all_centroids) - len(all_centroids_no_dmso)
        logger.info(f"Filtered out {dmso_filtered} DMSO treatments")
        logger.info(f"Computing distance matrix for {len(all_centroids_no_dmso)} non-DMSO treatments")
        
        # Extract feature vectors
        feature_matrix = all_centroids_no_dmso[self.feature_cols].values
        treatment_names = all_centroids_no_dmso['Metadata_treatment'].values
        
        # Compute pairwise cosine distances
        logger.info("Computing pairwise cosine distances...")
        from scipy.spatial.distance import pdist, squareform
        distance_vector = pdist(feature_matrix, metric='cosine')
        distance_matrix = squareform(distance_vector)
        
        # Convert to similarity (1 - distance)
        similarity_matrix = 1 - distance_matrix
        
        # Create DataFrame with treatment names as index/columns
        distance_df = pd.DataFrame(
            distance_matrix,
            index=treatment_names,
            columns=treatment_names
        )
        
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=treatment_names,
            columns=treatment_names
        )
        
        # Save both distance and similarity matrices
        distance_path = self.output_dir / 'cosine_distance_matrix_for_clustering.parquet'
        similarity_path = self.output_dir / 'cosine_similarity_matrix_for_clustering.parquet'
        
        distance_df.to_parquet(distance_path)
        similarity_df.to_parquet(similarity_path)
        
        logger.info(f" Saved cosine distance matrix: {distance_path}")
        logger.info(f" Saved cosine similarity matrix: {similarity_path}")
        logger.info(f"   Matrix shape: {distance_matrix.shape}")
        logger.info(f"   Distance range: {distance_matrix.min():.4f} to {distance_matrix.max():.4f}")
        logger.info(f"   Similarity range: {similarity_matrix.min():.4f} to {similarity_matrix.max():.4f}")
        
        # Also save treatment metadata for the clustering script
        metadata_cols = [col for col in all_centroids_no_dmso.columns 
                        if col.startswith('Metadata_') or col == 'library' or col == 'n_replicates']
        treatment_metadata = all_centroids_no_dmso[metadata_cols].copy()
        
        metadata_path = self.output_dir / 'treatment_metadata_for_clustering.csv'
        treatment_metadata.to_csv(metadata_path, index=False)
        logger.info(f" Saved treatment metadata: {metadata_path}")
        logger.info(f"   Metadata columns: {len(metadata_cols)}")
        
        logger.info("\nThese matrices can be directly used by hierarchical_chunk_clustering.py")



    def _create_full_landmark_distance_matrix(self):
        """
        Create full distance matrix: ALL treatments × ALL landmarks (SLIM VERSION)
        
        This creates TWO types of files:
        1. Distance matrices (treatment × landmark distances only)
        2. Landmark metadata (saved once, not repeated)
        
        Output files:
        - reference_distances.parquet (slim, distances only)
        - test_distances.parquet (slim, distances only)
        - landmark_metadata.parquet (saved once, ~50MB)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 13: CREATING SLIM LANDMARK DISTANCE MATRICES")
        logger.info("="*80)
        logger.info("NEW FORMAT: Distances + landmark metadata saved separately")
        logger.info("Expected size reduction: 68GB → 3GB per file!")
        logger.info("="*80)
        
        # Combine reference and test centroids to get all treatment data with features
        all_treatment_data = pd.concat([self.reference_centroids, self.test_centroids], ignore_index=True)
        logger.info(f"Combined {len(all_treatment_data):,} total treatments (reference + test)")
        
        # Get landmark embeddings and metadata
        landmark_embeddings = {}
        landmark_metadata_list = []
        
        for idx, lm_row in self.landmarks.iterrows():
            lm_treatment = lm_row['treatment']
            
            # Get features from all_treatment_data
            lm_treatment_data = all_treatment_data[all_treatment_data['Metadata_treatment'] == lm_treatment]
            if len(lm_treatment_data) > 0:
                lm_vector = lm_treatment_data.iloc[0][self.feature_cols].values
                landmark_embeddings[lm_treatment] = lm_vector
                
                # Store metadata for this landmark
                landmark_meta = {'treatment': lm_treatment}
                
                # Add ALL metadata columns
                metadata_cols = [col for col in lm_row.index 
                                if (col.startswith('Metadata_') or col == 'library') 
                                and col != 'treatment']
                for col in metadata_cols:
                    landmark_meta[col] = lm_row[col]
                
                # Add derived columns
                landmark_meta = self._add_truncated_metadata_columns(landmark_meta, prefix='')
                
                landmark_metadata_list.append(landmark_meta)
            else:
                logger.warning(f"Landmark treatment '{lm_treatment}' not found in treatment data")
        
        n_landmarks = len(landmark_embeddings)
        logger.info(f"Using {n_landmarks} landmarks for distance matrix")
        
        # ========================================================================
        # SAVE LANDMARK METADATA ONCE (separate file)
        # ========================================================================
        logger.info("\nSaving landmark metadata (saved once, not repeated)...")
        landmark_metadata_df = pd.DataFrame(landmark_metadata_list)
        
        # Sort by treatment name for consistent ordering across test and reference files
        landmark_metadata_df = landmark_metadata_df.sort_values('treatment').reset_index(drop=True)
        
        landmark_meta_path = self.output_dir / 'landmark_metadata.parquet'
        landmark_metadata_df.to_parquet(landmark_meta_path, index=False)
        
        # ALSO save CSV sample (first 200 rows)
        landmark_meta_csv_path = self.output_dir / 'landmark_metadata_sample.csv'
        landmark_metadata_df.head(200).to_csv(landmark_meta_csv_path, index=False)
        logger.info(f"✓ Saved landmark metadata sample CSV (first 200 rows): {landmark_meta_csv_path}")
        
        landmark_size_mb = landmark_meta_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Saved landmark metadata: {landmark_meta_path}")
        logger.info(f"  Size: {landmark_size_mb:.1f} MB (only {len(landmark_metadata_df)} rows)")
        logger.info(f"  Columns: {len(landmark_metadata_df.columns)}")
        logger.info(f"  This file contains metadata for ALL landmarks")
        logger.info(f"  Landmarks are sorted alphabetically by treatment name for consistent ordering")
        
        # Prepare landmark embeddings/metadata in the correct order
        landmark_names_ordered = landmark_metadata_df['treatment'].tolist()
        landmark_embeddings_ordered = {name: landmark_embeddings[name] for name in landmark_names_ordered}
        landmark_metadata_ordered = {name: row.to_dict() for idx, row in landmark_metadata_df.iterrows() 
                                    for name in [row['treatment']]}
        
        # ========================================================================
        # Process reference compounds
        # ========================================================================
        logger.info("\nProcessing REFERENCE compounds...")
        reference_matrix = self._build_full_distance_matrix(
            self.reference_centroids,
            landmark_embeddings_ordered,
            landmark_metadata_ordered,
            is_reference=True,
            exclude_self=True
        )
        
        if reference_matrix is not None:
            # Keep float64 precision (pandas default) to prevent rounding errors
            # File size: ~560 MB vs ~140 MB with float16, but maintains full precision
            # This prevents ranking inconsistencies in top 3 landmark matching
            distance_cols = [col for col in reference_matrix.columns if col.endswith('_distance')]
            logger.info(f"  Keeping {len(distance_cols)} distance columns as float64 for full precision")
            
            ref_path = self.output_dir / 'reference_distances.parquet'
            reference_matrix.to_parquet(ref_path, index=False)
            
            # ALSO save CSV sample (first 200 rows)
            ref_csv_path = self.output_dir / 'reference_distances_sample.csv'
            reference_matrix.head(200).to_csv(ref_csv_path, index=False)
            logger.info(f"✓ Saved reference distances sample CSV (first 200 rows): {ref_csv_path}")
            
            ref_size_mb = ref_path.stat().st_size / (1024 * 1024)
            ref_size_gb = ref_size_mb / 1024
            
            logger.info(f"✓ Saved reference distances: {ref_path}")
            logger.info(f"  Matrix shape: {reference_matrix.shape}")
            logger.info(f"  File size: {ref_size_gb:.2f} GB (was ~68 GB in old format!)")
            logger.info(f"  Space saved: ~{68 - ref_size_gb:.1f} GB ({100 * (68 - ref_size_gb) / 68:.1f}% reduction)")
        
        # ========================================================================
        # Process test compounds
        # ========================================================================
        logger.info("\nProcessing TEST compounds...")
        test_matrix = self._build_full_distance_matrix(
            self.test_centroids,
            landmark_embeddings_ordered,
            landmark_metadata_ordered,
            is_reference=False,
            exclude_self=False
        )
        
        if test_matrix is not None:
            # Keep float64 precision (pandas default) to prevent rounding errors
            # File size: ~366 MB vs ~92 MB with float16, but maintains full precision
            # This prevents ranking inconsistencies in top 3 landmark matching
            distance_cols = [col for col in test_matrix.columns if col.endswith('_distance')]
            logger.info(f"  Keeping {len(distance_cols)} distance columns as float64 for full precision")
            
            test_path = self.output_dir / 'test_distances.parquet'
            test_matrix.to_parquet(test_path, index=False)
            
            # ALSO save CSV sample (first 200 rows)
            test_csv_path = self.output_dir / 'test_distances_sample.csv'
            test_matrix.head(200).to_csv(test_csv_path, index=False)
            logger.info(f"✓ Saved test distances sample CSV (first 200 rows): {test_csv_path}")
            
            test_size_mb = test_path.stat().st_size / (1024 * 1024)
            test_size_gb = test_size_mb / 1024
            
            logger.info(f"✓ Saved test distances: {test_path}")
            logger.info(f"  Matrix shape: {test_matrix.shape}")
            logger.info(f"  File size: {test_size_gb:.2f} GB (was ~47 GB in old format!)")
            logger.info(f"  Space saved: ~{47 - test_size_gb:.1f} GB ({100 * (47 - test_size_gb) / 47:.1f}% reduction)")
        
        # ========================================================================
        # Summary
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("✓ SLIM LANDMARK DISTANCE MATRICES CREATED!")
        logger.info("="*80)
        logger.info(f"Output files:")
        logger.info(f"  1. landmark_metadata.parquet ({landmark_size_mb:.1f} MB)")
        logger.info(f"     - {len(landmark_metadata_df)} landmarks with full metadata")
        logger.info(f"     - Landmarks sorted alphabetically by treatment for consistency")
        logger.info(f"     - landmark_metadata_sample.csv (first 200 rows)")
        logger.info(f"")
        logger.info(f"  2. reference_distances.parquet ({ref_size_gb:.2f} GB)")
        logger.info(f"     - {len(reference_matrix):,} treatments × {n_landmarks:,} distances")
        logger.info(f"     - NEW FORMAT: Each landmark = 'treatment_distance' column")
        logger.info(f"     - Example: 'TreatmentXXXX@1.0_distance'")
        logger.info(f"     - Includes query metadata + query_dmso_distance")
        logger.info(f"     - reference_distances_sample.csv (first 200 rows)")
        logger.info(f"")
        logger.info(f"  3. test_distances.parquet ({test_size_gb:.2f} GB)")
        logger.info(f"     - {len(test_matrix):,} treatments × {n_landmarks:,} distances")
        logger.info(f"     - NEW FORMAT: Each landmark = 'treatment_distance' column")
        logger.info(f"     - Example: 'TreatmentXXXX@1.0_distance'")
        logger.info(f"     - Includes query metadata + query_dmso_distance")
        logger.info(f"     - test_distances_sample.csv (first 200 rows)")
        logger.info(f"")
        total_saved = (68 + 47) - (ref_size_gb + test_size_gb + landmark_size_mb / 1024)
        logger.info(f"Total space saved: ~{total_saved:.1f} GB ({100 * total_saved / 115:.1f}% reduction)")
        logger.info("="*80)
        logger.info("")
        logger.info("KEY IMPROVEMENTS:")
        logger.info("  ✓ Single column per landmark (no redundant treatment columns)")
        logger.info("  ✓ Landmarks ordered alphabetically (same across test & reference)")
        logger.info("  ✓ CSV samples for manual inspection (200 rows each)")
        logger.info("="*80)


    def _build_full_distance_matrix(self, query_centroids, landmark_embeddings, 
                            landmark_metadata, is_reference=False, exclude_self=True):
        """
        Build full distance matrix for a set of query compounds (SLIM VERSION)
        
        NEW FORMAT: Each landmark gets a single column named {treatment}_distance
        e.g., 'TreatmentXXXX@1.0_distance' instead of 'lm_1_dist' + 'lm_1_treatment'
        
        This version only stores distances, not landmark metadata in each row.
        Landmark metadata is saved separately in landmark_metadata.parquet
        
        Args:
            query_centroids: DataFrame of query treatments
            landmark_embeddings: Dict of {landmark_name: embedding_vector}
            landmark_metadata: Dict of {landmark_name: metadata_dict}
            is_reference: Whether these are reference compounds
            exclude_self: Whether to exclude self-matches (for reference compounds)
        
        Returns:
            pd.DataFrame: Slim matrix with distances only (treatment_distance columns)
        """
        from sklearn.metrics.pairwise import cosine_distances
        
        rows = []
        self_matches_excluded = 0
        
        logger.info(f"Building SLIM distance matrix for {len(query_centroids):,} query treatments...")
        logger.info(f"  NEW FORMAT: Each landmark = single column 'treatment_distance'")
        
        # Pre-compute landmark matrix (ONCE for all queries)
        # CRITICAL: Sort landmark names alphabetically for consistent ordering
        landmark_names = sorted(list(landmark_embeddings.keys()))
        landmark_matrix = np.array([landmark_embeddings[name] for name in landmark_names])
        logger.info(f"  Landmark matrix shape: {landmark_matrix.shape}")
        logger.info(f"  Landmarks sorted alphabetically for consistent ordering")
        
        # Extract all query vectors at once
        query_matrix = query_centroids[self.feature_cols].values
        logger.info(f"  Query matrix shape: {query_matrix.shape}")
        
        # Compute ALL distances at once (VECTORIZED!)
        logger.info("  Computing all pairwise distances...")
        all_distances = cosine_distances(query_matrix, landmark_matrix)
        logger.info(f"  Distance matrix shape: {all_distances.shape}")
        
        # Now build the output dataframe
        logger.info("  Building slim output dataframe (single distance column per landmark)...")
        for idx, query_row in tqdm(query_centroids.iterrows(), total=len(query_centroids), 
                                desc="Processing results"):
            query_idx = list(query_centroids.index).index(idx)
            query_treatment = query_row['Metadata_treatment']
            
            # Start with query treatment metadata
            row = {'treatment': query_treatment}
            
            # Add ALL query metadata columns
            query_metadata_cols = [col for col in query_row.index 
                                if (col.startswith('Metadata_') or col == 'library') 
                                and col != 'Metadata_treatment']
            for col in query_metadata_cols:
                row[col] = query_row[col]
            
            # Add is_reference flag
            row['is_reference'] = is_reference
            
            # Add n_replicates if available
            if 'n_replicates' in query_row:
                row['n_replicates'] = query_row['n_replicates']
            
            # Add query's own metrics if available
            if is_reference and self.reference_mad is not None:
                query_metrics = self.reference_mad[self.reference_mad['treatment'] == query_treatment]
                if len(query_metrics) > 0:
                    row['query_mad'] = query_metrics.iloc[0].get('mad_cosine')
                    row['query_dmso_distance'] = query_metrics.iloc[0].get('cosine_distance_from_dmso')
                    row['is_landmark'] = query_metrics.iloc[0].get('is_landmark', False)
            elif not is_reference and self.test_mad is not None:
                query_metrics = self.test_mad[self.test_mad['treatment'] == query_treatment]
                if len(query_metrics) > 0:
                    row['query_mad'] = query_metrics.iloc[0].get('mad_cosine')
                    row['query_dmso_distance'] = query_metrics.iloc[0].get('cosine_distance_from_dmso')
            
            # Get distances for this query (already computed!)
            query_distances = all_distances[query_idx]
            
            # Create list of (landmark_name, distance) tuples (already sorted by landmark_names)
            landmark_distances = list(zip(landmark_names, query_distances))
            
            # Handle self-matching (exclude from landmark_distances if needed)
            if exclude_self:
                original_len = len(landmark_distances)
                landmark_distances = [(name, dist) for name, dist in landmark_distances 
                                    if name != query_treatment]
                if len(landmark_distances) < original_len:
                    self_matches_excluded += 1
            
            # NEW FORMAT: Add each landmark as {treatment}_distance column
            # This eliminates redundant lm_N_treatment columns
            for lm_treatment, dist in landmark_distances:
                row[f'{lm_treatment}_distance'] = dist
            
            rows.append(row)
        
        if self_matches_excluded > 0:
            logger.info(f"  Excluded {self_matches_excluded} self-matches")
        
        # Create DataFrame
        matrix_df = pd.DataFrame(rows)
        
        # Count distance columns
        distance_cols = [col for col in matrix_df.columns if col.endswith('_distance') and col != 'query_dmso_distance']
        
        logger.info(f"  ✓ Slim matrix created: {matrix_df.shape}")
        logger.info(f"  Distance columns: {len(distance_cols)} (one per landmark)")
        logger.info(f"  Column format: 'treatment_distance' (e.g., 'TreatmentXXXX@1.0_distance')")
        
        return matrix_df
    
    def _print_summary_statistics(self):
        """Print summary statistics"""
        logger.info("\n" + "="*80)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*80)
        
        logger.info("\nReference Statistics:")
        logger.info(f"  Total treatments: {len(self.reference_centroids)}")
        logger.info(f"  With MAD calculated: {len(self.reference_mad)}")
        logger.info(f"  Landmarks identified: {len(self.landmarks)}")
        if 'is_landmark' in self.reference_mad.columns:
            n_landmarks = self.reference_mad['is_landmark'].sum()
            logger.info(f"   is_landmark column: {n_landmarks} True, {len(self.reference_mad) - n_landmarks} False")
        
        if len(self.reference_landmark_results) > 0 and 'closest_landmark_distance' in self.reference_landmark_results.columns:
            logger.info(f"  Distance to closest landmark:")
            logger.info(f"    Mean: {self.reference_landmark_results['closest_landmark_distance'].mean():.4f}")
            logger.info(f"    Median: {self.reference_landmark_results['closest_landmark_distance'].median():.4f}")
            logger.info(f"    Min: {self.reference_landmark_results['closest_landmark_distance'].min():.4f}")
            logger.info(f"    Max: {self.reference_landmark_results['closest_landmark_distance'].max():.4f}")
        
        logger.info("\nTest Statistics:")
        logger.info(f"  Total treatments: {len(self.test_centroids)}")
        logger.info(f"  With MAD calculated: {len(self.test_mad)}")
        
        if len(self.test_landmark_results) > 0:
            if 'valid_for_phenotypic_makeup' in self.test_landmark_results.columns:
                n_valid = self.test_landmark_results['valid_for_phenotypic_makeup'].sum()
                n_total = len(self.test_landmark_results)
                logger.info(f"   valid_for_phenotypic_makeup: {n_valid} True, {n_total - n_valid} False")
                logger.info(f"     ({100*n_valid/n_total:.1f}% valid for phenotypic makeup analysis)")
            
            if 'closest_landmark_distance' in self.test_landmark_results.columns:
                logger.info(f"  Distance to closest landmark:")
                logger.info(f"    Mean: {self.test_landmark_results['closest_landmark_distance'].mean():.4f}")
                logger.info(f"    Median: {self.test_landmark_results['closest_landmark_distance'].median():.4f}")
                logger.info(f"    Min: {self.test_landmark_results['closest_landmark_distance'].min():.4f}")
                logger.info(f"    Max: {self.test_landmark_results['closest_landmark_distance'].max():.4f}")
        
        logger.info("\nNew Output Files:")
        logger.info("   cosine_distance_matrix_for_clustering.parquet")
        logger.info("   cosine_similarity_matrix_for_clustering.parquet")
        logger.info("   treatment_metadata_for_clustering.csv")
        logger.info("\nThese files are ready for use with hierarchical_chunk_clustering.py")