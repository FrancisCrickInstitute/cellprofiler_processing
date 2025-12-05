"""
Enhanced Hierarchical Clustering using Pre-Computed Distance Matrix
Integrated version for cellprofiler_analyser package
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from pathlib import Path
import re
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class HierarchicalClusteringAnalyzer:
    """Handles hierarchical clustering analysis using pre-computed distance matrices"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize hierarchical clustering analyzer
        
        Args:
            output_dir: Base output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_full_analysis(self, 
                         distance_matrix_path: Path,
                         similarity_matrix_path: Path, 
                         treatment_metadata_path: Path,
                         reference_mad_path: Path,
                         test_landmark_path: Path,
                         config: dict,  # Add config parameter
                         chunk_size: int = 200) -> bool:
        """
        Run complete hierarchical clustering analysis
        
        Args:
            distance_matrix_path: Path to cosine_distance_matrix_for_clustering.parquet
            similarity_matrix_path: Path to cosine_similarity_matrix_for_clustering.parquet
            treatment_metadata_path: Path to treatment_metadata_for_clustering.csv
            reference_mad_path: Path to reference_mad_and_dmso.csv
            test_landmark_path: Path to test_to_landmark_distances.csv
            chunk_size: Size of chunks for heatmaps
            
        Returns:
            bool: Success status
        """
        try:
            logger.info("="*80)
            logger.info("HIERARCHICAL CLUSTERING ANALYSIS")
            logger.info("="*80)
            
            # 1. Load pre-computed data
            distance_matrix, treatment_names, metadata_df, ref_df, test_df = self._load_precomputed_data(
                distance_matrix_path, treatment_metadata_path, reference_mad_path, test_landmark_path
            )
                
            # 2. Create enhanced labels with config
            enhanced_labels, libraries = self._create_enhanced_labels(metadata_df, config)
            
            # 3. Extract metadata for color bars
            metadata_colorbars = self._extract_metadata_for_colorbars(metadata_df)
            
            # 4. Load similarity matrix
            similarity_matrix = self._load_similarity_matrix(similarity_matrix_path)
            
            # 5. Perform global clustering
            linkage_matrix, global_order, ordered_names = self._perform_global_clustering(
                distance_matrix, treatment_names
            )
            
            # 6. Reorder everything based on global clustering
            ordered_data = self._reorder_data_by_clustering(
                similarity_matrix, enhanced_labels, libraries, metadata_colorbars, 
                treatment_names, global_order
            )
            
            # 7. Create splits using ordered data and config
            splits_ordered = self._create_splits_ordered(
                ordered_data['treatments'], ordered_data['libraries'], ref_df, test_df, config
            )
            
            # 8. Create chunked heatmaps
            self._create_chunked_heatmaps(
                ordered_data['similarity'], ordered_data['treatments'],
                ordered_data['enhanced_labels'], ordered_data['libraries'],
                splits_ordered, ordered_data['metadata_colorbars'], chunk_size, 
                config
            )
            
            # 9. Save results
            self._save_results(
                ordered_data['similarity'], ordered_data['enhanced_labels'],
                ordered_data['treatments'], ordered_data['libraries'],
                linkage_matrix, global_order
            )
            
            logger.info("✓ Hierarchical clustering analysis completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Hierarchical clustering analysis failed: {e}", exc_info=True)
            return False
    
    def _load_precomputed_data(self, distance_matrix_path, metadata_path, ref_mad_path, test_landmark_path):
        """Load pre-computed distance matrix and metadata"""
        logger.info("Loading pre-computed data...")
        
        # Load pre-computed distance matrix
        logger.info(f"Loading distance matrix from: {distance_matrix_path}")
        distance_matrix_df = pd.read_parquet(distance_matrix_path)
        distance_matrix = distance_matrix_df.values
        treatment_names = distance_matrix_df.index.tolist()
        
        # Load treatment metadata
        logger.info(f"Loading treatment metadata from: {metadata_path}")
        metadata_df = pd.read_csv(metadata_path)
        
        # Load reference data with is_landmark column (optional - may not exist in test-only mode)
        if ref_mad_path is not None and Path(ref_mad_path).exists():
            logger.info(f"Loading reference data from: {ref_mad_path}")
            ref_df = pd.read_csv(ref_mad_path)
        else:
            logger.info("No reference MAD data available (test-only mode)")
            ref_df = pd.DataFrame()
        
        # Load test data with valid_for_phenotypic_makeup column (optional - may not exist in reference-only mode)
        if test_landmark_path is not None and Path(test_landmark_path).exists():
            logger.info(f"Loading test data from: {test_landmark_path}")
            test_df = pd.read_csv(test_landmark_path)
        else:
            logger.info("No test landmark data available (reference-only mode)")
            test_df = pd.DataFrame()
        
        logger.info(f"✓ Loaded data: {len(treatment_names)} treatments")
        return distance_matrix, treatment_names, metadata_df, ref_df, test_df
    
    def _load_similarity_matrix(self, similarity_matrix_path):
        """Load similarity matrix"""
        logger.info(f"Loading similarity matrix from: {similarity_matrix_path}")
        similarity_matrix_df = pd.read_parquet(similarity_matrix_path)
        similarity_matrix = similarity_matrix_df.values
        logger.info(f"✓ Loaded similarity matrix: {similarity_matrix.shape}")
        return similarity_matrix
    
    def _create_enhanced_labels(self, metadata_df, config):
        """Create enhanced labels using metadata and config-defined libraries"""
        enhanced_labels = []
        libraries = []
        
        # Get library definitions from config (handle None values)
        library_defs = config.get('library_definitions') or {}
        TEST_LIBRARIES = library_defs.get('test_libraries') or []
        REFERENCE_LIBRARIES = library_defs.get('reference_libraries') or []
        
        for _, row in metadata_df.iterrows():
            treatment = row.get('Metadata_treatment', '')
            library = row.get('Metadata_library')
            concentration = row.get('Metadata_compound_uM', '')
            
            libraries.append(library)
            
            if pd.isna(concentration) or concentration == '':
                conc_match = re.search(r'@([0-9]+\.?[0-9]*)$', str(treatment))
                concentration = conc_match.group(1) if conc_match else "0.0"
            
            if pd.notna(library) and library in TEST_LIBRARIES:
                pp_id = row.get('Metadata_PP_ID', '')
                if pd.notna(pp_id) and pp_id != '':
                    label = f"{pp_id}@{concentration}uM"
                else:
                    label = f"{treatment}"
            
            elif pd.notna(library) and library in REFERENCE_LIBRARIES:
                pp_id = row.get('Metadata_PP_ID', '')
                target_desc = row.get('Metadata_annotated_target_description_truncated_10', 
                                    row.get('Metadata_annotated_target_description', ''))
                moa = row.get('Metadata_annotated_target_first', '')
                
                if pd.notna(target_desc) and target_desc != "":
                    target_with_conc = str(target_desc) + f"@{concentration}uM"
                else:
                    target_with_conc = treatment
                
                parts = []
                if pd.notna(moa) and str(moa).strip() != "":
                    parts.append(str(moa))
                parts.append(target_with_conc)
                if pd.notna(pp_id) and str(pp_id).strip() != "":
                    parts.append(f"{pp_id}@{concentration}uM")
                
                label = " | ".join(parts) if parts else treatment
            
            else:
                label = f"{treatment}"
            
            enhanced_labels.append(label)
        
        logger.info(f"✓ Created enhanced labels for {len(enhanced_labels)} treatments")
        return enhanced_labels, libraries
    
    def _extract_metadata_for_colorbars(self, metadata_df):
        """Extract metadata for color bars"""
        well_rows = []
        well_columns = []
        plate_barcodes = []
        manual_annotations = []
        
        for _, row in metadata_df.iterrows():
            well = row.get('Metadata_well', '')
            if pd.notna(well) and len(str(well)) >= 2:
                well_str = str(well).strip()
                well_row = well_str[0] if well_str[0].isalpha() else 'Unknown'
                well_col = well_str[1:].zfill(2) if len(well_str) > 1 else 'Unknown'
            else:
                well_row = 'Unknown'
                well_col = 'Unknown'
            
            well_rows.append(well_row)
            well_columns.append(well_col)
            plate_barcodes.append(str(row.get('Metadata_plate_barcode', 'Unknown')))
            manual_annotations.append(str(row.get('Metadata_manual_annotation', 'Unknown')))
        
        return {
            'well_row': well_rows,
            'well_column': well_columns,
            'plate_barcode': plate_barcodes,
            'manual_annotation': manual_annotations
        }
    
    def _perform_global_clustering(self, distance_matrix, treatment_names, method='average'):
        """Perform global hierarchical clustering"""
        logger.info("Performing global hierarchical clustering...")
        
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method=method)
        global_order = leaves_list(linkage_matrix)
        ordered_names = [treatment_names[i] for i in global_order]
        
        logger.info("✓ Global clustering complete!")
        return linkage_matrix, global_order, ordered_names
    
    def _reorder_data_by_clustering(self, similarity_matrix, enhanced_labels, libraries, 
                                   metadata_colorbars, treatment_names, global_order):
        """Reorder all data based on global clustering order"""
        logger.info("Reordering data by clustering...")
        
        ordered_similarity = similarity_matrix[np.ix_(global_order, global_order)]
        ordered_enhanced_labels = [enhanced_labels[i] for i in global_order]
        ordered_libraries = [libraries[i] for i in global_order]
        ordered_treatments = [treatment_names[i] for i in global_order]
        
        ordered_metadata = {
            'well_row': [metadata_colorbars['well_row'][i] for i in global_order],
            'well_column': [metadata_colorbars['well_column'][i] for i in global_order],
            'plate_barcode': [metadata_colorbars['plate_barcode'][i] for i in global_order],
            'manual_annotation': [metadata_colorbars['manual_annotation'][i] for i in global_order]
        }
        
        logger.info("✓ Data reordered by hierarchical clustering")
        return {
            'similarity': ordered_similarity,
            'enhanced_labels': ordered_enhanced_labels,
            'libraries': ordered_libraries,
            'treatments': ordered_treatments,
            'metadata_colorbars': ordered_metadata
        }
    
    def _create_splits_ordered(self, ordered_treatments, ordered_libraries, ref_df, test_df, config):
        """Create splits using ordered data and config-defined libraries"""
        # Get library definitions from config (handle None values)
        library_defs = config.get('library_definitions') or {}
        TEST_LIBRARIES = library_defs.get('test_libraries') or []
        REFERENCE_LIBRARIES = library_defs.get('reference_libraries') or []
        
        # Get landmark treatments (may be empty in test-only mode)
        landmark_treatments = []
        if len(ref_df) > 0 and 'is_landmark' in ref_df.columns:
            landmark_treatments = ref_df[ref_df['is_landmark'] == True]['treatment'].tolist()
        
        # Get valid test treatments (may be empty in reference-only mode)
        valid_test_treatments = []
        if len(test_df) > 0 and 'valid_for_phenotypic_makeup' in test_df.columns:
            valid_test_treatments = test_df[test_df['valid_for_phenotypic_makeup'] == True]['treatment'].tolist()
        
        relevant_landmarks = set()
        if len(test_df) > 0 and 'treatment' in test_df.columns:
            for treatment in valid_test_treatments:
                if treatment in test_df['treatment'].values:
                    row = test_df[test_df['treatment'] == treatment].iloc[0]
                    for landmark_col in ['closest_landmark_treatment', 'second_closest_landmark_treatment', 'third_closest_landmark_treatment']:
                        if landmark_col in row and pd.notna(row.get(landmark_col)):
                            landmark_treatment = row.get(landmark_col)
                            if landmark_treatment in ordered_treatments:
                                relevant_landmarks.add(landmark_treatment)
        
        splits_ordered = {
            'test_and_reference': {
                'filter_fn': lambda idx: True,
                'description': 'All treatments (test + reference)'
            },
            'test_only': {
                'filter_fn': lambda idx: (pd.notna(ordered_libraries[idx]) and 
                                         ordered_libraries[idx] in TEST_LIBRARIES),
                'description': 'Test libraries only'
            },
            'reference_only': {
                'filter_fn': lambda idx: (pd.notna(ordered_libraries[idx]) and 
                                         ordered_libraries[idx] in REFERENCE_LIBRARIES),
                'description': 'Reference libraries only'
            },
            'reference_landmarks': {
                'filter_fn': lambda idx: (pd.notna(ordered_libraries[idx]) and 
                                         ordered_libraries[idx] in REFERENCE_LIBRARIES and
                                         ordered_treatments[idx] in landmark_treatments),
                'description': 'Reference landmarks (is_landmark==True)'
            },
            'test_and_all_reference_landmarks': {
                'filter_fn': lambda idx: (
                    (pd.notna(ordered_libraries[idx]) and 
                     ordered_libraries[idx] in TEST_LIBRARIES) or
                    (pd.notna(ordered_libraries[idx]) and 
                     ordered_libraries[idx] in REFERENCE_LIBRARIES and
                     ordered_treatments[idx] in landmark_treatments)
                ),
                'description': 'All test + All reference landmarks (is_landmark==True)'
            },
            'test_valid_and_relevant_landmarks': {
                'filter_fn': lambda idx: (
                    (pd.notna(ordered_libraries[idx]) and 
                     ordered_libraries[idx] in TEST_LIBRARIES and
                     ordered_treatments[idx] in valid_test_treatments) or
                    (ordered_treatments[idx] in relevant_landmarks)
                ),
                'description': 'Valid test + Relevant reference landmarks'
            }
        }
        
        for split_name, split_config in splits_ordered.items():
            split_indices = [i for i in range(len(ordered_treatments)) if split_config['filter_fn'](i)]
            logger.info(f"  {split_name}: {len(split_indices)} treatments")
        
        return splits_ordered
    
    def _create_colormap_for_metadata(self, values, colormap_type='categorical'):
        """Create colormap for metadata"""
        import matplotlib.colors as mcolors
        import hashlib
        
        unique_vals = sorted(list(set([v for v in values if v != 'Unknown'])))
        if 'Unknown' in values:
            unique_vals.append('Unknown')
        
        n_unique = len(unique_vals)
        value_to_color = {}
        
        if colormap_type == 'categorical':
            if n_unique <= 20:
                cmap = plt.colormaps['tab20'].resampled(20)
                colors = [cmap(i) for i in range(n_unique)]
            else:
                cmap1 = plt.colormaps['tab20'].resampled(20)
                cmap2 = plt.colormaps['tab20b'].resampled(20)
                cmap3 = plt.colormaps['tab20c'].resampled(20)
                colors = []
                for i in range(n_unique):
                    if i < 20:
                        colors.append(cmap1(i))
                    elif i < 40:
                        colors.append(cmap2(i - 20))
                    elif i < 60:
                        colors.append(cmap3(i - 40))
                    else:
                        colors.append(cmap1(i % 20))
            
            for i, val in enumerate(unique_vals):
                value_to_color[val] = colors[i]
        
        elif colormap_type == 'sequential':
            cmap = plt.colormaps['viridis'].resampled(n_unique)
            for i, val in enumerate(unique_vals):
                value_to_color[val] = cmap(i / max(1, n_unique - 1))
        
        elif colormap_type == 'hashed':
            cmap = plt.colormaps['tab20'].resampled(20)
            cmap2 = plt.colormaps['tab20b'].resampled(20)
            
            for val in unique_vals:
                hash_val = int(hashlib.md5(str(val).encode()).hexdigest(), 16)
                color_idx = hash_val % 40
                
                if color_idx < 20:
                    value_to_color[val] = cmap(color_idx)
                else:
                    value_to_color[val] = cmap2(color_idx - 20)
        
        if 'Unknown' in value_to_color:
            value_to_color['Unknown'] = (0.7, 0.7, 0.7, 1.0)
        
        return value_to_color
    
    def _save_split_as_single_pdf(self, split_dir, split_name, figures):
        """Save all figures for a split into a single PDF"""
        output_path = split_dir / f"{split_name}_all_chunks.pdf"
        
        logger.info(f"Creating PDF: {output_path}")
        logger.info(f"Total chunks: {len(figures)}")
        
        with PdfPages(output_path) as pdf:
            for idx, fig in enumerate(figures, 1):
                logger.info(f"Adding chunk {idx}/{len(figures)} to PDF...")
                pdf.savefig(fig, dpi=300)
                plt.close(fig)
        
        logger.info(f"✓ Saved: {output_path}")
    
    def _create_chunked_heatmaps(self, ordered_similarity, ordered_treatments, 
                                ordered_enhanced_labels, ordered_libraries,
                                splits_ordered, ordered_metadata, chunk_size, config):
        """Create chunked heatmaps with global ordering and FIXED color bar alignment"""
        logger.info("Creating chunked heatmaps...")
        
        # Get library definitions from config instead of hardcoding
        library_defs = config.get('library_definitions') or {}
        TEST_LIBRARIES = library_defs.get('test_libraries') or []
        REFERENCE_LIBRARIES = library_defs.get('reference_libraries') or []

        
        n_treatments = len(ordered_treatments)
        n_chunks = int(np.ceil(n_treatments / chunk_size))
        
        logger.info(f"Total treatments: {n_treatments}")
        logger.info(f"Chunk size: {chunk_size}")
        logger.info(f"Number of chunks: {n_chunks}")
        
        # Temporary storage for figures before PDF compilation
        temp_figures = {}
        
        for split_name, split_config in splits_ordered.items():
            logger.info(f"Processing split: {split_name}")
            logger.info(f"Description: {split_config['description']}")
            
            # Filter treatments for this split
            split_indices = [i for i in range(n_treatments) if split_config['filter_fn'](i)]
            
            if not split_indices:
                logger.warning(f"No treatments found for split {split_name}. Skipping...")
                continue
                
            logger.info(f"Treatments in split: {len(split_indices)}")
            
            # Create submatrix for this split
            split_matrix = ordered_similarity[np.ix_(split_indices, split_indices)]
            split_labels = [ordered_enhanced_labels[i] for i in split_indices]
            split_libraries = [ordered_libraries[i] for i in split_indices]
            
            # Create output directory for this split
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Create chunks for this split
            n_split_chunks = int(np.ceil(len(split_indices) / chunk_size))
            temp_figures[split_name] = []
            
            # PROCESS ALL CHUNKS
            for chunk_idx in range(n_split_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(split_indices))
                
                logger.info(f"Chunk {chunk_idx + 1}/{n_split_chunks}: treatments {start_idx + 1}-{end_idx}")
                
                # Extract chunk from the SPLIT (not original)
                chunk_matrix = split_matrix[start_idx:end_idx, start_idx:end_idx]
                chunk_labels = split_labels[start_idx:end_idx]
                chunk_libraries = split_libraries[start_idx:end_idx]
                
                # Calculate actual chunk size for this specific chunk
                n_chunk = end_idx - start_idx
                
                # Create figure with precise GridSpec layout
                fig = plt.figure(figsize=(22, 22))

                # Define grid: 4 small rows for colorbars + 1 large row for heatmap
                n_colorbar_rows = 4
                heatmap_row_height = 20
                colorbar_row_height = 0.3

                gs = GridSpec(
                    n_colorbar_rows + 1,  # +1 for the main heatmap
                    1, 
                    figure=fig,
                    height_ratios=[colorbar_row_height] * n_colorbar_rows + [heatmap_row_height],
                    hspace=0.02  # Small space between subplots
                )

                # Create axes - all share the same x-axis
                ax_main = fig.add_subplot(gs[n_colorbar_rows, 0])  # Bottom row for heatmap

                # Create colorbar axes that share the x-axis with main heatmap
                ax_top1 = fig.add_subplot(gs[0, 0], sharex=ax_main)
                ax_top2 = fig.add_subplot(gs[1, 0], sharex=ax_main)  
                ax_top3 = fig.add_subplot(gs[2, 0], sharex=ax_main)
                ax_top4 = fig.add_subplot(gs[3, 0], sharex=ax_main)

                # Turn off spines and ticks for colorbar axes
                for ax in [ax_top1, ax_top2, ax_top3, ax_top4]:
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(axis='both', which='both', length=0)
                    ax.set_ylabel('')
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Plot main heatmap FIRST
                heatmap = sns.heatmap(
                    chunk_matrix,
                    cmap='RdBu_r',
                    center=0,
                    vmin=-1,
                    vmax=1,
                    xticklabels=False,
                    yticklabels=chunk_labels,
                    cbar_kws={
                        'label': 'Cosine Similarity',
                        'shrink': 0.5,
                        'aspect': 15,
                        'pad': 0.02
                    },
                    square=True,
                    ax=ax_main
                )

                # Color y-axis labels by library type
                for idx, (label, lib) in enumerate(zip(ax_main.get_yticklabels(), chunk_libraries)):
                    if pd.notna(lib) and lib in TEST_LIBRARIES:
                        label.set_color('blue')
                    elif pd.notna(lib) and lib in REFERENCE_LIBRARIES:
                        label.set_color('green')

                # Set y-axis label properties
                ax_main.set_yticklabels(chunk_labels, rotation=0, fontsize=3)

                # Get axis limits AFTER seaborn has done all its adjustments
                heatmap_xlim = ax_main.get_xlim()
                heatmap_ylim = ax_main.get_ylim()
                
                logger.debug(f"Heatmap xlim after seaborn: {heatmap_xlim}")
                logger.debug(f"Heatmap ylim after seaborn: {heatmap_ylim}")
                logger.debug(f"Number of treatments in chunk: {n_chunk}")

                # NOW create color bars using color mapping approach
                if ordered_metadata is not None:
                    # Get chunk metadata
                    chunk_global_indices = [split_indices[i] for i in range(start_idx, end_idx)]
                    chunk_well_row = [ordered_metadata['well_row'][i] for i in chunk_global_indices]
                    chunk_well_col = [ordered_metadata['well_column'][i] for i in chunk_global_indices]
                    chunk_plate = [ordered_metadata['plate_barcode'][i] for i in chunk_global_indices]
                    chunk_annotation = [ordered_metadata['manual_annotation'][i] for i in chunk_global_indices]

                    # Create color maps
                    well_row_cmap = self._create_colormap_for_metadata(chunk_well_row, 'categorical')
                    well_col_cmap = self._create_colormap_for_metadata(chunk_well_col, 'sequential')
                    plate_cmap = self._create_colormap_for_metadata(chunk_plate, 'hashed')
                    annotation_cmap = self._create_colormap_for_metadata(chunk_annotation, 'categorical')

                    # Convert to color arrays
                    well_row_colors = [well_row_cmap[val] for val in chunk_well_row]
                    well_col_colors = [well_col_cmap[val] for val in chunk_well_col]
                    plate_colors = [plate_cmap[val] for val in chunk_plate]
                    
                    # For annotation: make TEST compounds white
                    annotation_colors = []
                    for i, (annotation, lib) in enumerate(zip(chunk_annotation, chunk_libraries)):
                        if pd.notna(lib) and lib in TEST_LIBRARIES:
                            annotation_colors.append((1.0, 1.0, 1.0, 1.0))  # White for test
                        else:
                            annotation_colors.append(annotation_cmap[annotation])

                    # Use the exact same format as SPC
                    colorbar_configs = [
                        (well_row_colors, 'Row', ax_top1),
                        (well_col_colors, 'Column', ax_top2),
                        (plate_colors, 'Plate', ax_top3), 
                        (annotation_colors, 'Annotation', ax_top4)
                    ]

                    for colors, label, ax in colorbar_configs:
                        # Convert color list to array for imshow
                        color_array = np.array(colors).reshape(1, -1, 4)
                        
                        # Use imshow with extent that matches EXACTLY the heatmap x-range
                        im = ax.imshow(
                            color_array,
                            aspect='auto',
                            extent=[heatmap_xlim[0], heatmap_xlim[1], 0, 1],
                            interpolation='nearest'
                        )
                        
                        # Force the EXACT same x-limits as main heatmap
                        ax.set_xlim(heatmap_xlim)
                        ax.set_ylim(0, 1)
                        
                        # Remove everything except the label
                        ax.spines['top'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.tick_params(axis='both', which='both', length=0)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        # Set label
                        ax.set_ylabel(label, fontsize=6, rotation=0, ha='right', va='center')

                # Set IDENTICAL position for all axes in figure coordinates
                heatmap_pos = ax_main.get_position()
                
                # Force all colorbar axes to have EXACTLY the same x-position and width
                for ax in [ax_top1, ax_top2, ax_top3, ax_top4]:
                    ax_pos = ax.get_position()
                    ax.set_position([heatmap_pos.x0, ax_pos.y0, heatmap_pos.width, ax_pos.height])

                # Set title
                fig.suptitle(
                    f"Cosine Similarity - {split_name.replace('_', ' ').title()}\n"
                    f"Chunk {chunk_idx + 1}/{n_split_chunks} - Treatments {start_idx + 1}-{end_idx}\n"
                    f"(Red = Similar, White = Orthogonal, Blue = Opposite)",
                    fontsize=12,
                    y=0.98
                )

                # Store the figure temporarily for later PDF compilation
                temp_figures[split_name].append(fig)
                
                logger.info(f"Created chunk {chunk_idx + 1}/{n_split_chunks} (in memory)")
            
            # After all chunks for this split are created, save as one PDF
            logger.info(f"Saving all chunks as single PDF...")
            self._save_split_as_single_pdf(split_dir, split_name, temp_figures[split_name])
            
            # Clean up memory
            temp_figures[split_name] = []
        
        logger.info("✓ All PDFs created!")
    
    def _save_results(self, ordered_similarity, ordered_enhanced_labels, 
                     ordered_treatments, ordered_libraries, linkage_matrix, global_order):
        """Save clustering results"""
        logger.info("Saving results...")
        
        # AFTER (always works):
        dist_df = pd.DataFrame(
            ordered_similarity,
            index=ordered_treatments,
            columns=ordered_treatments
        )
        dist_path = self.output_dir / "ordered_cosine_similarity_matrix.parquet"
        dist_df.to_parquet(dist_path)
        logger.info(f"✓ Saved ordered similarity matrix: {dist_path}")
        
        # Save clustering info
        cluster_info = pd.DataFrame({
            'treatment': ordered_treatments,
            'enhanced_label': ordered_enhanced_labels,
            'library': ordered_libraries,
            'global_order': range(len(ordered_treatments))
        })
        info_path = self.output_dir / "clustering_information.csv"
        cluster_info.to_csv(info_path, index=False)
        logger.info(f"✓ Saved clustering information: {info_path}")
        
        # Save linkage matrix
        linkage_df = pd.DataFrame(linkage_matrix)
        linkage_path = self.output_dir / "linkage_matrix.csv"
        linkage_df.to_csv(linkage_path, index=False)
        logger.info(f"✓ Saved linkage matrix: {linkage_path}")