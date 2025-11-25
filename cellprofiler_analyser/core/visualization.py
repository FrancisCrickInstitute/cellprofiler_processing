"""
Visualization utilities for Cell Painting data - Optimized to plot from coordinate files
UPDATED: Now includes hierarchical clustering integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import glob

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:
    umap = None

# Plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    go = None
    px = None

from ..utils.logging_utils import get_logger

# Replace with this simple approach:
def aggregate_coordinates_by_treatment(*args, **kwargs):
    """DEPRECATED - Using pre-computed matrices instead"""
    logger.error("❌ ERROR: clustering_utils.aggregate_coordinates_by_treatment() was called!")
    logger.error("This function is DEPRECATED and should not be used.")
    raise DeprecationWarning("aggregate_coordinates_by_treatment is deprecated - use pre-computed matrices")

def create_hierarchical_clustering_plots(*args, **kwargs):
    """DEPRECATED - Using HierarchicalClusteringAnalyzer instead"""
    logger.error("Hierarchical clustering plots function is deprecated")
    return False

def create_clustering_from_existing_coordinates(*args, **kwargs):
    """DEPRECATED - Using HierarchicalClusteringAnalyzer instead"""
    logger.error("Clustering from coordinates function is deprecated")
    return False

try:
    import plotly.colors as pc
except ImportError:
    pc = None

logger = get_logger(__name__)


class DataVisualizer:
    """Handles visualization and dimensionality reduction operations"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer with organized directory structure
        
        Args:
            output_dir: Output directory for saving plots
        """
        self.output_dir = Path(output_dir)
        
        # Create organized directory structure
        self.analysis_dir = self.output_dir / "analysis"
        self.pca_dir = self.analysis_dir / "pca"
        self.correlation_dir = self.analysis_dir / "correlation"
        self.histograms_dir = self.analysis_dir / "histograms"
        
        self.viz_dir = self.output_dir / "visualizations"

        self.umap_dir = self.viz_dir / "umap"
        self.umap_interactive_dir = self.umap_dir / "interactive"

        self.tsne_dir = self.viz_dir / "tsne"
        self.tsne_interactive_dir = self.tsne_dir / "interactive"

        self.coords_dir = self.viz_dir / "coordinates"  # Single coordinates directory

        # NEW: Add visualizations_redo directory
        self.viz_redo_dir = self.output_dir / "visualizations_redo"
        self.umap_redo_dir = self.viz_redo_dir / "umap"
        self.tsne_redo_dir = self.viz_redo_dir / "tsne"
        
        # Create all directories
        for directory in [
            self.analysis_dir, self.pca_dir, self.correlation_dir, self.histograms_dir,
            self.viz_dir, self.umap_dir, self.umap_interactive_dir,
            self.tsne_dir, self.tsne_interactive_dir, self.coords_dir,
            self.viz_redo_dir, self.umap_redo_dir, self.tsne_redo_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Models and results
        self.pca_model = None
        self.pca_feature_names = None
        self.umap_results = {}
        self.tsne_results = {}
        self.correlation_matrix = None
        
        # Essential metadata columns for visualization plots - UPDATED to include Metadata_treatment
        self.essential_viz_metadata_cols = [
            'Metadata_replicate', 'Metadata_compound_uM', 'Metadata_library',
            'Metadata_perturbation_name', 'Metadata_chemical_name',
            'Metadata_annotated_target_truncated', 'Metadata_annotated_target_description_truncated', 
            'Metadata_PP_ID', 'Metadata_plate_barcode', 'Metadata_well', 'Metadata_treatment'
        ]
    
    def get_visualization_metadata_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Get list of metadata columns suitable for categorical visualization
        Uses the essential list plus any additional valid metadata columns
        
        Args:
            data: DataFrame to check
            
        Returns:
            List of visualization-suitable metadata columns
        """
        # Start with essential columns that exist in data
        viz_cols = [col for col in self.essential_viz_metadata_cols if col in data.columns]
        
        # Add any other metadata columns that aren't feature measurements
        potential_metadata_cols = [col for col in data.columns if col.startswith('Metadata_')]
        
        # Filter out unwanted columns (feature measurements, technical metadata)
        unwanted_keywords = [
            'intensity', 'area', 'texture', 'radial', 'granularity', 'correlation',
            'colocalization', 'neighbors', 'location', 'shape', 'radialfeatures',
            'moments', 'zernike', 'radialdistribution', 'mean', 'median', 'std',
            'mad', 'min', 'max', 'integrated', 'massDisplacement',
            'ExecutionTime', 'FileName', 'PathName', 'Height', 'Width', 
            'MD5Digest', 'Scaling', 'URL', 'Group_Index', 'Group_Length', 
            'Group_Number', 'plane', 'sequence', 'site', 'timepoint'
        ]
        
        for col in potential_metadata_cols:
            if col not in viz_cols and not any(keyword.lower() in col.lower() for keyword in unwanted_keywords):
                viz_cols.append(col)
        
        logger.info(f"Found {len(viz_cols)} visualization metadata columns")
        return viz_cols

    def create_hover_text_from_coords(self, coord_data: pd.DataFrame) -> List[str]:
        """
        Create compact hover text from coordinate data using essential metadata
        
        Args:
            coord_data: DataFrame with coordinates and metadata
            
        Returns:
            List of hover text strings
        """
        # Get available essential hover columns
        available_hover_cols = [col for col in self.essential_viz_metadata_cols if col in coord_data.columns]
        
        hover_text = []
        for i in range(len(coord_data)):
            text_parts = []
            for col in available_hover_cols:
                value = str(coord_data[col].iloc[i])
                # Truncate very long values to keep hover compact
                if len(value) > 30:
                    value = value[:27] + '...'
                clean_col_name = col.replace('Metadata_', '')
                text_parts.append(f"{clean_col_name}: {value}")
            hover_text.append("<br>".join(text_parts))
        
        return hover_text

    def create_categorical_plot_from_coordinates(self, coord_data: pd.DataFrame, metadata_col: str, 
                                           embedding_type: str, param_name: str) -> Optional[Any]:
        """
        Create memory-efficient plot using SINGLE TRACE - no data duplication
        """
        if go is None or px is None:
            logger.error("Plotly not available for interactive plots")
            return None
        
        try:
            # FIXED: Use lowercase embedding type to match saved coordinate column names
            embedding_prefix = embedding_type.lower()  # "UMAP" -> "umap", "tSNE" -> "tsne"
            
            # Get embedding coordinates - look for the specific parameter combination
            x_col = f'{embedding_prefix}_{param_name}_x'
            y_col = f'{embedding_prefix}_{param_name}_y'
            
            if x_col not in coord_data.columns or y_col not in coord_data.columns:
                logger.error(f"Missing {embedding_type} coordinates for {param_name}: {x_col}, {y_col} not in data")
                logger.debug(f"Available columns starting with {embedding_prefix}_: {[col for col in coord_data.columns if col.startswith(f'{embedding_prefix}_')]}")
                return None
            
            # Get categorical data for coloring
            if metadata_col not in coord_data.columns:
                logger.error(f"Column {metadata_col} not found in coordinate data")
                return None
            
            categorical_values = coord_data[metadata_col].fillna('N/A').astype(str)
            unique_categories = categorical_values.nunique()
            
            logger.info(f"Creating {embedding_type} plot for {metadata_col}: {unique_categories} unique values")
            
            # Create full hover text with all essential metadata - NO TRUNCATION
            available_hover_cols = [col for col in self.essential_viz_metadata_cols if col in coord_data.columns]
            hover_text = []
            for i in range(len(coord_data)):
                text_parts = []
                for col in available_hover_cols:
                    value = str(coord_data[col].iloc[i])  # NO truncation
                    clean_col_name = col.replace('Metadata_', '')
                    text_parts.append(f"{clean_col_name}: {value}")
                hover_text.append("<br>".join(text_parts))
            
            # ALWAYS use single trace approach to avoid data duplication
            logger.info(f"Using single trace for {unique_categories} categories")
            
            # Create color palette that cycles
            base_colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
            ]
            
            # Map categories to colors (cycling through available colors)
            unique_cats = list(categorical_values.unique())
            color_map = {}
            for i, cat in enumerate(unique_cats):
                color_map[cat] = base_colors[i % len(base_colors)]
            
            # Assign colors to each point
            point_colors = [color_map[cat] for cat in categorical_values]
            
            # Create single trace figure - NO DATA DUPLICATION
            fig = go.Figure()
            
            fig.add_trace(go.Scattergl(  # Use Scattergl for better performance
                x=coord_data[x_col].values,  # Direct numpy array
                y=coord_data[y_col].values,  # Direct numpy array
                mode='markers',
                marker=dict(
                    size=5,
                    color=point_colors,  # Direct color array - no separate traces
                    opacity=0.6,
                    line=dict(width=0)  # No marker borders to save space
                ),
                customdata=hover_text,
                hovertemplate='<b>%{customdata}</b><br>' +
                            f'{x_col}: %{{x:.1f}}<br>' +
                            f'{y_col}: %{{y:.1f}}<extra></extra>',
                showlegend=False,  # No legend
                name=f"{embedding_type}_{param_name}"
            ))
            
            # Full size layout
            fig.update_layout(
                title=dict(
                    text=f"{embedding_type} {param_name} - {metadata_col.replace('Metadata_', '')}",
                    font=dict(size=16)
                ),
                width=1600,
                height=900,
                margin=dict(l=40, r=40, t=50, b=40),
                showlegend=False,
                xaxis=dict(
                    title=x_col,
                    title_font_size=12,
                    scaleanchor="y", 
                    scaleratio=1
                ),
                yaxis=dict(
                    title=y_col,
                    title_font_size=12
                ),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            logger.info(f"Created efficient single-trace plot")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating efficient plot: {e}")
            return None

    def create_continuous_plot_from_coordinates(self, coord_data: pd.DataFrame, 
                                              embedding_type: str, param_name: str) -> Optional[Any]:
        """
        Create continuous feature plot from coordinate data with dropdown
        Uses saved PCA feature selection if available, otherwise uses available feature columns
        
        Args:
            coord_data: DataFrame loaded from coordinate CSV file
            embedding_type: "UMAP" or "tSNE" 
            param_name: Parameter name (e.g., "n15_d0.1")
            
        Returns:
            Plotly figure or None if failed
        """
        if go is None or px is None:
            logger.error("Plotly not available for interactive plots")
            return None
        
        try:
            # Get embedding coordinates
            coord_cols = [col for col in coord_data.columns if col.startswith(f'{embedding_type}_')]
            if len(coord_cols) < 2:
                logger.error(f"Missing {embedding_type} coordinates in data")
                return None
            
            x_col, y_col = coord_cols[0], coord_cols[1]
            
            # Get feature columns from coordinate data
            feature_cols = [col for col in coord_data.columns 
                           if not col.startswith('Metadata_') and not col.startswith(f'{embedding_type}_')]
            
            if not feature_cols:
                logger.warning(f"No feature columns found in coordinate data for continuous plot")
                return None
            
            # Select top features (limit to 20 for dropdown efficiency)
            selected_features = feature_cols[:20]
            logger.info(f"Creating continuous plot with {len(selected_features)} features")
            
            # Create COMPACT hover text
            hover_text = self.create_hover_text_from_coords(coord_data)
            
            # Use first feature for initial display
            initial_feature = selected_features[0]
            initial_feature_data = coord_data[initial_feature].values
            
            # Create figure
            fig_continuous = go.Figure()
            
            # Add scatter trace
            scatter = go.Scatter(
                x=coord_data[x_col],
                y=coord_data[y_col],
                mode='markers',
                marker=dict(
                    size=4,
                    opacity=0.7,
                    color=initial_feature_data,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=initial_feature[:30], side='right'),
                        len=0.7,
                        thickness=15
                    )
                ),
                customdata=hover_text,
                hovertemplate='<b>%{customdata}</b><br>' + 
                            f'{x_col}: %{{x:.2f}}<br>' +
                            f'{y_col}: %{{y:.2f}}<br>' + 
                            f'{initial_feature[:20]}: %{{marker.color:.3f}}<extra></extra>',
                name=f'{embedding_type}_{param_name}'
            )
            fig_continuous.add_trace(scatter)
            
            # Create dropdown for features
            dropdown_buttons = []
            for i, col in enumerate(selected_features):
                feature_data = coord_data[col].values
                col_short = col[:30] + '...' if len(col) > 30 else col
                
                dropdown_buttons.append(dict(
                    label=f"Feature {i+1:02d}: {col_short}",
                    method="restyle",
                    args=[{
                        "marker.color": [feature_data],
                        "marker.colorbar.title": col_short,
                        "hovertemplate": '<b>%{customdata}</b><br>' + 
                                    f'{x_col}: %{{x:.2f}}<br>' + 
                                    f'{y_col}: %{{y:.2f}}<br>' + 
                                    f'{col_short}: %{{marker.color:.3f}}<extra></extra>'
                    }]
                ))
            
            # Optimized layout
            fig_continuous.update_layout(
                title=f"{embedding_type} {param_name} - Continuous Features ({len(selected_features)} features)",
                xaxis_title=x_col,
                yaxis_title=y_col,
                updatemenus=[{
                    'buttons': dropdown_buttons,
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.01,
                    'xanchor': 'left',
                    'y': 1.02,
                    'yanchor': 'top',
                    'borderwidth': 1
                }],
                annotations=[{
                    'text': 'Feature:',
                    'showarrow': False,
                    'x': 0.01,
                    'y': 1.08,
                    'xref': 'paper',
                    'yref': 'paper',
                    'font': dict(size=11)
                }],
                width=1000,
                height=700,
                margin=dict(l=50, r=100, t=80, b=50),
                xaxis=dict(scaleanchor="y", scaleratio=1),
                showlegend=False
            )
            
            logger.info(f"Created continuous plot with {len(selected_features)} features")
            return fig_continuous
            
        except Exception as e:
            logger.error(f"Error creating continuous plot from coordinates: {e}")
            return None

    def save_interactive_plots_from_coordinates(self) -> bool:
        """
        SIMPLIFIED: Create and save ONLY categorical plots from coordinate files
        No PCA needed, no continuous plots
        
        Returns:
            bool: Success status
        """
        logger.info("Creating categorical plots from coordinate files (SIMPLIFIED)...")
        
        total_plots_created = 0
        
        # Look for the single coordinate file
        coord_file = self.coords_dir / "embedding_coordinates.csv"

        if not coord_file.exists():
            logger.error(f"Coordinate file not found: {coord_file}")
            return False

        logger.info(f"Loading coordinates from: {coord_file}")

        try:
            # Load coordinate data
            coord_data = pd.read_csv(coord_file)
            logger.info(f"Loaded coordinate data with shape: {coord_data.shape}")
            
            # Find all UMAP and t-SNE coordinate columns (correct pattern matching)
            umap_coord_cols = [col for col in coord_data.columns if col.startswith('umap_') and col.endswith(('_x', '_y'))]
            tsne_coord_cols = [col for col in coord_data.columns if col.startswith('tsne_') and col.endswith(('_x', '_y'))]

            # Extract parameter names - ensure we get the exact parameter names used in coordinates
            umap_params = set()
            for col in umap_coord_cols:
                if col.endswith('_x'):
                    # Extract parameter name between 'umap_' and '_x'
                    param_name = col[5:-2]  # Remove 'umap_' prefix and '_x' suffix
                    umap_params.add(param_name)

            tsne_params = set()  
            for col in tsne_coord_cols:
                if col.endswith('_x'):
                    # Extract parameter name between 'tsne_' and '_x'
                    param_name = col[5:-2]  # Remove 'tsne_' prefix and '_x' suffix
                    tsne_params.add(param_name)
            
            logger.info(f"Found UMAP parameters: {list(umap_params)}")
            logger.info(f"Found t-SNE parameters: {list(tsne_params)}")
            
            # Get essential visualization metadata columns
            viz_metadata_cols = [col for col in self.essential_viz_metadata_cols if col in coord_data.columns]
            logger.info(f"Creating plots for {len(viz_metadata_cols)} metadata columns")
            
            # Create UMAP plots
            for param_name in umap_params:
                logger.info(f"Creating UMAP plots for {param_name}...")
                
                for metadata_col in viz_metadata_cols:
                    try:
                        fig = self.create_categorical_plot_from_coordinates(
                            coord_data, metadata_col, "UMAP", param_name
                        )
                        
                        if fig is not None:
                            safe_col = metadata_col.replace('Metadata_', '').replace(' ', '_').replace('/', '_')
                            plot_path = self.umap_interactive_dir / f"umap_{param_name}_{safe_col}.html"
                            fig.write_html(str(plot_path))
                            logger.info(f"Created UMAP plot: {plot_path.name}")
                            total_plots_created += 1
                            
                    except Exception as e:
                        logger.error(f"Error creating UMAP plot for {metadata_col}: {e}")
            
            # Create t-SNE plots
            for param_name in tsne_params:
                logger.info(f"Creating t-SNE plots for {param_name}...")
                
                for metadata_col in viz_metadata_cols:
                    try:
                        fig = self.create_categorical_plot_from_coordinates(
                            coord_data, metadata_col, "tSNE", param_name
                        )
                        
                        if fig is not None:
                            safe_col = metadata_col.replace('Metadata_', '').replace(' ', '_').replace('/', '_')
                            plot_path = self.tsne_interactive_dir / f"tsne_{param_name}_{safe_col}.html"
                            fig.write_html(str(plot_path))
                            logger.info(f"Created t-SNE plot: {plot_path.name}")
                            total_plots_created += 1
                            
                    except Exception as e:
                        logger.error(f"Error creating t-SNE plot for {metadata_col}: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing coordinate file {coord_file}: {e}")
            return False
        
        logger.info(f"Successfully created {total_plots_created} categorical plots")
        return total_plots_created > 0

    def recreate_plots_from_coordinates(self, use_pca_from_analysis: bool = True) -> bool:
        """
        Recreate categorical plots and hierarchical clustering from coordinate files
        UPDATED: Now includes hierarchical clustering
        
        Args:
            use_pca_from_analysis: Ignored - kept for compatibility
            
        Returns:
            bool: Success status
        """
        logger.info("Recreating categorical plots and hierarchical clustering from coordinate files...")
        
        # Load existing coordinates
        coordinates = self.load_existing_coordinates()
        
        if not coordinates['umap'] and not coordinates['tsne']:
            logger.error("No coordinate files found to recreate plots from")
            return False
        
        total_plots_created = 0
        
        # Process UMAP coordinates
        for param_name, coord_info in coordinates['umap'].items():
            coord_data = coord_info['data']  # This is the full coordinate DataFrame
            
            logger.info(f"Recreating UMAP plots for {param_name}...")
            
            # Get ONLY the essential visualization metadata columns that exist in coord data
            viz_metadata_cols = [col for col in self.essential_viz_metadata_cols if col in coord_data.columns]
            logger.info(f"Creating {len(viz_metadata_cols)} UMAP categorical plots")
            
            # Create ONLY categorical plots for each metadata column
            for metadata_col in viz_metadata_cols:
                try:
                    fig = self.create_categorical_plot_from_coordinates(
                        coord_data, metadata_col, "umap", param_name  # Changed to lowercase
                    )
                    
                    if fig is not None:
                        safe_col = metadata_col.replace('Metadata_', '').replace(' ', '_').replace('/', '_')
                        plot_path = self.umap_redo_dir / f"umap_{param_name}_{safe_col}.html"
                        fig.write_html(str(plot_path))
                        logger.info(f"Created UMAP plot: {plot_path.name}")
                        total_plots_created += 1
                        
                except Exception as e:
                    logger.error(f"Error creating UMAP plot for {metadata_col}: {e}")
        
        # Process t-SNE coordinates  
        for param_name, coord_info in coordinates['tsne'].items():
            coord_data = coord_info['data']
            
            logger.info(f"Recreating t-SNE plots for {param_name}...")
            
            # Get ONLY the essential visualization metadata columns that exist in coord data
            viz_metadata_cols = [col for col in self.essential_viz_metadata_cols if col in coord_data.columns]
            logger.info(f"Creating {len(viz_metadata_cols)} t-SNE categorical plots")
            
            # Create ONLY categorical plots for each metadata column
            for metadata_col in viz_metadata_cols:
                try:
                    fig = self.create_categorical_plot_from_coordinates(
                        coord_data, metadata_col, "tsne", param_name  # Changed to lowercase
                    )
                    
                    if fig is not None:
                        safe_col = metadata_col.replace('Metadata_', '').replace(' ', '_').replace('/', '_')
                        plot_path = self.tsne_redo_dir / f"tsne_{param_name}_{safe_col}.html"
                        fig.write_html(str(plot_path))
                        logger.info(f"Created t-SNE plot: {plot_path.name}")
                        total_plots_created += 1
                        
                except Exception as e:
                    logger.error(f"Error creating t-SNE plot for {metadata_col}: {e}")
        
        # NEW: Skip hierarchical clustering recreation - using pre-computed matrices instead
        logger.info("Skipping hierarchical clustering recreation")
        logger.info("Use HierarchicalClusteringAnalyzer with pre-computed matrices for hierarchical clustering")
        clustering_success = False  # Set to False since we're skipping it
        
        if clustering_success:
            logger.info("Successfully created hierarchical clustering plots")
            total_plots_created += 4  # Assume 4 clustering plots created
        else:
            logger.warning("Failed to create hierarchical clustering plots")
        
        logger.info(f"Successfully created {total_plots_created} plots from coordinates")
        logger.info(f"Plots saved to: {self.viz_redo_dir}")
        
        return total_plots_created > 0
        
    def save_interactive_plots(self, data: pd.DataFrame) -> None:
        """
        Generate coordinates and create categorical plots and hierarchical clustering
        UPDATED: Now includes hierarchical clustering
        
        Args:
            data: Well-aggregated data (used for coordinate generation only)
        """
        logger.info("Generating coordinates and creating categorical plots with hierarchical clustering...")
        
        # First, ensure we have embeddings computed (this saves coordinates)
        if not self.umap_results and not self.tsne_results:
            logger.error("No embeddings available. Run compute_multiple_umaps/tsnes first.")
            return
        
        # Save all coordinates to single file (includes clustering creation)
        self.save_all_coordinates_to_single_file(data)
        
        # Now create ONLY categorical plots from the saved coordinates
        success = self.save_interactive_plots_from_coordinates()
        
        if success:
            logger.info("Categorical plots creation completed using coordinate-based approach!")
        else:
            logger.error("Failed to create categorical plots from coordinates")

    def create_pca_variance_plot(self, data: pd.DataFrame, n_components: int = 100) -> Optional[int]:
        """
        Create PCA variance explained plot - saves to analysis/pca/
        """
        logger.info("Creating PCA variance analysis...")
        
        try:
            if data is None:
                logger.warning("No data available for PCA analysis")
                return None
                    
            # Get feature columns
            feature_cols = [col for col in data.columns if not col.startswith('Metadata_')]
            feature_data = data[feature_cols]
            
            # Remove constant features
            constant_features = feature_data.columns[feature_data.std() == 0]
            if len(constant_features) > 0:
                logger.info(f"Removing {len(constant_features)} constant features for PCA")
                feature_data = feature_data.drop(columns=constant_features)
            
            # Fit PCA
            max_components = min(n_components, feature_data.shape[1], feature_data.shape[0])
            pca = PCA(n_components=max_components)
            pca.fit(feature_data)
            
            # Store PCA model
            self.pca_model = pca
            
            # Get variance explained
            pca_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(pca_var)
            
            # Find components for 90% variance
            components_90 = np.argmax(cumulative_var >= 0.9) + 1
            logger.info(f"Components needed for 90% variance: {components_90}")
            
            # Save PCA components (feature loadings) to pca directory
            pca_components_df = pd.DataFrame(
                pca.components_.T,  # Transpose to get features as rows
                columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                index=feature_data.columns
            )
            pca_components_path = self.pca_dir / "pca_feature_loadings.csv"
            pca_components_df.to_csv(pca_components_path)
            logger.info(f"PCA feature loadings saved to: {pca_components_path}")

            # Also save explained variance for each component
            pca_variance_df = pd.DataFrame({
                'component': [f'PC{i+1}' for i in range(len(pca_var))],
                'variance_explained': pca_var,
                'cumulative_variance': cumulative_var
            })
            pca_variance_path = self.pca_dir / "pca_variance_explained.csv"
            pca_variance_df.to_csv(pca_variance_path, index=False)
            logger.info(f"PCA variance explained saved to: {pca_variance_path}")
            
            # Create PCA variance plot
            plt.figure(figsize=[15, 6])
            
            plt.subplot(221)
            plt.plot(pca_var[:50], ".")  # Show first 50 components
            plt.vlines(x=range(min(50, len(pca_var))), ymin=0, ymax=pca_var[:50])
            plt.ylabel("Proportion of variance")
            plt.xlabel("Principal component")
            plt.title("Proportion of variance explained\nby each principal component (first 50)")
            
            plt.subplot(222)
            plt.plot(cumulative_var[:50])
            plt.axhline(y=0.9, color='red', linestyle='--', label='90% variance')
            plt.axvline(x=components_90-1, color='red', linestyle='--', label=f'{components_90} components')
            plt.title("Cumulative variation explained\nby successive principal components")
            plt.xlabel("Principal component")
            plt.ylabel("Cumulative proportion of variance")
            plt.legend()
            
            plt.subplot(223)
            # Scree plot - looking for elbow
            plt.plot(range(1, min(21, len(pca_var)+1)), pca_var[:20], 'bo-')
            plt.title("Scree Plot (first 20 components)")
            plt.xlabel("Principal component")
            plt.ylabel("Proportion of variance")
            
            plt.subplot(224)
            # Show all components
            plt.plot(cumulative_var)
            plt.axhline(y=0.9, color='red', linestyle='--', label='90% variance')
            plt.axvline(x=components_90-1, color='red', linestyle='--', label=f'{components_90} components')
            plt.title(f"Full cumulative variance ({len(pca_var)} components)")
            plt.xlabel("Principal component")
            plt.ylabel("Cumulative proportion of variance")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.pca_dir / "pca_variance_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save variance data
            pca_results = pd.DataFrame({
                'component': range(1, len(pca_var) + 1),
                'variance_explained': pca_var,
                'cumulative_variance': cumulative_var
            })
            pca_results.to_csv(self.pca_dir / "pca_variance_data.csv", index=False)
            
            logger.info(f"PCA analysis completed. {components_90} components explain 90% variance.")
            logger.info(f"Total variance in first 50 components: {cumulative_var[49]:.3f}")
            
            return components_90
            
        except Exception as e:
            logger.error(f"Error creating PCA variance plot: {e}")
            return None
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> None:
        """
        Create correlation heatmap using pre-computed correlation matrix - saves to analysis/correlation/
        """
        logger.info("Creating correlation heatmap...")
        
        try:
            if correlation_matrix is None:
                logger.warning("No correlation matrix available for heatmap")
                return
                
            logger.info(f"Creating correlation heatmap for {len(correlation_matrix.columns)} features")
            
            # Save correlation matrix as CSV to correlation directory
            corr_csv_path = self.correlation_dir / "correlation_matrix.csv"
            correlation_matrix.to_csv(corr_csv_path)
            logger.info(f"Correlation matrix saved as CSV: {corr_csv_path}")
            
            # Create heatmap
            plt.figure(figsize=(20, 20))
            sns.heatmap(correlation_matrix, 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    xticklabels=False,
                    yticklabels=False,
                    cbar_kws={'label': 'Correlation Coefficient'})
            plt.title(f'Feature Correlation Matrix\n({len(correlation_matrix.columns)} features after missing/blacklisted removal)')
            
            plt.savefig(self.correlation_dir / "correlation_heatmap_all_features.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Correlation heatmap saved: {self.correlation_dir}/correlation_heatmap_all_features.png")
            
        except Exception as e:
            logger.warning(f"Could not create correlation heatmap: {e}")
    
    
    """
    Updated histogram creation methods for comprehensive folder structure
    """
    def create_comprehensive_histograms(self, raw_data: pd.DataFrame, normalized_data: pd.DataFrame) -> None:
        """
        Create comprehensive histogram plots for both raw and normalized data
        Creates organized folder structure with all combinations
        
        Args:
            raw_data: Raw feature data (before normalization)
            normalized_data: Normalized feature data (after normalization)
        """
        logger.info("Creating comprehensive histogram structure (raw + normalized)...")
        
        try:
            # Create comprehensive directory structure
            raw_base_dir = self.histograms_dir / "raw"
            normalized_base_dir = self.histograms_dir / "normalized"
            
            # All subdirectories we need
            subdirs = [
                "all_conditions",
                "all_conditions_log", 
                "dmso_only",
                "dmso_only_log",
                "treatments_only",
                "treatments_only_log"
            ]
            
            # Create all directories
            for base_dir in [raw_base_dir, normalized_base_dir]:
                for subdir in subdirs:
                    (base_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Create RAW histograms
            logger.info("Creating RAW data histograms...")
            self._create_data_histograms(raw_data, raw_base_dir, "RAW")
            
            # Create NORMALIZED histograms  
            logger.info("Creating NORMALIZED data histograms...")
            self._create_data_histograms(normalized_data, normalized_base_dir, "NORMALIZED")
            
            logger.info("Comprehensive histogram structure completed!")
            
        except Exception as e:
            logger.error(f"Error creating comprehensive histograms: {e}")

    def _create_data_histograms(self, data: pd.DataFrame, base_dir: Path, data_type: str) -> None:
        """
        Create all histogram combinations for a given dataset
        
        Args:
            data: DataFrame with feature data
            base_dir: Base directory (raw/ or normalized/)
            data_type: "RAW" or "NORMALIZED" for titles
        """
        if data is None or 'Metadata_perturbation_name' not in data.columns:
            logger.warning(f"Cannot create {data_type} histograms - missing data or perturbation column")
            return
        
        # Get feature columns
        feature_cols = [col for col in data.columns if not col.startswith('Metadata_')]
        n_features = len(feature_cols)
        
        if n_features == 0:
            logger.warning(f"No feature columns found in {data_type} data")
            return
        
        logger.info(f"Creating {data_type} histograms for {n_features} features...")
        
        # Split data
        dmso_mask = data['Metadata_perturbation_name'] == 'DMSO'
        dmso_data = data[dmso_mask]
        treatment_data = data[~dmso_mask]
        all_data = data
        
        logger.info(f"  All conditions: {len(all_data):,} samples")
        logger.info(f"  DMSO only: {len(dmso_data):,} samples") 
        logger.info(f"  Treatments only: {len(treatment_data):,} samples")
        
        # Create histogram combinations
        datasets = [
            (all_data, "all_conditions", "All Conditions"),
            (dmso_data, "dmso_only", "DMSO Controls Only"),
            (treatment_data, "treatments_only", "Treatments Only")
        ]
        
        features_per_plot = 20
        n_plots = (n_features + features_per_plot - 1) // features_per_plot
        
        for dataset, folder_name, display_name in datasets:
            if len(dataset) == 0:
                logger.warning(f"Skipping {display_name} - no data")
                continue
                
            logger.info(f"  Creating {display_name} histograms...")
            
            # Regular histograms
            regular_dir = base_dir / folder_name
            self._create_histogram_plots(
                dataset[feature_cols], regular_dir, n_plots, features_per_plot,
                f"{data_type} {display_name}", use_log_frequency=False
            )
            
            # Log frequency histograms
            log_dir = base_dir / f"{folder_name}_log"
            self._create_histogram_plots(
                dataset[feature_cols], log_dir, n_plots, features_per_plot,
                f"{data_type} {display_name} (LOG FREQUENCY)", use_log_frequency=True
            )

    def _create_histogram_plots(self, feature_data: pd.DataFrame, output_dir: Path, 
                            n_plots: int, features_per_plot: int, title_prefix: str,
                            use_log_frequency: bool = False) -> None:
        """
        Create histogram plots for feature data
        
        Args:
            feature_data: Feature data to plot
            output_dir: Directory to save plots
            n_plots: Number of plots to create
            features_per_plot: Features per plot
            title_prefix: Prefix for plot titles
            use_log_frequency: Whether to use log scale for y-axis
        """
        feature_cols = feature_data.columns
        n_features = len(feature_cols)
        
        for plot_idx in range(n_plots):
            start_idx = plot_idx * features_per_plot
            end_idx = min((plot_idx + 1) * features_per_plot, n_features)
            plot_features = feature_cols[start_idx:end_idx]
            
            n_features_this_plot = len(plot_features)
            cols = min(5, n_features_this_plot)
            rows = (n_features_this_plot + cols - 1) // cols
            
            fig_width = cols * 4
            fig_height = rows * 3
            
            plt.figure(figsize=(fig_width, fig_height))
            
            for i, feature in enumerate(plot_features):
                plt.subplot(rows, cols, i + 1)
                values = feature_data[feature].dropna()
                
                if len(values) > 0:
                    # Create histogram
                    plt.hist(values, bins=50, alpha=0.7, edgecolor='black')
                    
                    # Add reference lines
                    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero')
                    
                    mean_val = values.mean()
                    plt.axvline(x=mean_val, color='green', linestyle='-', alpha=0.7, 
                            label=f'Mean: {mean_val:.3f}')
                    
                    # Set log scale if requested
                    if use_log_frequency:
                        plt.yscale('log')
                        plt.ylabel('Frequency (log10)')
                    else:
                        plt.ylabel('Frequency')
                    
                    plt.title(feature[:30] + ('...' if len(feature) > 30 else ''), fontsize=8)
                    plt.xlabel('Value')
                    plt.legend(fontsize=6)
                    plt.xticks(rotation=45)
            
            # Hide empty subplots
            for j in range(i + 1, rows * cols):
                plt.subplot(rows, cols, j + 1)
                plt.axis('off')
            
            log_suffix = " (LOG FREQUENCY)" if use_log_frequency else ""
            plt.suptitle(f'{title_prefix}{log_suffix} - Plot {plot_idx + 1}/{n_plots} (Features {start_idx + 1}-{end_idx})', 
                        fontsize=14)
            plt.tight_layout()
            
            # Save plot
            filename = f"histograms_{plot_idx + 1:03d}.png"
            plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()

    # Update the existing methods in DataVisualizer class
    def create_enhanced_histograms(self, data: pd.DataFrame) -> None:
        """
        UPDATED: Create histogram plots for RAW data only
        Use create_comprehensive_histograms for full structure
        """
        logger.info("Creating enhanced histogram plots for RAW data...")
        
        try:
            if data is None:
                logger.warning("No data available for histograms")
                return
            
            # Create simplified directory structure for raw data only
            raw_hist_dir = self.histograms_dir / "raw" / "all_conditions"
            raw_hist_log_dir = self.histograms_dir / "raw" / "all_conditions_log"
            raw_hist_dir.mkdir(parents=True, exist_ok=True)
            raw_hist_log_dir.mkdir(parents=True, exist_ok=True)
            
            # Get feature data
            feature_cols = [col for col in data.columns if not col.startswith('Metadata_')]
            feature_data = data[feature_cols]
            n_features = len(feature_data.columns)
            
            logger.info(f"Creating histogram plots for {n_features} RAW features")
            
            features_per_plot = 20
            n_plots = (n_features + features_per_plot - 1) // features_per_plot
            
            # Create regular histograms
            self._create_histogram_plots(
                feature_data, raw_hist_dir, n_plots, features_per_plot,
                "RAW All Conditions", use_log_frequency=False
            )
            
            # Create log frequency histograms
            self._create_histogram_plots(
                feature_data, raw_hist_log_dir, n_plots, features_per_plot,
                "RAW All Conditions", use_log_frequency=True
            )
            
            logger.info(f"Created {n_plots} RAW histogram plots")
            
        except Exception as e:
            logger.warning(f"Could not create RAW histogram plots: {e}")

    def create_split_histograms_dmso_vs_treatment(self, normalized_data: pd.DataFrame) -> None:
        """
        UPDATED: Create comprehensive normalized histograms using new structure
        """
        logger.info("Creating comprehensive normalized histograms...")
        
        try:
            if normalized_data is None:
                logger.warning("No normalized data available for histograms")
                return
            
            # Use the comprehensive histogram creation
            self._create_data_histograms(normalized_data, self.histograms_dir / "normalized", "NORMALIZED")
            
            logger.info("Comprehensive normalized histograms completed!")
            
        except Exception as e:
            logger.warning(f"Could not create normalized histogram plots: {e}")

    
    def compute_multiple_umaps(self, data: pd.DataFrame, umap_params_list: List[Dict]) -> bool:
        """
        Compute multiple UMAP embeddings with different parameters
        """
        if umap is None:
            logger.error("UMAP package not available")
            return False
            
        logger.info(f"Computing {len(umap_params_list)} UMAP embeddings...")
        
        # Prepare data (same for all UMAP runs)
        if data is None:
            logger.error("No data available for UMAP")
            return False
        
        feature_cols = [col for col in data.columns if not col.startswith('Metadata_')]
        feature_data = data[feature_cols]
        
        # Remove constant features
        constant_features = feature_data.columns[feature_data.std() == 0]
        if len(constant_features) > 0:
            logger.info(f"Removing {len(constant_features)} constant features for UMAP")
            feature_data = feature_data.drop(columns=constant_features)
        
        # Apply PCA if needed
        if feature_data.shape[1] > 50:
            pca_components = min(50, feature_data.shape[1])
            logger.info(f"Applying PCA with {pca_components} components before UMAP")
            pca = PCA(n_components=pca_components)
            feature_data_pca = pca.fit_transform(feature_data)
        else:
            feature_data_pca = feature_data.values
        
        # Compute each UMAP
        for params in umap_params_list:
            name = params['name']
            logger.info(f"Computing UMAP with parameters: {name}")
            start_time = time.time()
            
            reducer = umap.UMAP(
                n_neighbors=params['n_neighbors'],
                min_dist=params['min_dist'],
                n_components=2,
                random_state=42,
                verbose=True
            )
            
            self.umap_results[name] = reducer.fit_transform(feature_data_pca)
            elapsed = time.time() - start_time
            logger.info(f"UMAP {name} completed in {elapsed:.1f} seconds")
        
        return True
    
    def compute_multiple_tsnes(self, data: pd.DataFrame, tsne_params_list: List[Dict]) -> bool:
        """
        Compute multiple t-SNE embeddings with different parameters
        """
        logger.info(f"Computing {len(tsne_params_list)} t-SNE embeddings...")
        
        # Prepare data (same for all t-SNE runs)
        if data is None:
            logger.error("No data available for t-SNE")
            return False
        
        feature_cols = [col for col in data.columns if not col.startswith('Metadata_')]
        feature_data = data[feature_cols]
        
        # Remove constant features
        constant_features = feature_data.columns[feature_data.std() == 0]
        if len(constant_features) > 0:
            logger.info(f"Removing {len(constant_features)} constant features for t-SNE")
            feature_data = feature_data.drop(columns=constant_features)
        
        # Apply PCA if needed
        if feature_data.shape[1] > 50:
            pca_components = min(50, feature_data.shape[1])
            logger.info(f"Applying PCA with {pca_components} components before t-SNE")
            pca = PCA(n_components=pca_components)
            feature_data_pca = pca.fit_transform(feature_data)
        else:
            feature_data_pca = feature_data.values
        
        # Compute each t-SNE
        for params in tsne_params_list:
            name = params['name']
            logger.info(f"Computing t-SNE with parameters: {name}")
            start_time = time.time()
            
            tsne = TSNE(
                n_components=2,
                perplexity=min(params['perplexity'], (len(data) - 1) // 3),
                random_state=42,
                verbose=1
            )
            
            self.tsne_results[name] = tsne.fit_transform(feature_data_pca)
            elapsed = time.time() - start_time
            logger.info(f"t-SNE {name} completed in {elapsed:.1f} seconds")
        
        return True
    
    def save_all_coordinates_to_single_file(self, data: pd.DataFrame) -> Optional[Path]:
        """
        Save all UMAP, t-SNE, and PCA coordinates with metadata to single CSV file
        UPDATED: Now includes treatment aggregation and hierarchical clustering
        """
        logger.info("Saving all embedding coordinates (including PCA) to single file...")
        
        try:
            # Get true metadata columns (existing logic)
            potential_metadata_cols = [col for col in data.columns if col.startswith('Metadata_')]
            
            feature_measurement_keywords = [
                'intensity', 'area', 'texture', 'radial', 'granularity', 'correlation',
                'colocalization', 'neighbors', 'location', 'shape', 'radialfeatures',
                'moments', 'zernike', 'radialdistribution', 'mean', 'median', 'std',
                'mad', 'min', 'max', 'integrated', 'massDisplacement'
            ]
            
            true_metadata_cols = []
            for col in potential_metadata_cols:
                if not any(keyword.lower() in col.lower() for keyword in feature_measurement_keywords):
                    true_metadata_cols.append(col)
            
            # Create coordinate data dictionary
            coords_data = {}
            
            # Add all metadata columns with truncation for specific columns
            for col in true_metadata_cols:
                if col in data.columns:
                    if col == 'Metadata_annotated_target_description':
                        # Truncate to 12 words for description
                        truncated_values = []
                        for value in data[col].values:
                            if pd.isna(value) or value == '':
                                truncated_values.append(value)
                            else:
                                words = str(value).split()
                                truncated = ' '.join(words[:12])
                                if len(words) > 12:
                                    truncated += '...'
                                truncated_values.append(truncated)
                        coords_data[col] = data[col].values  # Keep original
                        coords_data['Metadata_annotated_target_description_truncated'] = truncated_values  # Add truncated
                        
                    elif col == 'Metadata_annotated_target':
                        # Truncate to first 5 comma-separated values for target
                        truncated_values = []
                        for value in data[col].values:
                            if pd.isna(value) or value == '':
                                truncated_values.append(value)
                            else:
                                parts = str(value).split(',')
                                truncated = ', '.join(part.strip() for part in parts[:5])
                                if len(parts) > 5:
                                    truncated += '...'
                                truncated_values.append(truncated)
                        coords_data[col] = data[col].values  # Keep original
                        coords_data['Metadata_annotated_target_truncated'] = truncated_values  # Add truncated
                        
                    else:
                        coords_data[col] = data[col].values
            
            # NEW: Add PCA embeddings (first 50 components)
            if self.pca_model is not None:
                logger.info("Adding PCA embeddings to coordinate file...")
                
                # Get feature columns from the data
                feature_cols = [col for col in data.columns if not col.startswith('Metadata_')]
                feature_data = data[feature_cols]
                
                # Remove constant features (same as in create_pca_variance_plot)
                constant_features = feature_data.columns[feature_data.std() == 0]
                if len(constant_features) > 0:
                    logger.info(f"Removing {len(constant_features)} constant features for PCA embedding")
                    feature_data = feature_data.drop(columns=constant_features)
                
                # Transform the data using the fitted PCA model
                pca_embeddings = self.pca_model.transform(feature_data)
                
                # Save first 50 components (or all available if less than 50)
                n_components_to_save = min(50, pca_embeddings.shape[1])
                logger.info(f"Saving first {n_components_to_save} PCA components")
                
                # Add PCA component columns
                for i in range(n_components_to_save):
                    coords_data[f'pca_component_{i+1:02d}'] = pca_embeddings[:, i]
                
                logger.info(f"Added {n_components_to_save} PCA component columns to coordinate data")
            else:
                logger.warning("No PCA model available - skipping PCA embeddings")
            
            # Add all UMAP coordinates with prefixed names
            for name, embedding in self.umap_results.items():
                coords_data[f'umap_{name}_x'] = embedding[:, 0]
                coords_data[f'umap_{name}_y'] = embedding[:, 1]
            logger.info(f"Added {len(self.umap_results)} UMAP embedding coordinate pairs")
            
            # Add all t-SNE coordinates with prefixed names  
            for name, embedding in self.tsne_results.items():
                coords_data[f'tsne_{name}_x'] = embedding[:, 0]
                coords_data[f'tsne_{name}_y'] = embedding[:, 1]
            logger.info(f"Added {len(self.tsne_results)} t-SNE embedding coordinate pairs")
            
            # Create dataframe from dictionary (avoids fragmentation)
            coords_df = pd.DataFrame(coords_data)
            
            # Save to single coordinate file
            coords_path = self.coords_dir / "embedding_coordinates.csv"
            coords_df.to_csv(coords_path, index=False)
            
            # Enhanced logging
            pca_cols = [col for col in coords_df.columns if col.startswith('pca_component_')]
            umap_cols = [col for col in coords_df.columns if col.startswith('umap_')]
            tsne_cols = [col for col in coords_df.columns if col.startswith('tsne_')]
            metadata_cols = [col for col in coords_df.columns if col.startswith('Metadata_')]
            
            logger.info(f"All coordinates saved to: {coords_path}")
            logger.info(f"Saved {len(coords_df)} wells with {len(coords_df.columns)} total columns:")
            logger.info(f"  - PCA components: {len(pca_cols)} columns")
            logger.info(f"  - UMAP embeddings: {len(self.umap_results)} pairs ({len(umap_cols)} columns)")
            logger.info(f"  - t-SNE embeddings: {len(self.tsne_results)} pairs ({len(tsne_cols)} columns)")
            logger.info(f"  - Metadata columns: {len(metadata_cols)} columns")
            
            # Show first few PCA column names for verification
            if pca_cols:
                logger.info(f"PCA columns: {pca_cols[:5]}{'...' if len(pca_cols) > 5 else ''}")
            
            # NEW: Skip coordinate aggregation and hierarchical clustering - using pre-computed matrices instead
            logger.info("Skipping coordinate aggregation and hierarchical clustering")
            logger.info("Hierarchical clustering now uses pre-computed matrices via HierarchicalClusteringAnalyzer")
            
        except Exception as e:
            logger.error(f"Error saving coordinates: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def load_existing_coordinates(self) -> Dict[str, Any]:
        """
        Load existing UMAP and t-SNE coordinates from single coordinate file
        """
        logger.info("Loading existing embedding coordinates from single file...")
        
        coordinates = {
            'umap': {},
            'tsne': {}
        }
        
        # Look for the single coordinate file
        coord_file = self.coords_dir / "embedding_coordinates.csv"
        
        if not coord_file.exists():
            logger.error(f"Coordinate file not found: {coord_file}")
            return coordinates
        
        try:
            # Load coordinate data
            coord_data = pd.read_csv(coord_file)
            logger.info(f"Loaded coordinate data with shape: {coord_data.shape}")
            
            # Find all UMAP and t-SNE coordinate columns
            umap_coord_cols = [col for col in coord_data.columns if col.startswith('umap_') and col.endswith(('_x', '_y'))]
            tsne_coord_cols = [col for col in coord_data.columns if col.startswith('tsne_') and col.endswith(('_x', '_y'))]
            
            # Extract UMAP parameter names and create coordinate sets
            umap_params = set()
            for col in umap_coord_cols:
                if col.endswith('_x'):
                    param_name = col.replace('umap_', '').replace('_x', '')
                    umap_params.add(param_name)
            
            for param_name in umap_params:
                x_col = f'umap_{param_name}_x'
                y_col = f'umap_{param_name}_y'
                
                if x_col in coord_data.columns and y_col in coord_data.columns:
                    embedding = coord_data[[x_col, y_col]].values
                    coordinates['umap'][param_name] = {
                        'embedding': embedding,
                        'data': coord_data
                    }
                    logger.info(f"Loaded UMAP {param_name}: {embedding.shape}")
            
            # Extract t-SNE parameter names and create coordinate sets
            tsne_params = set()
            for col in tsne_coord_cols:
                if col.endswith('_x'):
                    param_name = col.replace('tsne_', '').replace('_x', '')
                    tsne_params.add(param_name)
            
            for param_name in tsne_params:
                x_col = f'tsne_{param_name}_x'
                y_col = f'tsne_{param_name}_y'
                
                if x_col in coord_data.columns and y_col in coord_data.columns:
                    embedding = coord_data[[x_col, y_col]].values
                    coordinates['tsne'][param_name] = {
                        'embedding': embedding,
                        'data': coord_data
                    }
                    logger.info(f"Loaded t-SNE {param_name}: {embedding.shape}")
            
        except Exception as e:
            logger.error(f"Error loading coordinates from {coord_file}: {e}")
        
        total_loaded = len(coordinates['umap']) + len(coordinates['tsne'])
        logger.info(f"Successfully loaded {total_loaded} embedding coordinate sets from single file")
        
        return coordinates

    def load_saved_pca_model(self, pca_dir: Path) -> bool:
        """
        Load previously saved PCA model and feature loadings
        """
        logger.info(f"Loading saved PCA model from {pca_dir}...")

        pca_loadings_path = pca_dir / "pca_feature_loadings.csv"
        pca_variance_path = pca_dir / "pca_variance_explained.csv"

        if not pca_loadings_path.exists():
            logger.error(f"PCA loadings file not found at: {pca_loadings_path}")
            return False

        if not pca_variance_path.exists():
            logger.error(f"PCA variance file not found at: {pca_variance_path}")
            return False

        try:
            logger.info(f"Loading PCA loadings from: {pca_loadings_path}")
            logger.info(f"Loading PCA variance from: {pca_variance_path}")
            
            # Load the feature loadings
            pca_loadings_df = pd.read_csv(pca_loadings_path, index_col=0)
            
            # Load variance explained
            pca_variance_df = pd.read_csv(pca_variance_path)
            variance_explained = pca_variance_df['variance_explained'].values
            
            logger.info(f"Loaded PCA with {len(variance_explained)} components")
            logger.info(f"Feature names: {len(pca_loadings_df.index)} features")
            logger.info(f"Variance in first component: {variance_explained[0]:.3f}")
            
            # Create a mock PCA object with the essential attributes
            class MockPCA:
                def __init__(self, components, variance_explained):
                    self.components_ = components.T  # Transpose back to (n_components, n_features)
                    self.explained_variance_ratio_ = variance_explained
                    self.n_components_ = len(variance_explained)
            
            # Create mock PCA model
            self.pca_model = MockPCA(pca_loadings_df.values, variance_explained)
            
            # Store feature names for later use
            self.pca_feature_names = pca_loadings_df.index.tolist()
            
            logger.info("Successfully loaded saved PCA model")
            return True
            
        except Exception as e:
            logger.error(f"Error loading saved PCA: {e}")
            return False