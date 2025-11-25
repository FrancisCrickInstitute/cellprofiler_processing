"""
Landmark Distance Threshold Analysis for Cell Painting Pipeline

This module analyzes the distribution of landmarks within various Euclidean distance 
thresholds to help determine optimal threshold values for landmark selection.

Analyzes thresholds: 0.10, 0.15, 0.20, 0.25, 0.30 (default)
Can be customized via config file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances
from typing import List, Dict, Any, Optional, Tuple

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def calculate_landmark_counts_at_thresholds(
    df: pd.DataFrame, 
    landmark_treatments: List[str],
    feature_cols: List[str], 
    thresholds: List[float]
) -> pd.DataFrame:
    """
    For each treatment, count how many landmarks fall within each distance threshold
    (VECTORIZED VERSION - uses COSINE distance to match landmark analysis)
    
    Args:
        df: DataFrame with well-level aggregated data and feature columns
        landmark_treatments: List of treatments identified as landmarks
        feature_cols: List of feature column names to use for distance calculation
        thresholds: List of distance thresholds to analyze
        
    Returns:
        DataFrame: Counts of landmarks within each threshold for each treatment
    """
    logger.info("Calculating landmark counts at multiple thresholds (VECTORIZED)...")
    logger.info(f"Total treatments in data: {df['Metadata_treatment'].nunique()}")
    logger.info(f"Total landmarks: {len(landmark_treatments)}")
    logger.info(f"Thresholds to analyze: {thresholds}")
    
    # ========================================
    # STEP 1: Create treatment-level centroids
    # ========================================
    logger.info("Creating treatment-level centroids...")
    
    # Group by treatment and take median of features
    treatment_groups = df.groupby('Metadata_treatment')[feature_cols].median().reset_index()
    treatments = treatment_groups['Metadata_treatment'].values
    treatment_embeddings = treatment_groups[feature_cols].values
    
    logger.info(f"  Created centroids for {len(treatments)} treatments")
    
    # ========================================
    # STEP 2: Extract landmark embeddings
    # ========================================
    logger.info("Extracting landmark embeddings...")
    
    # Get landmark embeddings from treatment centroids
    landmark_mask = treatment_groups['Metadata_treatment'].isin(landmark_treatments)
    landmark_names = treatment_groups[landmark_mask]['Metadata_treatment'].values
    landmark_matrix = treatment_groups[landmark_mask][feature_cols].values
    
    logger.info(f"  Extracted {len(landmark_names)} landmark embeddings")
    logger.info(f"  Landmark matrix shape: {landmark_matrix.shape}")
    logger.info(f"  Treatment matrix shape: {treatment_embeddings.shape}")
    
    # ========================================
    # STEP 3: Compute ALL distances at once (VECTORIZED!)
    # ========================================
    logger.info("Computing all pairwise COSINE distances...")
    logger.info("  (Using cosine distance to match landmark identification method)")
    
    # Compute distance matrix: treatments × landmarks (COSINE!)
    all_distances = cosine_distances(treatment_embeddings, landmark_matrix)
    
    logger.info(f"  Distance matrix shape: {all_distances.shape}")
    logger.info(f"  Computed {all_distances.size:,} distances in vectorized operation!")
    
    # ========================================
    # STEP 4: Build results dataframe
    # ========================================
    logger.info("Building results dataframe...")
    
    results = []
    
    # Get metadata for each treatment (first occurrence)
    treatment_metadata = df.groupby('Metadata_treatment').first().reset_index()
    treatment_metadata_dict = treatment_metadata.set_index('Metadata_treatment').to_dict('index')
    
    for i, treatment in enumerate(treatments):
        if (i + 1) % 5000 == 0:
            logger.info(f"  Processing treatment {i+1}/{len(treatments)}...")
        
        # Check if this treatment is itself a landmark
        is_landmark = treatment in landmark_treatments
        
        # Get pre-computed distances for this treatment
        treatment_distances = all_distances[i]
        
        # If treatment is a landmark, exclude self-distance
        if is_landmark:
            # Find index of self in landmark matrix
            self_idx = np.where(landmark_names == treatment)[0]
            if len(self_idx) > 0:
                # Create mask excluding self
                mask = np.ones(len(treatment_distances), dtype=bool)
                mask[self_idx[0]] = False
                treatment_distances = treatment_distances[mask]
        
        if len(treatment_distances) == 0:
            continue
        
        # Count landmarks within each threshold
        threshold_counts = {}
        for threshold in thresholds:
            count = np.sum(treatment_distances <= threshold)
            threshold_counts[f'landmarks_within_{threshold:.2f}'] = count
        
        # Add closest landmark distance
        closest_distance = np.min(treatment_distances)
        
        # Create result record
        result = {
            'Metadata_treatment': treatment,
            'is_landmark': is_landmark,
            'closest_landmark_distance': closest_distance,
            **threshold_counts,
            'total_landmarks_available': len(landmark_names) - (1 if is_landmark else 0)
        }
        
        # Add metadata columns from pre-computed dict
        if treatment in treatment_metadata_dict:
            metadata_row = treatment_metadata_dict[treatment]
            
            metadata_cols_to_include = [
                'Metadata_perturbation_name',
                'Metadata_compound_uM',
                'Metadata_PP_ID',
                'Metadata_PP_ID_uM',
                'Metadata_moa',
                'Metadata_lib_plate_order',
                'Metadata_library'
            ]
            
            for col in metadata_cols_to_include:
                if col in metadata_row:
                    result[col] = metadata_row[col]
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    logger.info(f"✓ Calculated threshold counts for {len(results_df)} treatments")
    
    return results_df


def plot_landmark_counts_by_threshold(
    threshold_counts_df: pd.DataFrame, 
    output_dir: Path, 
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create visualizations of landmark counts at different thresholds
    UPDATED: Matches reference script with cumulative distribution and violin plots
    
    Args:
        threshold_counts_df: DataFrame with landmark counts at each threshold
        output_dir: Path to save plots
        config: Configuration dictionary
    """
    logger.info("=" * 80)
    logger.info("CREATING LANDMARK THRESHOLD ANALYSIS PLOTS")
    logger.info("=" * 80)
    
    # Create output directories
    dist_dir = output_dir / 'distributions'
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    comp_dir = output_dir / 'comparisons'
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    by_lib_dir = output_dir / 'by_library'
    by_lib_dir.mkdir(parents=True, exist_ok=True)
    
    # Get threshold columns
    threshold_cols = [col for col in threshold_counts_df.columns if col.startswith('landmarks_within_')]
    thresholds = sorted([float(col.split('_')[-1]) for col in threshold_cols])
    
    logger.info(f"Creating plots for {len(thresholds)} thresholds: {thresholds}")
    
    # Separate landmark and non-landmark treatments
    landmarks_df = threshold_counts_df[threshold_counts_df['is_landmark'] == True].copy()
    non_landmarks_df = threshold_counts_df[threshold_counts_df['is_landmark'] == False].copy()
    
    logger.info(f"Total treatments: {len(threshold_counts_df)}")
    logger.info(f"  Landmarks: {len(landmarks_df)}")
    logger.info(f"  Non-landmarks: {len(non_landmarks_df)}")
    
    # =========================================================================
    # PLOT 1: Stacked bar chart - landmark counts by threshold (non-landmarks)
    # =========================================================================
    logger.info("Creating stacked bar chart...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for stacking
    bins = [0, 1, 2, 3, 4, 11, 21, np.inf]
    bin_labels = ['0', '1', '2', '3', '4-10', '11-20', '21+']
    
    plot_data = []
    for threshold in thresholds:
        col_name = f'landmarks_within_{threshold:.2f}'
        counts = non_landmarks_df[col_name]
        
        # Bin the counts
        binned = pd.cut(counts, bins=bins, labels=bin_labels, right=False)
        bin_counts = binned.value_counts().sort_index()
        
        plot_data.append({
            'threshold': f'{threshold:.2f}',
            **{label: bin_counts.get(label, 0) for label in bin_labels}
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create stacked bar chart
    x = np.arange(len(plot_df))
    width = 0.6
    bottom = np.zeros(len(plot_df))
    colors = sns.color_palette("viridis", len(bin_labels))
    
    for idx, label in enumerate(bin_labels):
        values = plot_df[label].values
        ax.bar(x, values, width, label=f'{label} landmarks', bottom=bottom, color=colors[idx])
        bottom += values
    
    ax.set_xlabel('Distance Threshold', fontsize=12)
    ax.set_ylabel('Number of Non-Landmark Treatments', fontsize=12)
    ax.set_title('Distribution of Landmark Counts at Different Distance Thresholds\n(Non-Landmark Treatments Only)', 
                fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['threshold'])
    ax.legend(title='Landmarks within threshold', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = dist_dir / 'landmark_counts_by_threshold_stacked.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {output_path}")
    
    # =========================================================================
    # PLOT 2: Cumulative distribution curves (NEW!)
    # =========================================================================
    logger.info("Creating cumulative distribution plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for threshold in thresholds:
        col_name = f'landmarks_within_{threshold:.2f}'
        counts = non_landmarks_df[col_name].values
        
        # Calculate cumulative distribution
        unique_counts = np.arange(0, counts.max() + 1)
        cumulative = [np.sum(counts >= count) / len(counts) * 100 for count in unique_counts]
        
        ax.plot(unique_counts, cumulative, marker='o', label=f'{threshold:.2f}', linewidth=2)
    
    ax.set_xlabel('Number of Landmarks Within Threshold', fontsize=12)
    ax.set_ylabel('Percentage of Non-Landmark Treatments (%)', fontsize=12)
    ax.set_title('Cumulative Distribution: % of Treatments with ≥N Landmarks\n(Non-Landmark Treatments Only)',
                fontsize=14, pad=20)
    ax.legend(title='Distance Threshold', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=-0.5)
    ax.set_ylim(0, 105)
    
    # Add reference lines
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(ax.get_xlim()[1] * 0.95, 52, '50%', fontsize=9, color='red', ha='right')
    ax.text(ax.get_xlim()[1] * 0.95, 77, '75%', fontsize=9, color='orange', ha='right')
    
    plt.tight_layout()
    output_path = dist_dir / 'cumulative_landmark_coverage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {output_path}")
    
    # =========================================================================
    # PLOT 3: Heatmap - thresholds vs landmark counts (NEW VERSION!)
    # =========================================================================
    logger.info("Creating threshold heatmap...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create matrix: rows = thresholds, columns = landmark count bins
    max_count = max([non_landmarks_df[f'landmarks_within_{t:.2f}'].max() for t in thresholds])
    count_bins = list(range(0, min(int(max_count) + 2, 51)))  # Cap at 50 for readability
    
    heatmap_data = []
    for threshold in thresholds:
        col_name = f'landmarks_within_{threshold:.2f}'
        counts = non_landmarks_df[col_name]
        
        row_data = []
        for count in count_bins:
            n_compounds = np.sum(counts == count)
            row_data.append(n_compounds)
        
        heatmap_data.append(row_data)
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                              index=[f'{t:.2f}' for t in thresholds],
                              columns=[str(c) for c in count_bins])
    
    # Plot heatmap
    sns.heatmap(heatmap_df, cmap='YlOrRd', annot=False, fmt='d', 
                cbar_kws={'label': 'Number of Treatments'}, ax=ax)
    
    ax.set_xlabel('Number of Landmarks Within Threshold', fontsize=12)
    ax.set_ylabel('Distance Threshold', fontsize=12)
    ax.set_title('Heatmap: Non-Landmark Treatments by Threshold and Landmark Count', fontsize=14, pad=20)
    
    # Only show every 5th x-tick for readability
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::5])
    ax.set_xticklabels([count_bins[int(i)] if int(i) < len(count_bins) else '' 
                        for i in xticks[::5]], rotation=0)
    
    plt.tight_layout()
    output_path = dist_dir / 'threshold_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {output_path}")
    
    # =========================================================================
    # PLOT 4: Violin plots - distribution of closest distances (NEW!)
    # =========================================================================
    logger.info("Creating violin plot...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create categories based on number of landmarks within 0.20
    if 'landmarks_within_0.20' in non_landmarks_df.columns:
        non_landmarks_df['landmark_category'] = pd.cut(
            non_landmarks_df['landmarks_within_0.20'],
            bins=[-1, 0, 1, 2, 5, np.inf],
            labels=['0', '1', '2', '3-5', '6+']
        )
        
        # Prepare data for violin plot
        violin_data = []
        categories = ['0', '1', '2', '3-5', '6+']
        
        for cat in categories:
            cat_data = non_landmarks_df[non_landmarks_df['landmark_category'] == cat]['closest_landmark_distance']
            if len(cat_data) > 0:
                for val in cat_data:
                    if pd.notna(val):
                        violin_data.append({'category': cat, 'distance': val})
        
        if violin_data:
            violin_df = pd.DataFrame(violin_data)
            
            # Create violin plot
            sns.violinplot(data=violin_df, x='category', y='distance', ax=ax, palette='Set2')
            
            # Add horizontal line at 0.2
            ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Current threshold (0.2)')
            
            ax.set_xlabel('Number of Landmarks Within 0.20 Threshold', fontsize=12)
            ax.set_ylabel('Closest Landmark Distance', fontsize=12)
            ax.set_title('Distribution of Closest Landmark Distances\nGrouped by Landmark Count (0.20 threshold)',
                        fontsize=14, pad=20)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            output_path = dist_dir / 'closest_distance_by_landmark_count.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Saved: {output_path}")
    
    # =========================================================================
    # PLOT 5: Distribution of closest landmark distances (histogram)
    # =========================================================================
    logger.info("Creating closest distance distribution plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    distances = non_landmarks_df['closest_landmark_distance'].dropna()
    
    # Create histogram
    ax.hist(distances, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Add threshold lines
    colors_thresh = sns.color_palette("Set1", len(thresholds))
    for idx, threshold in enumerate(thresholds):
        ax.axvline(threshold, color=colors_thresh[idx], linestyle='--', linewidth=1.5, 
                  label=f'Threshold {threshold:.2f}')
    
    # Add statistics
    mean_dist = distances.mean()
    median_dist = distances.median()
    ax.axvline(mean_dist, color='black', linestyle='-', linewidth=2, label=f'Mean: {mean_dist:.3f}')
    ax.axvline(median_dist, color='purple', linestyle='-', linewidth=2, label=f'Median: {median_dist:.3f}')
    
    ax.set_xlabel('Distance to Closest Landmark', fontsize=12)
    ax.set_ylabel('Number of Non-Landmark Treatments', fontsize=12)
    ax.set_title('Distribution of Closest Landmark Distances\n(Non-Landmark Treatments)', 
                fontsize=14, pad=20)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = dist_dir / 'closest_landmark_distance_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {output_path}")
    
    # =========================================================================
    # PLOT 6: Line plot - mean/median landmark counts vs threshold
    # =========================================================================
    logger.info("Creating mean/median trends plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    stats_data = []
    for threshold in thresholds:
        col_name = f'landmarks_within_{threshold:.2f}'
        counts = non_landmarks_df[col_name]
        
        stats_data.append({
            'threshold': threshold,
            'mean': counts.mean(),
            'median': counts.median(),
            'q25': counts.quantile(0.25),
            'q75': counts.quantile(0.75)
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Plot mean and median
    ax.plot(stats_df['threshold'], stats_df['mean'], marker='o', linewidth=2, 
           markersize=8, label='Mean', color='steelblue')
    ax.plot(stats_df['threshold'], stats_df['median'], marker='s', linewidth=2, 
           markersize=8, label='Median', color='orange')
    
    # Add shaded region for IQR
    ax.fill_between(stats_df['threshold'], stats_df['q25'], stats_df['q75'], 
                    alpha=0.2, color='steelblue', label='IQR (Q1-Q3)')
    
    ax.set_xlabel('Distance Threshold', fontsize=12)
    ax.set_ylabel('Number of Landmarks', fontsize=12)
    ax.set_title('Mean and Median Landmark Counts vs Distance Threshold\n(Non-Landmark Treatments)', 
                fontsize=14, pad=20)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = comp_dir / 'mean_median_vs_threshold.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {output_path}")
    
    # =========================================================================
    # PLOT 7: Heatmap - percentage of treatments with N+ landmarks (coverage)
    # =========================================================================
    logger.info("Creating coverage heatmap...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Calculate percentages for different cutoffs
    cutoffs = [0, 1, 2, 3, 5, 10, 20]
    heatmap_data = []
    
    for threshold in thresholds:
        col_name = f'landmarks_within_{threshold:.2f}'
        counts = non_landmarks_df[col_name]
        
        row_data = {'Threshold': f'{threshold:.2f}'}
        for cutoff in cutoffs:
            pct = (counts >= cutoff).sum() / len(counts) * 100
            row_data[f'{cutoff}+ landmarks'] = pct
        
        heatmap_data.append(row_data)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index('Threshold')
    
    # Create heatmap
    sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='RdYlGn', 
               cbar_kws={'label': 'Percentage of Treatments (%)'}, ax=ax,
               vmin=0, vmax=100)
    
    ax.set_title('Coverage: Percentage of Treatments with N+ Landmarks\n(Non-Landmark Treatments)', 
                fontsize=14, pad=20)
    ax.set_xlabel('Minimum Number of Landmarks', fontsize=12)
    ax.set_ylabel('Distance Threshold', fontsize=12)
    
    plt.tight_layout()
    output_path = comp_dir / 'coverage_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {output_path}")
    
    # =========================================================================
    # PLOT 8: Threshold comparison - histograms for key thresholds
    # =========================================================================
    logger.info("Creating threshold comparison plot...")
    
    if all(f'landmarks_within_{t:.2f}' in non_landmarks_df.columns for t in [0.10, 0.20, 0.30]):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, threshold in enumerate([0.10, 0.20, 0.30]):
            col_name = f'landmarks_within_{threshold:.2f}'
            counts = non_landmarks_df[col_name]
            
            # Create histogram
            axes[idx].hist(counts, bins=range(0, int(counts.max()) + 2), 
                          edgecolor='black', alpha=0.7, color=f'C{idx}')
            
            # Add statistics
            mean_val = counts.mean()
            median_val = counts.median()
            
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                            label=f'Mean: {mean_val:.1f}')
            axes[idx].axvline(median_val, color='orange', linestyle='--', linewidth=2, 
                            label=f'Median: {median_val:.1f}')
            
            axes[idx].set_xlabel('Number of Landmarks', fontsize=11)
            axes[idx].set_ylabel('Number of Treatments', fontsize=11)
            axes[idx].set_title(f'Threshold: {threshold:.2f}\n({len(counts):,} treatments)', fontsize=12)
            axes[idx].legend(fontsize=9)
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add text box with statistics
            stats_text = f'0 landmarks: {(counts == 0).sum():,} ({(counts == 0).sum()/len(counts)*100:.1f}%)\n'
            stats_text += f'1+ landmarks: {(counts >= 1).sum():,} ({(counts >= 1).sum()/len(counts)*100:.1f}%)\n'
            stats_text += f'3+ landmarks: {(counts >= 3).sum():,} ({(counts >= 3).sum()/len(counts)*100:.1f}%)'
            
            axes[idx].text(0.98, 0.98, stats_text, transform=axes[idx].transAxes,
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          fontsize=9)
        
        plt.suptitle('Comparison of Landmark Counts at Different Thresholds\n(Non-Landmark Treatments)', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        output_path = comp_dir / 'threshold_comparison_histograms.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {output_path}")
    
    # =========================================================================
    # PLOT 9: By library analysis (if library info available)
    # =========================================================================
    if 'Metadata_library' in non_landmarks_df.columns:
        logger.info("Creating by-library analysis...")
        
        libraries = non_landmarks_df['Metadata_library'].dropna().unique()
        logger.info(f"Found {len(libraries)} libraries")
        
        for library in libraries:
            lib_df = non_landmarks_df[non_landmarks_df['Metadata_library'] == library]
            
            if len(lib_df) < 10:  # Skip libraries with few compounds
                logger.info(f"  Skipping {library} (only {len(lib_df)} treatments)")
                continue
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot boxplots for each threshold
            box_data = []
            for threshold in thresholds:
                col_name = f'landmarks_within_{threshold:.2f}'
                for count in lib_df[col_name]:
                    box_data.append({
                        'threshold': f'{threshold:.2f}',
                        'count': count
                    })
            
            box_df = pd.DataFrame(box_data)
            
            sns.boxplot(data=box_df, x='threshold', y='count', ax=ax, palette='Set3')
            
            ax.set_xlabel('Distance Threshold', fontsize=12)
            ax.set_ylabel('Number of Landmarks', fontsize=12)
            ax.set_title(f'Landmark Counts by Threshold - {library}\n({len(lib_df):,} treatments)', 
                        fontsize=14, pad=20)
            ax.grid(axis='y', alpha=0.3)
            
            # Add mean markers
            for i, threshold in enumerate(thresholds):
                col_name = f'landmarks_within_{threshold:.2f}'
                mean_val = lib_df[col_name].mean()
                ax.plot(i, mean_val, marker='D', markersize=10, color='red', zorder=10)
            
            plt.tight_layout()
            safe_lib_name = str(library).replace('/', '_').replace(' ', '_')
            output_path = by_lib_dir / f'{safe_lib_name}_threshold_analysis.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"  ✓ Saved: {output_path}")
    
    # =========================================================================
    # Save summary statistics
    # =========================================================================
    logger.info("Saving summary statistics...")
    
    summary_stats = []
    for threshold in thresholds:
        col_name = f'landmarks_within_{threshold:.2f}'
        counts = non_landmarks_df[col_name]
        
        stats = {
            'threshold': threshold,
            'mean': counts.mean(),
            'median': counts.median(),
            'std': counts.std(),
            'min': counts.min(),
            'max': counts.max(),
            'q25': counts.quantile(0.25),
            'q75': counts.quantile(0.75),
            'treatments_with_0_landmarks': (counts == 0).sum(),
            'treatments_with_1plus_landmarks': (counts >= 1).sum(),
            'treatments_with_2plus_landmarks': (counts >= 2).sum(),
            'treatments_with_3plus_landmarks': (counts >= 3).sum(),
            'pct_with_0_landmarks': (counts == 0).sum() / len(counts) * 100,
            'pct_with_1plus_landmarks': (counts >= 1).sum() / len(counts) * 100,
            'pct_with_2plus_landmarks': (counts >= 2).sum() / len(counts) * 100,
            'pct_with_3plus_landmarks': (counts >= 3).sum() / len(counts) * 100
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    output_path = output_dir / 'summary_statistics.csv'
    summary_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved: {output_path}")
    
    # Print summary to log
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS (Non-Landmark Treatments)")
    logger.info("="*80)
    for _, row in summary_df.iterrows():
        logger.info(f"\nThreshold: {row['threshold']:.2f}")
        logger.info(f"  Mean landmarks: {row['mean']:.2f}")
        logger.info(f"  Median landmarks: {row['median']:.1f}")
        logger.info(f"  Treatments with 0 landmarks: {row['treatments_with_0_landmarks']:,} ({row['pct_with_0_landmarks']:.1f}%)")
        logger.info(f"  Treatments with 1+ landmarks: {row['treatments_with_1plus_landmarks']:,} ({row['pct_with_1plus_landmarks']:.1f}%)")
        logger.info(f"  Treatments with 3+ landmarks: {row['treatments_with_3plus_landmarks']:,} ({row['pct_with_3plus_landmarks']:.1f}%)")
    
    logger.info("\n✓ All threshold analysis plots created successfully")


def run_landmark_threshold_analysis(
    well_data: pd.DataFrame,
    landmark_file: Path,
    feature_cols: List[str],
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Main function to run landmark threshold analysis
    
    Args:
        well_data: DataFrame with well-level aggregated data
        landmark_file: Path to landmark CSV file (cellprofiler_landmarks.csv)
        feature_cols: List of feature column names to use for distance calculation
        output_dir: Base output directory for results
        config: Configuration dictionary
        
    Returns:
        bool: Success status
    """
    logger.info("=" * 80)
    logger.info("LANDMARK DISTANCE THRESHOLD ANALYSIS")
    logger.info("=" * 80)
    
    # Check if landmark file exists
    if not landmark_file.exists():
        logger.error(f"Landmark file not found: {landmark_file}")
        logger.error("Please run landmark analysis first!")
        return False
    
    try:
        # Load landmarks
        landmarks_df = pd.read_csv(landmark_file)
        logger.info(f"Loaded {len(landmarks_df)} landmarks from {landmark_file}")
        
        # Get landmark treatments (handle both column naming conventions)
        if 'treatment' in landmarks_df.columns:
            treatment_col = 'treatment'
        elif 'Metadata_treatment' in landmarks_df.columns:
            treatment_col = 'Metadata_treatment'
        else:
            logger.error("Treatment column not found in landmarks file!")
            logger.error(f"Available columns: {list(landmarks_df.columns)}")
            return False

        landmark_treatments = landmarks_df[treatment_col].unique().tolist()
        logger.info(f"Using column '{treatment_col}' for treatment names")
        logger.info(f"Identified {len(landmark_treatments)} unique landmark treatments")
        
        
        # Get thresholds from config or use defaults
        if config:
            thresholds = config.get('landmark_thresholds_to_analyze', [0.10, 0.15, 0.20, 0.25, 0.30])
        else:
            thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
        
        logger.info(f"Analyzing {len(thresholds)} thresholds: {thresholds}")
        
        # Create output directory
        threshold_output_dir = output_dir / 'landmark_threshold_analysis'
        threshold_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {threshold_output_dir}")
        
        # Calculate landmark counts at each threshold
        threshold_counts = calculate_landmark_counts_at_thresholds(
            well_data, 
            landmark_treatments, 
            feature_cols, 
            thresholds
        )
        
        # Save the threshold counts data
        output_path = threshold_output_dir / 'threshold_counts_per_treatment.csv'
        threshold_counts.to_csv(output_path, index=False)
        logger.info(f"✓ Saved threshold counts to: {output_path}")
        
        # Create visualizations
        plot_landmark_counts_by_threshold(threshold_counts, threshold_output_dir, config)
        
        logger.info("=" * 80)
        logger.info("✓ LANDMARK THRESHOLD ANALYSIS COMPLETED!")
        logger.info(f"Results saved to: {threshold_output_dir}")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"ERROR in landmark threshold analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False