import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import statsmodels.stats.multitest as smm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

class AnalysisConfig:
    """Central configuration for the analysis - change settings here"""
    
    # Data paths
    DATA_PATH = "/nemo/stp/hts/working/Phenix/Joe_Stan_Astrocytes_cell_paint_30062025/cellprofiler/processed_data_3/data/processed_image_data_well_level.parquet"
    OUTPUT_DIR = "/nemo/stp/hts/working/Phenix/Joe_Stan_Astrocytes_cell_paint_30062025/cellprofiler/processed_data_3/data/feature_analysis_2"
    
    # Experimental design
    CONTROL_LINES = ['CTRL1', 'CTRL2', 'CTRL3', 'CTRL5', 'CTRL6']
    MUTANT_LINES = ['CB1D', 'CB1E', 'NC2', 'GliA', 'GliB']
    
    # Analysis parameters
    FDR_THRESHOLD = 0.05
    MISSING_THRESHOLD = 0.05
    EXTREME_FEATURE_THRESHOLD = 100  # multiplier for median absolute difference
    
    # Plot settings
    PLOT_COLORS = {
        'Control_Untreated': '#2E8B57',      # Sea Green
        'Control_SB431542': '#4682B4',       # Steel Blue  
        'ALS_Untreated': '#DC143C',          # Crimson
        'ALS_SB431542': '#FF6347'            # Tomato
    }
    
    PLOT_DPI = 300
    DEFAULT_FIGSIZE = (12, 8)
    
    # Feature selection
    TOP_FEATURES_FOR_PLOTS = 8
    TOP_FEATURES_FOR_HEATMAP = 50

# =============================================================================
# HELPER FUNCTIONS  
# =============================================================================

def create_plot_directory(output_dir, subdir=''):
    """Quick helper to create plot directories"""
    plot_dir = os.path.join(output_dir, 'plots', subdir)
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def get_significance_stars(p_value):
    """Convert p-value to significance stars"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    elif p_value < 0.1:
        return '†'
    else:
        return 'ns'

# =============================================================================
# MAIN ANALYSIS CLASS - YOUR ORIGINAL CODE (MINOR ADJUSTMENTS FOR CONFIG)
# =============================================================================

class CellPaintingAnalysis:
    """
    Simplified Cell Painting Analysis Pipeline for Astrocyte Morphology
    Focus on paired Mann-Whitney U comparisons
    """
    
    def __init__(self):
        """Initialize the analysis with experimental design"""
        self.data_path = AnalysisConfig.DATA_PATH
        self.output_dir = AnalysisConfig.OUTPUT_DIR
        self.df = None
        self.morphology_features = None
        self.metadata_cols = None
        self.biological_replicates = None
        self.results = {}
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define control and mutant lines from config
        self.control_lines = AnalysisConfig.CONTROL_LINES
        self.mutant_lines = AnalysisConfig.MUTANT_LINES
        
        # Define comparisons - all vs Control_Untreated
        self.comparisons = {
            'ALS_Untreated_vs_Control': {
                'reference': 'Control_Untreated',
                'comparison': 'ALS_Untreated',
                'description': 'ALS untreated vs Control untreated'
            },
            'ALS_Treated_vs_Control': {
                'reference': 'Control_Untreated',
                'comparison': 'ALS_SB431542',
                'description': 'ALS treated vs Control untreated'
            },
            'Control_Treated_vs_Control': {
                'reference': 'Control_Untreated',
                'comparison': 'Control_SB431542',
                'description': 'Control treated vs Control untreated'
            }
        }
    
    def load_data(self):
        """Load and examine the data structure"""
        print("Loading well-level data...")
        self.df = pd.read_parquet(self.data_path)
        
        # Validate required columns exist
        required_cols = ['Metadata_Patient_Line', 'Metadata_Treatment', 'Metadata_lib_plate_order']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Identify metadata and morphology feature columns
        self.metadata_cols = [col for col in self.df.columns if col.startswith('Metadata_')]
        self.morphology_features = [col for col in self.df.columns if not col.startswith('Metadata_')]
        
        print(f"Data loaded: {self.df.shape[0]} wells, {self.df.shape[1]} columns")
        print(f"Morphology features: {len(self.morphology_features)}")
        print(f"Metadata columns: {len(self.metadata_cols)}")
        
        # Print replicate counts per condition
        if 'Metadata_Condition_Description' in self.df.columns:
            print("\nReplicate counts per Metadata_Condition_Description:")
            condition_counts = self.df['Metadata_Condition_Description'].value_counts().sort_index()
            for condition, count in condition_counts.items():
                print(f"  {condition}: {count} wells")
        else:
            print("Warning: Metadata_Condition_Description column not found!")
        
        return self
    
    def quality_control(self, missing_threshold=None, remove_extreme_features=True):
        """Basic quality control on features"""
        if missing_threshold is None:
            missing_threshold = AnalysisConfig.MISSING_THRESHOLD
            
        print(f"\nPerforming quality control (removing features with >{missing_threshold*100}% missing values)...")
        initial_features = len(self.morphology_features)
        
        # Remove features with no variance or too many missing values
        feature_variance = self.df[self.morphology_features].var()
        feature_missing = self.df[self.morphology_features].isnull().sum() / len(self.df)
        
        good_features = []
        for feature in self.morphology_features:
            if (feature_variance[feature] > 0 and 
                feature_missing[feature] <= missing_threshold and
                not self.df[feature].isnull().all()):
                good_features.append(feature)
        
        self.morphology_features = good_features
        print(f"Features retained after basic QC: {len(self.morphology_features)} out of {initial_features}")
        
        if remove_extreme_features:
            # Check for features with extreme ranges that might be artifacts
            print("\nChecking for features with extreme value ranges...")
            extreme_features = []
            
            for feature in self.morphology_features:
                feature_range = self.df[feature].max() - self.df[feature].min()
                feature_std = self.df[feature].std()
                
                # Flag features where range is > 1000x the standard deviation
                # or absolute max value is > 10000 (likely unnormalized)
                if (feature_range > 1000 * feature_std) or (self.df[feature].abs().max() > 10000):
                    extreme_features.append(feature)
                    print(f"  Flagged extreme feature: {feature[:50]}")
                    print(f"    Range: {feature_range:.0f}, Max: {self.df[feature].max():.0f}, Min: {self.df[feature].min():.0f}")
            
            if extreme_features:
                print(f"\nRemoving {len(extreme_features)} features with extreme ranges")
                self.morphology_features = [f for f in self.morphology_features if f not in extreme_features]
                
                # Save list of removed features
                with open(os.path.join(self.output_dir, 'removed_extreme_features.txt'), 'w') as f:
                    for feature in extreme_features:
                        f.write(f"{feature}\n")
        
        print(f"Final features retained: {len(self.morphology_features)}")
        return self

    def identify_extreme_features(self, threshold_multiplier=None):
        """
        Identify features with extreme mean differences that may be artifacts
        """
        if threshold_multiplier is None:
            threshold_multiplier = AnalysisConfig.EXTREME_FEATURE_THRESHOLD
            
        print(f"\nIdentifying features with extreme values (threshold: {threshold_multiplier}x median)...")
        
        extreme_features = set()
        extreme_details = []
        
        for comp_name, results_df in self.results.items():
            # Calculate median absolute mean difference for reference
            median_abs_diff = results_df['Mean_difference'].abs().median()
            
            # Identify extreme features
            extreme_mask = results_df['Mean_difference'].abs() > (threshold_multiplier * median_abs_diff)
            extreme_in_comp = results_df[extreme_mask]
            
            if len(extreme_in_comp) > 0:
                print(f"\n{comp_name}:")
                print(f"  Median absolute difference: {median_abs_diff:.3f}")
                print(f"  Found {len(extreme_in_comp)} extreme features:")
                
                for _, row in extreme_in_comp.iterrows():
                    extreme_features.add(row['Feature'])
                    extreme_details.append({
                        'Comparison': comp_name,
                        'Feature': row['Feature'],
                        'Mean_difference': row['Mean_difference'],
                        'Effect_size': row['Effect_size']
                    })
                    print(f"    {row['Feature'][:50]}: Mean diff = {row['Mean_difference']:.0f}, Effect size = {row['Effect_size']:.2f}")
        
        # Save extreme features list
        if extreme_details:
            extreme_df = pd.DataFrame(extreme_details)
            filename = os.path.join(self.output_dir, 'extreme_features_identified.csv')
            extreme_df.to_csv(filename, index=False)
            print(f"\n  ✓ Saved list of {len(extreme_features)} extreme features to: extreme_features_identified.csv")
        
        return self

    def create_biological_replicates(self):
        """Create biological replicates by averaging wells per patient line, treatment, and plate"""
        print("\nCreating biological replicates...")
    
        # Add disease status based on patient line
        self.df['Disease_Status'] = self.df['Metadata_Patient_Line'].apply(
            lambda x: 'Control' if x in self.control_lines else ('ALS' if x in self.mutant_lines else 'Unknown')
        )
    
        # NEW: Group by Patient_Line, Treatment, and Plate
        grouping_cols = ['Metadata_Patient_Line', 'Metadata_Treatment', 'Metadata_lib_plate_order']
        
        print(f"Grouping by: {grouping_cols}")
        
        # Average morphology features across technical replicates
        bio_rep_morphology = self.df.groupby(grouping_cols)[self.morphology_features].mean().reset_index()
        
        # Keep additional metadata
        metadata_to_keep = ['Metadata_Patient_Group', 'Metadata_Condition_Description']
        available_metadata = [col for col in metadata_to_keep if col in self.df.columns]
        
        if available_metadata:
            bio_rep_metadata = self.df.groupby(grouping_cols)[available_metadata].first().reset_index()
            self.biological_replicates = pd.merge(bio_rep_morphology, bio_rep_metadata, 
                                                on=grouping_cols, how='left')
        else:
            self.biological_replicates = bio_rep_morphology
        
        # Add disease status to biological replicates
        self.biological_replicates['Disease_Status'] = self.biological_replicates['Metadata_Patient_Line'].apply(
            lambda x: 'Control' if x in self.control_lines else ('ALS' if x in self.mutant_lines else 'Unknown')
        )
        
        # Create experimental group labels for compatibility with rest of code
        self.biological_replicates['Experimental_Group'] = (
            self.biological_replicates['Disease_Status'] + '_' + 
            self.biological_replicates['Metadata_Treatment'].str.replace('SB-431542', 'SB431542')
        )
        
        # Print detailed biological replicate counts
        print(f"\nBiological replicates created: {len(self.biological_replicates)} total")
        print("\nDetailed breakdown by Patient Line, Treatment, and Plate:")
        print("-" * 70)
        
        # Group and count
        replicate_counts = self.biological_replicates.groupby(
            ['Metadata_Patient_Line', 'Metadata_Treatment', 'Metadata_lib_plate_order']
        ).size().reset_index(name='Technical_Replicates_Averaged')
        
        # Add disease status for clarity
        replicate_counts['Disease_Status'] = replicate_counts['Metadata_Patient_Line'].apply(
            lambda x: 'Control' if x in self.control_lines else ('ALS' if x in self.mutant_lines else 'Unknown')
        )
        
        # Sort for better display
        replicate_counts = replicate_counts.sort_values(
            ['Disease_Status', 'Metadata_Treatment', 'Metadata_Patient_Line', 'Metadata_lib_plate_order']
        )
        
        # Print the counts
        for _, row in replicate_counts.iterrows():
            print(f"  {row['Disease_Status']:8} | {row['Metadata_Patient_Line']:6} | {row['Metadata_Treatment']:12} | Plate {row['Metadata_lib_plate_order']} | n=1 biological replicate")
        
        print("-" * 70)
        
        # Summary by experimental group and plate
        print("\nSummary by Experimental Group and Plate:")
        group_plate_counts = self.biological_replicates.groupby(
            ['Experimental_Group', 'Metadata_lib_plate_order']
        ).size().reset_index(name='Count')
        
        for _, row in group_plate_counts.iterrows():
            print(f"  {row['Experimental_Group']:20} | Plate {row['Metadata_lib_plate_order']} | {row['Count']} biological replicates")
        
        # Overall summary by experimental group
        print("\nOverall by Experimental Group (across all plates):")
        for group_name in self.biological_replicates['Experimental_Group'].unique():
            group_data = self.biological_replicates[
                self.biological_replicates['Experimental_Group'] == group_name
            ]
            count = len(group_data)
            unique_lines = group_data['Metadata_Patient_Line'].nunique()
            unique_plates = group_data['Metadata_lib_plate_order'].nunique()
            print(f"  {group_name}: {count} biological replicates ({unique_lines} lines × {unique_plates} plates)")
        
        return self
    
    def perform_comparisons_vs_control(self, fdr_threshold=None):
        """Perform comparisons vs Control_Untreated as specified in methods"""
        if fdr_threshold is None:
            fdr_threshold = AnalysisConfig.FDR_THRESHOLD
            
        print(f"\nPerforming comparisons vs Control_Untreated (FDR < {fdr_threshold})...")
        
        # Get reference group data
        reference_group = 'Control_Untreated'
        reference_data = self.biological_replicates[
            self.biological_replicates['Experimental_Group'] == reference_group
        ][self.morphology_features]
        
        print(f"Reference group ({reference_group}): {len(reference_data)} samples")
        
        self.results = {}
        
        for comparison_name, comparison_info in self.comparisons.items():
            comparison_group = comparison_info['comparison']
            description = comparison_info['description']
            
            print(f"\n{comparison_name}: {description}")
            
            # Get comparison group data
            comparison_data = self.biological_replicates[
                self.biological_replicates['Experimental_Group'] == comparison_group
            ][self.morphology_features]
            
            print(f"  Reference ({reference_group}): {len(reference_data)} samples")
            print(f"  Comparison ({comparison_group}): {len(comparison_data)} samples")
            
            if len(reference_data) < 2 or len(comparison_data) < 2:
                print(f"  ⚠️  Insufficient samples for comparison!")
                continue
            
            # Perform Wilcoxon rank-sum test for each feature
            p_values = []
            mean_differences = []
            effect_sizes = []
            feature_names = []
            
            for feature in self.morphology_features:
                ref_vals = reference_data[feature].dropna()
                comp_vals = comparison_data[feature].dropna()
                
                if len(ref_vals) < 2 or len(comp_vals) < 2:
                    continue
                
                try:
                    # Wilcoxon rank-sum test
                    statistic, p_value = mannwhitneyu(comp_vals, ref_vals, alternative='two-sided')
                    p_values.append(p_value)
                    
                    # Calculate mean difference (appropriate for z-scored data)
                    ref_mean = ref_vals.mean()
                    comp_mean = comp_vals.mean()
                    mean_difference = comp_mean - ref_mean
                    mean_differences.append(mean_difference)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(ref_vals)-1)*ref_vals.var() + 
                                        (len(comp_vals)-1)*comp_vals.var()) / 
                                    (len(ref_vals) + len(comp_vals) - 2))
                    if pooled_std > 0:
                        effect_size = (comp_mean - ref_mean) / pooled_std
                    else:
                        effect_size = 0
                    effect_sizes.append(effect_size)
                    
                    feature_names.append(feature)
                    
                except Exception as e:
                    continue
            
            if len(p_values) == 0:
                print(f"  ⚠️  No valid comparisons for this group!")
                continue
            
            # FDR correction using Benjamini-Hochberg method
            rejected, p_values_corrected, _, _ = smm.multipletests(p_values, method='fdr_bh')
            
            # Apply custom FDR threshold
            rejected = p_values_corrected < fdr_threshold
            
            # Store results
            results_df = pd.DataFrame({
                'Feature': feature_names,
                'P_value': p_values,
                'P_value_FDR': p_values_corrected,
                'Mean_difference': mean_differences,
                'Effect_size': effect_sizes,
                'Significant_FDR': rejected
            })
            
            # Sort by significance and effect size
            results_df = results_df.sort_values(['Significant_FDR', 'P_value_FDR'], 
                                            ascending=[False, True])
            
            self.results[comparison_name] = results_df
            
            significant_count = sum(rejected)
            total_features = len(results_df)
            print(f"  ✓ {significant_count}/{total_features} significant features (FDR < {fdr_threshold})")
            
            if significant_count > 0:
                print(f"  Top significant feature: {results_df.iloc[0]['Feature']}")
                print(f"    Mean Diff: {results_df.iloc[0]['Mean_difference']:.3f}")
                print(f"    FDR p-value: {results_df.iloc[0]['P_value_FDR']:.2e}")
        
        return self
    
    def create_summary_plot(self, save_plots=True):
        """Create summary bar plot of significant features per comparison"""
        print("\nCreating summary plot...")
        
        plot_dir = create_plot_directory(self.output_dir)
        
        # Prepare data for plotting
        comparison_names = []
        sig_counts = []
        total_counts = []
        
        for comp_name, results_df in self.results.items():
            comparison_names.append(comp_name.replace('_', ' '))
            sig_counts.append(sum(results_df['Significant_FDR']))
            total_counts.append(len(results_df))
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=AnalysisConfig.DEFAULT_FIGSIZE)
        
        x_pos = np.arange(len(comparison_names))
        bars = ax.bar(x_pos, sig_counts, color=['#2E8B57', '#4682B4', '#DC143C', '#FF6347'])
        
        # Add value labels on bars
        for i, (bar, sig_count, total_count) in enumerate(zip(bars, sig_counts, total_counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(sig_counts)*0.01,
                   f'{sig_count}\n({sig_count/total_count*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Significant Features (FDR < 0.05)', fontsize=12, fontweight='bold')
        ax.set_title('Significant Morphological Features by Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(comparison_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{plot_dir}/significant_features_summary.png', 
                       dpi=AnalysisConfig.PLOT_DPI, bbox_inches='tight')
        plt.show()
        
        return self
    
    def create_volcano_plots(self, save_plots=True):
        """Create enhanced volcano plots - both traditional and effect size versions"""
        print("\nCreating enhanced volcano plots...")
        
        plot_dir = create_plot_directory(self.output_dir)
        
        n_comparisons = len(self.results)
        
        # Create both traditional and effect size volcano plots
        for plot_type in ['traditional', 'effect_size']:
            print(f"  Creating {plot_type} volcano plots...")
            
            n_cols = 2 if n_comparisons > 1 else 1
            n_rows = (n_comparisons + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12*n_cols, 8*n_rows))
            if n_comparisons == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, (comp_name, results_df) in enumerate(self.results.items()):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                # Filter out infinite or NaN values
                if plot_type == 'traditional':
                    mask = (np.isfinite(results_df['Mean_difference']) & 
                        np.isfinite(-np.log10(results_df['P_value_FDR'])) &
                        (results_df['P_value_FDR'] > 0))
                else:  # effect_size
                    mask = (np.isfinite(results_df['Mean_difference']) & 
                        np.isfinite(results_df['Effect_size']))
                
                plot_data = results_df[mask].copy()
                
                if len(plot_data) == 0:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Enhanced color scheme and point sizes
                colors = []
                sizes = []
                alphas = []
                
                for _, row_data in plot_data.iterrows():
                    effect_size = abs(row_data['Effect_size'])
                    mean_diff = row_data['Mean_difference']
                    is_significant = row_data['Significant_FDR']
                    
                    # Color based on direction and effect size
                    if effect_size > 1.0:  # Large effect
                        color = '#DC143C' if mean_diff > 0 else '#1E90FF'  # Red/Blue
                        size = 80
                        alpha = 0.8
                    elif effect_size > 0.5:  # Medium effect
                        color = '#FF6347' if mean_diff > 0 else '#87CEEB'  # Light Red/Blue
                        size = 60
                        alpha = 0.7
                    else:  # Small effect
                        color = '#D3D3D3'  # Light gray
                        size = 30
                        alpha = 0.4
                    
                    # Boost opacity for significant features in traditional plot
                    if plot_type == 'traditional' and is_significant:
                        alpha = min(alpha + 0.2, 1.0)
                    
                    colors.append(color)
                    sizes.append(size)
                    alphas.append(alpha)
                
                # Create the appropriate volcano plot
                if plot_type == 'traditional':
                    y_data = -np.log10(plot_data['P_value_FDR'])
                    y_label = '-Log10(FDR-corrected p-value)'
                    title_suffix = 'Traditional Volcano Plot'
                    
                    # Add significance threshold line
                    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, 
                            linewidth=2, label='FDR = 0.05')
                    
                else:  # effect_size
                    y_data = plot_data['Effect_size'].abs()
                    y_label = '|Effect Size| (Cohen\'s d)'
                    title_suffix = 'Effect Size Volcano Plot'
                    
                    # Add effect size threshold lines
                    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, 
                            linewidth=2, label='Medium Effect (d=0.5)')
                    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, 
                            linewidth=2, label='Large Effect (d=0.8)')
                    ax.axhline(y=1.2, color='darkred', linestyle='--', alpha=0.7, 
                            linewidth=2, label='Very Large Effect (d=1.2)')
                
                # Create scatter plot with enhanced styling
                for i, (color, size, alpha) in enumerate(zip(colors, sizes, alphas)):
                    ax.scatter(plot_data.iloc[i]['Mean_difference'], y_data.iloc[i],
                            c=color, s=size, alpha=alpha, edgecolors='white', linewidth=0.5)
                
                # Add mean difference threshold lines
                ax.axvline(x=0.5, color='blue', linestyle=':', alpha=0.6, linewidth=2, label='Mean Diff = ±0.5')
                ax.axvline(x=-0.5, color='blue', linestyle=':', alpha=0.6, linewidth=2)
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
                
                # Enhanced annotations - top features by effect size
                if plot_type == 'effect_size':
                    top_features = plot_data.nlargest(3, 'Effect_size')
                else:
                    top_features = plot_data[plot_data['Significant_FDR']].nlargest(3, 'Effect_size')
                
                for _, row_data in top_features.iterrows():
                    feature_name = row_data['Feature'][:25] + '...' if len(row_data['Feature']) > 25 else row_data['Feature']
                    x_pos = row_data['Mean_difference']
                    y_pos = y_data[y_data.index == row_data.name].iloc[0]
                    
                    ax.annotate(feature_name, 
                            (x_pos, y_pos),
                            xytext=(10, 10), textcoords='offset points', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                            arrowprops=dict(arrowstyle='->', color='black', alpha=0.8))
                
                # Styling
                ax.set_xlabel('Mean Difference vs Control Untreated', fontsize=12, fontweight='bold')
                ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
                ax.set_title(f'{comp_name.replace("_", " ")}\n{title_suffix}', fontsize=13, fontweight='bold')
                ax.legend(loc='upper right', fontsize=10)
                ax.grid(True, alpha=0.2)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add counts
                total_features = len(plot_data)
                if plot_type == 'traditional':
                    sig_count = sum(plot_data['Significant_FDR'])
                    large_effect_count = sum(plot_data['Effect_size'].abs() > 0.8)
                    count_text = f'Total: {total_features}\nSignificant: {sig_count}\nLarge Effects: {large_effect_count}'
                else:
                    large_effect_count = sum(plot_data['Effect_size'].abs() > 0.8)
                    medium_effect_count = sum(plot_data['Effect_size'].abs() > 0.5)
                    count_text = f'Total: {total_features}\nLarge Effects: {large_effect_count}\nMedium+ Effects: {medium_effect_count}'
                
                ax.text(0.02, 0.98, count_text,
                    transform=ax.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Hide empty subplots
            for idx in range(n_comparisons, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                if n_rows > 1:
                    axes[row, col].set_visible(False)
                else:
                    axes[col].set_visible(False)
            
            plt.suptitle(f'{title_suffix}s - Enhanced Visualization', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            if save_plots:
                filename = f'{plot_dir}/volcano_plots_{plot_type}_enhanced.png'
                plt.savefig(filename, dpi=AnalysisConfig.PLOT_DPI, bbox_inches='tight')
                print(f"  ✓ Saved: volcano_plots_{plot_type}_enhanced.png")
            plt.show()
        
        return self

    def create_top_features_plot(self, comparison='ALS_Untreated_vs_Control', top_n=None, save_plots=True):
        """Create enhanced box plots for top significant features with individual points and statistics"""
        if top_n is None:
            top_n = AnalysisConfig.TOP_FEATURES_FOR_PLOTS
            
        print(f"\nCreating enhanced box plots for {comparison} (top {top_n})...")
        
        if comparison not in self.results:
            print(f"Comparison {comparison} not found!")
            return self
        
        results_df = self.results[comparison]
        
        # Get features that are significant in ANY comparison (not just this one)
        all_significant_features = set()
        for comp_name, comp_results in self.results.items():
            sig_in_comp = comp_results[comp_results['Significant_FDR']]['Feature'].tolist()
            all_significant_features.update(sig_in_comp)

        if len(all_significant_features) == 0:
            print("No significant features found in any comparison!")
            return self

        # Filter current results to only include features significant somewhere
        sig_features = results_df[results_df['Feature'].isin(all_significant_features)]
        
        # Sort by effect size magnitude and take top N
        top_features = sig_features.reindex(sig_features['Effect_size'].abs().sort_values(ascending=False).index).head(top_n)
        
        plot_dir = create_plot_directory(self.output_dir)
        
        # Prepare data for plotting with better group names
        plot_data = []
        group_order = ['Control Untreated', 'Control SB431542', 'ALS Untreated', 'ALS SB431542']
        group_colors = AnalysisConfig.PLOT_COLORS
        
        for group_name in ['Control_Untreated', 'Control_SB431542', 'ALS_Untreated', 'ALS_SB431542']:
            group_data = self.biological_replicates[
                self.biological_replicates['Experimental_Group'] == group_name
            ]
            
            for feature in top_features['Feature']:
                for _, row in group_data.iterrows():
                    plot_data.append({
                        'Group': group_name.replace('_', ' '),
                        'Feature': feature,
                        'Value': row[feature],
                        'Patient_Line': row.get('Metadata_Patient_Line', 'Unknown'),
                        'Plate': row.get('Metadata_plate_barcode', 'Unknown')
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Calculate grid layout
        n_cols = 2
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Statistical testing function
        def get_comparison_stats(data, feature, reference_group='Control Untreated'):
            """Calculate Mann-Whitney U statistics between reference and other groups"""
            ref_data = data[(data['Feature'] == feature) & (data['Group'] == reference_group)]['Value'].dropna()
            stats_results = {}
            
            for group in group_order:
                if group != reference_group:
                    group_data = data[(data['Feature'] == feature) & (data['Group'] == group)]['Value'].dropna()
                    
                    if len(ref_data) >= 3 and len(group_data) >= 3:
                        try:
                            from scipy.stats import mannwhitneyu
                            statistic, p_value = mannwhitneyu(group_data, ref_data, alternative='two-sided')
                            
                            # Use helper function for significance stars
                            significance = get_significance_stars(p_value)
                            
                            stats_results[group] = {
                                'p_value': p_value,
                                'significance': significance,
                                'n_ref': len(ref_data),
                                'n_comp': len(group_data)
                            }
                        except:
                            stats_results[group] = {'p_value': np.nan, 'significance': 'ns', 'n_ref': len(ref_data), 'n_comp': len(group_data)}
                    else:
                        stats_results[group] = {'p_value': np.nan, 'significance': 'ns', 'n_ref': len(ref_data), 'n_comp': len(group_data)}
            
            return stats_results
        
        # Create box plots
        for idx, (_, feature_row) in enumerate(top_features.iterrows()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            feature = feature_row['Feature']
            feature_data = plot_df[plot_df['Feature'] == feature]
            
            # Create the box plot with custom styling
            box_plot = ax.boxplot([feature_data[feature_data['Group'] == group]['Value'].dropna() 
                                for group in group_order],
                                positions=range(len(group_order)),
                                patch_artist=True,
                                widths=0.6,
                                showfliers=False,  # We'll add individual points instead
                                medianprops=dict(color='white', linewidth=2),
                                boxprops=dict(linewidth=1.5),
                                whiskerprops=dict(linewidth=1.5),
                                capprops=dict(linewidth=1.5))
            
            # Color the boxes
            for patch, group in zip(box_plot['boxes'], group_order):
                patch.set_facecolor(group_colors[group.replace(' ', '_')])
                patch.set_alpha(0.7)
            
            # Add individual data points with jitter
            for i, group in enumerate(group_order):
                group_values = feature_data[feature_data['Group'] == group]['Value'].dropna()
                if len(group_values) > 0:
                    # Add jitter to x-coordinates
                    np.random.seed(42)  # For reproducible jitter
                    x_jitter = np.random.normal(i, 0.08, len(group_values))
                    
                    ax.scatter(x_jitter, group_values, 
                            color='white', 
                            s=25, 
                            alpha=0.8, 
                            edgecolors='black', 
                            linewidth=0.5,
                            zorder=10)
            
            # Add statistical annotations
            stats_results = get_comparison_stats(plot_df, feature)
            
            # Find the maximum y value for positioning annotations
            max_y = feature_data['Value'].max()
            min_y = feature_data['Value'].min()
            y_range = max_y - min_y
            
            # Add significance annotations above each comparison group
            for i, group in enumerate(['Control SB431542', 'ALS Untreated', 'ALS SB431542']):
                if group in stats_results:
                    significance = stats_results[group]['significance']
                    if significance != 'ns':
                        # Position annotation above the box
                        y_pos = max_y + y_range * (0.1 + 0.05 * (i % 2))  # Stagger heights
                        ax.text(i + 1, y_pos, significance, 
                            ha='center', va='bottom', 
                            fontsize=12, fontweight='bold',
                            color='black')
            
            # Customize the plot
            feature_short = feature.replace('Intensity_', '').replace('AreaShape_', '').replace('Texture_', '')
            if len(feature_short) > 40:
                # Split long feature names into two lines
                words = feature_short.split('_')
                if len(words) > 1:
                    mid_point = len(words) // 2
                    line1 = '_'.join(words[:mid_point])
                    line2 = '_'.join(words[mid_point:])
                    feature_short = f"{line1[:30]}\n{line2[:30]}"
                else:
                    feature_short = feature_short[:40] + '...'
            
            ax.set_title(feature_short, fontsize=11, fontweight='bold', pad=15)
            ax.set_xticks(range(len(group_order)))
            ax.set_xticklabels(group_order, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Feature Value', fontsize=10)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            
            # Add sample sizes and statistics in corner
            mean_difference = feature_row['Mean_difference']
            effect_size = feature_row['Effect_size']
            p_value_fdr = feature_row['P_value_FDR']
            
            stats_text = f'Effect size: {effect_size:.2f}\nFDR p-val: {p_value_fdr:.1e}'
            
            # Add sample size info
            n_control = len(feature_data[feature_data['Group'] == 'Control Untreated']['Value'].dropna())
            stats_text += f'\nn = {n_control} per group'
            
            ax.text(0.02, 0.98, stats_text,
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
            
            # Improve axis formatting
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)
        
        # Hide empty subplots
        for idx in range(len(top_features), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        # Add overall title and legend
        plt.suptitle(f'Top Significant Features: {comparison.replace("_", " ")}\n' +
                    f'Individual Data Points with Statistical Comparisons vs Control Untreated', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Add legend for significance levels
        legend_text = 'Significance: *** p<0.001, ** p<0.01, * p<0.05, † p<0.1, ns = not significant'
        fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.08)
        
        if save_plots:
            filename = f'{plot_dir}/enhanced_boxplots_{comparison}.png'
            plt.savefig(filename, dpi=AnalysisConfig.PLOT_DPI, bbox_inches='tight')
            print(f"  ✓ Saved: enhanced_boxplots_{comparison}.png")
        plt.show()
        
        return self
    
    def create_single_feature_plots(self, comparison='ALS_Untreated_vs_Control', top_n=10, save_plots=True):
        """Create individual plots for each significant feature showing all four conditions"""
        print(f"\nCreating individual feature plots for {comparison} (top {top_n})...")
        
        if comparison not in self.results:
            print(f"Comparison {comparison} not found!")
            return self
        
        results_df = self.results[comparison]
        
        # Get features that are significant in ANY comparison (not just this one)
        all_significant_features = set()
        for comp_name, comp_results in self.results.items():
            sig_in_comp = comp_results[comp_results['Significant_FDR']]['Feature'].tolist()
            all_significant_features.update(sig_in_comp)

        if len(all_significant_features) == 0:
            print("No significant features found in any comparison!")
            return self

        # Filter current results to only include features significant somewhere
        sig_features = results_df[results_df['Feature'].isin(all_significant_features)]
        
        # Sort by effect size magnitude and take top N
        top_features = sig_features.reindex(sig_features['Effect_size'].abs().sort_values(ascending=False).index).head(top_n)
        
        plot_dir = create_plot_directory(self.output_dir, 'individual_features')
        
        # Group settings
        group_order = ['Control_Untreated', 'Control_SB431542', 'ALS_Untreated', 'ALS_SB431542']
        group_labels = ['Control\nUntreated', 'Control\nSB431542', 'ALS\nUntreated', 'ALS\nSB431542']
        group_colors = AnalysisConfig.PLOT_COLORS
        
        # Create individual plots for each feature
        for idx, (_, feature_row) in enumerate(top_features.iterrows()):
            feature = feature_row['Feature']
            
            # Prepare data for this feature
            plot_data = []
            for group_name in group_order:
                group_data = self.biological_replicates[
                    self.biological_replicates['Experimental_Group'] == group_name
                ]
                
                for _, row in group_data.iterrows():
                    plot_data.append({
                        'Group': group_name,
                        'Value': row[feature],
                        'Patient_Line': row.get('Metadata_Patient_Line', 'Unknown')
                    })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create box plot
            box_data = [plot_df[plot_df['Group'] == group]['Value'].dropna() for group in group_order]
            box_plot = ax.boxplot(box_data,
                                positions=range(len(group_order)),
                                patch_artist=True,
                                widths=0.6,
                                showfliers=False,
                                medianprops=dict(color='white', linewidth=2),
                                boxprops=dict(linewidth=1.5),
                                whiskerprops=dict(linewidth=1.5),
                                capprops=dict(linewidth=1.5))
            
            # Color the boxes
            for patch, group in zip(box_plot['boxes'], group_order):
                patch.set_facecolor(group_colors[group])
                patch.set_alpha(0.7)
            
            # Add individual data points with jitter
            for i, group in enumerate(group_order):
                group_values = plot_df[plot_df['Group'] == group]['Value'].dropna()
                if len(group_values) > 0:
                    np.random.seed(42)  # For reproducible jitter
                    x_jitter = np.random.normal(i, 0.08, len(group_values))
                    
                    ax.scatter(x_jitter, group_values, 
                            color='white', 
                            s=40, 
                            alpha=0.9, 
                            edgecolors='black', 
                            linewidth=0.8,
                            zorder=10)
            
            # Perform Mann-Whitney U tests vs Control_Untreated
            control_data = plot_df[plot_df['Group'] == 'Control_Untreated']['Value'].dropna()
            
            # Add significance annotations
            max_y = plot_df['Value'].max()
            min_y = plot_df['Value'].min()
            y_range = max_y - min_y
            
            for i, group in enumerate(['Control_SB431542', 'ALS_Untreated', 'ALS_SB431542']):
                group_data = plot_df[plot_df['Group'] == group]['Value'].dropna()
                
                if len(control_data) >= 3 and len(group_data) >= 3:
                    try:
                        from scipy.stats import mannwhitneyu
                        statistic, p_value = mannwhitneyu(group_data, control_data, alternative='two-sided')
                        
                        # Use helper function for significance stars
                        significance = get_significance_stars(p_value)
                        
                        # Always show significance annotation (including 'ns')
                        y_pos = max_y + y_range * (0.05 + 0.03 * i)
                        color = 'black' if significance != 'ns' else 'gray'
                        ax.text(i + 1, y_pos, significance, 
                            ha='center', va='bottom', 
                            fontsize=12 if significance != 'ns' else 10, 
                            fontweight='bold',
                            color=color)
                                            
                    except Exception as e:
                        continue
            
            # Customize the plot
            feature_short = feature.replace('Intensity_', '').replace('AreaShape_', '').replace('Texture_', '')
            if len(feature_short) > 60:
                words = feature_short.split('_')
                if len(words) > 1:
                    mid_point = len(words) // 2
                    line1 = '_'.join(words[:mid_point])
                    line2 = '_'.join(words[mid_point:])
                    feature_short = f"{line1}\n{line2}"
                else:
                    feature_short = feature_short[:60] + '...'
            
            ax.set_title(f'{feature_short}\n(Effect size: {feature_row["Effect_size"]:.2f}, FDR p-val: {feature_row["P_value_FDR"]:.1e})', 
                        fontsize=12, fontweight='bold', pad=20)
            ax.set_xticks(range(len(group_order)))
            ax.set_xticklabels(group_labels, fontsize=11)
            ax.set_ylabel('Feature Value', fontsize=12, fontweight='bold')
            
            # Add grid and clean up axes
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add sample size info
            n_control = len(control_data)
            ax.text(0.02, 0.98, f'n = {n_control} per group\nMann-Whitney U test vs Control Untreated',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
            
            plt.tight_layout()
            
            if save_plots:
                # Clean feature name for filename
                clean_name = feature.replace('/', '_').replace('\\', '_').replace(' ', '_')[:50]
                filename = f'{plot_dir}/feature_{idx+1:02d}_{clean_name}.png'
                plt.savefig(filename, dpi=AnalysisConfig.PLOT_DPI, bbox_inches='tight')
                
            plt.show()
            
        print(f"  ✓ Created {len(top_features)} individual feature plots")
        return self

    def create_heatmap(self, top_n_per_comparison=5, save_plots=True):
        """Create heatmap of top significant features across all groups"""
        print(f"\nCreating heatmap (top {top_n_per_comparison} features per comparison)...")
        
        # Collect top features from all comparisons
        all_top_features = set()
        for comp_name, results_df in self.results.items():
            sig_features = results_df[results_df['Significant_FDR']].head(top_n_per_comparison)
            all_top_features.update(sig_features['Feature'].tolist())
        
        if len(all_top_features) == 0:
            print("No significant features found for heatmap!")
            return self
        
        print(f"Selected {len(all_top_features)} unique features for heatmap")
        
        # Create matrix of mean values for each group
        heatmap_data = []
        group_names = []
        
        for group_name in ['Control_Untreated', 'Control_SB431542', 'ALS_Untreated', 'ALS_SB431542']:
            group_data = self.biological_replicates[
                self.biological_replicates['Experimental_Group'] == group_name
            ]
            group_means = group_data[list(all_top_features)].mean()
            heatmap_data.append(group_means.values)
            group_names.append(group_name.replace('_', ' '))
        
        heatmap_matrix = np.array(heatmap_data)
        
        # Z-score normalization across groups for each feature
        heatmap_matrix_zscore = ((heatmap_matrix - heatmap_matrix.mean(axis=0)) / 
                                heatmap_matrix.std(axis=0))
        
        # Create heatmap
        plt.figure(figsize=(20, 8))
        
        # Truncate feature names for display
        feature_labels = [f[:30] + '...' if len(f) > 30 else f for f in all_top_features]
        
        sns.heatmap(heatmap_matrix_zscore, 
                   xticklabels=feature_labels,
                   yticklabels=group_names,
                   cmap='RdBu_r', 
                   center=0,
                   cbar_kws={'label': 'Z-score'},
                   linewidths=0.5)
        
        plt.title('Top Differential Features Across Experimental Groups', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Morphological Features', fontsize=12)
        plt.ylabel('Experimental Groups', fontsize=12)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plot_dir = create_plot_directory(self.output_dir)
        if save_plots:
            plt.savefig(f'{plot_dir}/features_heatmap.png', dpi=AnalysisConfig.PLOT_DPI, bbox_inches='tight')
        plt.show()
        
        return self
    

    def create_mean_difference_heatmap(self, top_n=None, save_plots=True):
        """
        Create a mean difference heatmap showing top differential features across all conditions vs Control
        """
        if top_n is None:
            top_n = AnalysisConfig.TOP_FEATURES_FOR_HEATMAP
            
        print(f"\nCreating mean difference heatmap (top {top_n} features)...")
        
        plot_dir = create_plot_directory(self.output_dir)
        
        # Collect mean differences from all comparisons
        mean_diff_matrix = {}
        comparison_labels = []
        
        for comp_name, results_df in self.results.items():
            # Clean up comparison name for display
            if 'ALS_Untreated_vs_Control' in comp_name:
                label = 'ALS Untreated'
            elif 'ALS_Treated_vs_Control' in comp_name:
                label = 'ALS Treated'
            elif 'Control_Treated_vs_Control' in comp_name:
                label = 'Control Treated'
            else:
                label = comp_name.replace('_', ' ')
            
            comparison_labels.append(label)
            
            # Store mean differences for each feature
            for _, row in results_df.iterrows():
                feature = row['Feature']
                mean_difference = row['Mean_difference']
                
                if feature not in mean_diff_matrix:
                    mean_diff_matrix[feature] = {}
                mean_diff_matrix[feature][label] = mean_difference
        
        # Convert to DataFrame
        fc_df = pd.DataFrame.from_dict(mean_diff_matrix, orient='index')
        fc_df = fc_df.fillna(0)  # Fill missing values with 0
        
        # Calculate maximum absolute mean difference for ranking
        fc_df['max_abs_diff'] = fc_df.abs().max(axis=1)
        
        # Select top features (or all if top_n is None)
        if top_n is None:
            top_features_df = fc_df.sort_values('max_abs_diff', ascending=False)
            top_n_display = len(fc_df)
        else:
            top_features_df = fc_df.nlargest(top_n, 'max_abs_diff')
            top_n_display = top_n
        
        # Remove the ranking column for plotting
        heatmap_data = top_features_df.drop('max_abs_diff', axis=1)
        
        print(f"  Selected {len(heatmap_data)} features for heatmap")
        print(f"  Comparisons: {list(heatmap_data.columns)}")
        print(f"  Mean difference range: {heatmap_data.min().min():.2f} to {heatmap_data.max().max():.2f}")
        
        # Create the heatmap
        figure_height = max(12, top_n_display * 0.3) if top_n_display else 200  # Cap at reasonable height
        plt.figure(figsize=(8, min(figure_height, 200)))  # Limit max height to 200
        
        # Truncate feature names for better display
        feature_labels = [feat[:50] + '...' if len(feat) > 50 else feat for feat in heatmap_data.index]
        
        # Create heatmap with diverging colormap centered at 0
        sns.heatmap(heatmap_data, 
                   xticklabels=heatmap_data.columns,
                   yticklabels=feature_labels,
                   cmap='RdBu_r',  # Red-Blue diverging, reversed so red=positive
                   center=0,
                   cbar_kws={'label': 'Mean Difference vs Control Untreated'},
                   linewidths=0.1,
                   annot=False,  # Set to True if you want numbers in cells
                   fmt='.2f')
        
        plt.title(f'Top {top_n_display} Differential Features: Mean Difference vs Control Untreated\n' +
         f'(Ranked by Maximum Absolute Mean Difference)',
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Conditions', fontsize=12, fontweight='bold')
        plt.ylabel('Morphological Features', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0, fontsize=8)
        
        # Add colorbar interpretation
        plt.figtext(0.02, 0.02, 'Red = Higher in condition, Blue = Lower in condition (vs Control Untreated)', 
                   fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{plot_dir}/mean_difference_heatmap_top{top_n}.png', 
                       dpi=AnalysisConfig.PLOT_DPI, bbox_inches='tight')
            print(f"  ✓ Saved: mean_difference_heatmap_top{top_n}.png")                
        plt.show()
        
        # Export the mean difference matrix to CSV
        export_df = heatmap_data.copy()
        export_df.index.name = 'Feature'
        filename = os.path.join(self.output_dir, f'mean_difference_matrix_top{top_n}.csv')
        export_df.to_csv(filename)
        print(f"  ✓ Saved data: mean_difference_matrix_top{top_n}.csv")
        
        return self
    

    def create_effect_size_summary(self, min_effect_size=0.5, save_plots=True):
        """
        Create summary focusing on effect sizes regardless of statistical significance
        """
        print(f"\nCreating effect size summary (|effect size| > {min_effect_size})...")
        
        plot_dir = create_plot_directory(self.output_dir)
        
        # Collect features with large effect sizes
        large_effects = []
        
        for comp_name, results_df in self.results.items():
            large_effect_features = results_df[results_df['Effect_size'].abs() > min_effect_size]
            
            if len(large_effect_features) > 0:
                for _, row in large_effect_features.iterrows():
                    large_effects.append({
                        'Comparison': comp_name,
                        'Feature': row['Feature'],
                        'Effect_size': row['Effect_size'],
                        'Mean_difference': row['Mean_difference'],
                        'P_value': row['P_value'],
                        'P_value_FDR': row['P_value_FDR']
                    })
        
        if not large_effects:
            print(f"  No features found with effect size > {min_effect_size}")
            return self
        
        large_effects_df = pd.DataFrame(large_effects)
        
        # Create summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Count of large effect features per comparison
        effect_counts = large_effects_df['Comparison'].value_counts()
        bars = ax1.bar(range(len(effect_counts)), effect_counts.values, 
                    color=['#2E8B57', '#DC143C', '#4682B4'])
        ax1.set_xticks(range(len(effect_counts)))
        ax1.set_xticklabels([comp.replace('_', ' ') for comp in effect_counts.index], rotation=45, ha='right')
        ax1.set_ylabel('Number of Features')
        ax1.set_title(f'Features with Large Effect Size (|d| > {min_effect_size})')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, effect_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(effect_counts.values)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Effect size vs mean difference scatter
        colors = {'ALS_Untreated_vs_Control': '#DC143C', 
                'ALS_Treated_vs_Control': '#FF6347',
                'Control_Treated_vs_Control': '#4682B4'}
        
        for comp in large_effects_df['Comparison'].unique():
            comp_data = large_effects_df[large_effects_df['Comparison'] == comp]
            ax2.scatter(comp_data['Mean_difference'], comp_data['Effect_size'], 
                    c=colors.get(comp, '#808080'), label=comp.replace('_', ' '), 
                    alpha=0.7, s=60)
        
        ax2.set_xlabel('Mean Difference')
        ax2.set_ylabel('Effect Size (Cohen\'s d)')
        ax2.set_title('Effect Size vs Mean Difference')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{plot_dir}/effect_size_summary.png', dpi=AnalysisConfig.PLOT_DPI, bbox_inches='tight')
            print(f"  ✓ Saved: effect_size_summary.png")
        plt.show()
        
        # Export large effects
        filename = os.path.join(self.output_dir, f'large_effect_features_d{min_effect_size}.csv')
        large_effects_df.to_csv(filename, index=False)
        print(f"  ✓ Saved: large_effect_features_d{min_effect_size}.csv")
        print(f"  Found {len(large_effects_df)} feature-comparison pairs with large effects")
        
        return self
        
    def export_results(self):
        """Export all results to CSV files"""
        print("\nExporting results...")
        
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Export biological replicates (averaged technical replicates)
        if self.biological_replicates is not None:
            filename = os.path.join(self.output_dir, 'biological_replicates_averaged.csv')
            self.biological_replicates.to_csv(filename, index=False)
            print(f"  ✓ Saved: biological_replicates_averaged.csv")
            print(f"    - {len(self.biological_replicates)} biological replicates")
            print(f"    - {len(self.morphology_features)} morphology features")
            print(f"    - Each row = 1 patient line averaged across technical replicates (wells)")
        
        # 2. Export experimental design summary
        design_summary = pd.DataFrame({
            'Patient_Line': self.control_lines + self.mutant_lines,
            'Disease_Status': ['Control']*len(self.control_lines) + ['ALS']*len(self.mutant_lines)
        })
        filename = os.path.join(self.output_dir, 'experimental_design_summary.csv')
        design_summary.to_csv(filename, index=False)
        print(f"  ✓ Saved: experimental_design_summary.csv")
        
        # 3. Export well-level replicate counts
        if 'Metadata_Condition_Description' in self.df.columns:
            replicate_counts = self.df['Metadata_Condition_Description'].value_counts().reset_index()
            replicate_counts.columns = ['Metadata_Condition_Description', 'Well_Count']
            filename = os.path.join(self.output_dir, 'well_level_replicate_counts.csv')
            replicate_counts.to_csv(filename, index=False)
            print(f"  ✓ Saved: well_level_replicate_counts.csv")
        
        # 4. Export each statistical comparison
        for comp_name, results_df in self.results.items():
            filename = os.path.join(self.output_dir, f'{comp_name}_statistical_results.csv')
            results_df.to_csv(filename, index=False)
            sig_count = sum(results_df['Significant_FDR'])
            print(f"  ✓ Saved: {comp_name}_statistical_results.csv ({sig_count} significant features)")
        
        # 5. Export summary statistics
        summary_stats = []
        for comp_name, results_df in self.results.items():
            comp_info = self.comparisons[comp_name]
            summary_stats.append({
                'Comparison': comp_name,
                'Description': comp_info['description'],
                'Reference_Group': comp_info['reference'],
                'Comparison_Group': comp_info['comparison'],
                'Total_features_tested': len(results_df),
                'Significant_features_FDR05': sum(results_df['Significant_FDR']),
                'Percent_significant': (sum(results_df['Significant_FDR']) / len(results_df)) * 100,
                'Max_absolute_mean_difference': results_df['Mean_difference'].abs().max(),
                'Min_FDR_pvalue': results_df['P_value_FDR'].min(),
                'Features_with_large_mean_diff': sum((results_df['Significant_FDR']) & (results_df['Mean_difference'].abs() > 0.5))
            })
        
        summary_df = pd.DataFrame(summary_stats)
        filename = os.path.join(self.output_dir, 'analysis_summary_statistics.csv')
        summary_df.to_csv(filename, index=False)
        print(f"  ✓ Saved: analysis_summary_statistics.csv")
        
        # 6. Export top significant features across all comparisons
        all_significant = []
        for comp_name, results_df in self.results.items():
            sig_features = results_df[results_df['Significant_FDR']].copy()
            sig_features['Comparison'] = comp_name
            all_significant.append(sig_features)
        
        if all_significant:
            combined_sig = pd.concat(all_significant, ignore_index=True)
            filename = os.path.join(self.output_dir, 'all_significant_features_combined.csv')
            combined_sig.to_csv(filename, index=False)
            print(f"  ✓ Saved: all_significant_features_combined.csv ({len(combined_sig)} total significant results)")
        
        print(f"\n  📁 All results saved to: {self.output_dir}/")
        print(f"  🔬 Key file for downstream analysis: biological_replicates_averaged.csv")
        
        return self
    
    def run_analysis(self):
        """Run the complete simplified analysis pipeline"""
        print("=" * 70)
        print("ASTROCYTE CELL PAINTING ANALYSIS - SIMPLIFIED PIPELINE")
        print("=" * 70)
        
        (self.load_data()
         .quality_control(missing_threshold=AnalysisConfig.MISSING_THRESHOLD, remove_extreme_features=True)
         .create_biological_replicates()
         .perform_comparisons_vs_control(fdr_threshold=AnalysisConfig.FDR_THRESHOLD)
         .identify_extreme_features(threshold_multiplier=AnalysisConfig.EXTREME_FEATURE_THRESHOLD)
         .create_summary_plot()
         .create_volcano_plots()
         .create_mean_difference_heatmap(top_n=AnalysisConfig.TOP_FEATURES_FOR_HEATMAP)
         .create_effect_size_summary(min_effect_size=0.5)
         .create_top_features_plot('ALS_Untreated_vs_Control', top_n=AnalysisConfig.TOP_FEATURES_FOR_PLOTS)
         .create_single_feature_plots('ALS_Untreated_vs_Control', top_n=10)
         .create_heatmap(top_n_per_comparison=5)
         .export_results())
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        
        # Print final summary
        print("\nFINAL SUMMARY:")
        for comp_name, results_df in self.results.items():
            sig_count = sum(results_df['Significant_FDR'])
            total_count = len(results_df)
            print(f"  {comp_name}: {sig_count}/{total_count} significant features")
        
        print("=" * 70)
        
        return self


# =============================================================================
# USAGE AND MAIN EXECUTION
# =============================================================================

def main():
    """Run the simplified analysis"""
    analysis = CellPaintingAnalysis()
    analysis.run_analysis()
    return analysis


if __name__ == "__main__":
    analysis = main()