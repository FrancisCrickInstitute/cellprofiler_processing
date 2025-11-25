#!/usr/bin/env python3
"""
Main entry point for Enhanced Cell Profiler Data Analyser - Now with post-landmark-full-dist mode
"""

import sys
import argparse
from pathlib import Path
import datetime

# Add parent directory to path to allow imports from cellprofiler_analyser package
sys.path.insert(0, str(Path(__file__).parent.parent))

from cellprofiler_analyser.core.processor import EnhancedCellPaintingProcessor
from cellprofiler_analyser.core.visualization import DataVisualizer
from cellprofiler_analyser.utils.logging_utils import setup_logging

def run_replot_only(output_dir: str, use_pca: bool = True, verbose: bool = False) -> bool:
    """
    Run visualization replotting only using optimized coordinate-based approach
    Regenerates UMAP/t-SNE HTML plots from existing coordinate files
    
    Args:
        output_dir: Output directory containing existing coordinate files
        use_pca: Whether to use PCA model from analysis directory
        verbose: Enable verbose logging
        
    Returns:
        bool: Success status
    """
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(level=log_level)
    
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"ERROR: Output directory does not exist: {output_path}")
        return False
    
    # Check for coordinate file
    coords_dir = output_path / "visualizations" / "coordinates"
    coord_file = coords_dir / "embedding_coordinates.csv"
    
    if not coord_file.exists():
        print(f"ERROR: No coordinate file found: {coord_file}")
        print("Expected file:")
        print(f"  - {coord_file}")
        return False
    
    # Initialize visualizer
    visualizer = DataVisualizer(output_path)
    
    print(f"Found coordinate file: {coord_file}")
    print(f"Recreating UMAP/t-SNE plots and hierarchical clustering...")
    
    # Recreate plots using optimized coordinate-based approach (now includes clustering)
    success = visualizer.recreate_plots_from_coordinates(use_pca_from_analysis=use_pca)
    
    if success:
        print("Plot recreation completed successfully!")
        print(f"New optimized plots saved to: {output_path}/visualizations_redo/")
        print("Check the following directories:")
        print(f"  - UMAP plots: {output_path}/visualizations_redo/umap/")
        print(f"  - t-SNE plots: {output_path}/visualizations_redo/tsne/")
        print("")
        print("KEY OPTIMIZATIONS APPLIED:")
        print("  - No legends displayed (major file size reduction)")
        print("  - Essential metadata hover only")
        print("  - Plots regenerated from saved coordinates")
    else:
        print("ERROR: Plot recreation failed!")
    
    return success


def main():
    """Main function that handles command line arguments"""
    
    parser = argparse.ArgumentParser(description='Enhanced Cell Profiler data processing with Z-score normalization')
    
    # PRIMARY MODE CONTROL
    parser.add_argument('--start-from', 
                choices=['full', 'well', 'replot'],
                default='full',
                help='Pipeline entry point: full (complete analysis), well (from well-level data), or replot (regenerate plots from coordinates)')
    
    parser.add_argument('--previous-run-dir',
                    type=str,
                    help='Base directory of previous run (for skip modes)')
    
    # INPUT/OUTPUT
    parser.add_argument('--input', '-i', nargs='+', 
                    help='OPTIONAL: Override input parquet file(s) from config')
    parser.add_argument('--metadata', '-m', help='Metadata CSV file (for full mode)')
    parser.add_argument('--config', '-c', help='Unified config YAML file (required)')
    parser.add_argument('--output', '-o', help='Base output directory (timestamp will be added automatically)')
    parser.add_argument('--no-timestamp', action='store_true', 
                       help='Disable automatic timestamp in output directory name')
    
    # THRESHOLD OVERRIDES (optional)
    parser.add_argument('--missing-threshold', type=float, help='Override missing threshold from config')
    parser.add_argument('--correlation-threshold', type=float, help='Override correlation threshold from config')
    parser.add_argument('--high-var-threshold', type=float, help='Override high variability threshold from config')
    parser.add_argument('--low-var-threshold', type=float, help='Override low variability threshold from config')
    
    # NORMALIZATION OPTIONS
    parser.add_argument('--normalization-baseline', choices=['dmso', 'all-conditions'], default=None,
                       help='Z-score normalization baseline: "dmso" (DMSO controls) or "all-conditions" (all treatments)')
    
    # ANALYSIS FLAGS (only apply in 'full' mode - for backward compatibility)
    parser.add_argument('--run-landmark-analysis', action='store_true',
                   help='Run landmark analysis (in full mode, or use --start-from landmark)')
    parser.add_argument('--run-landmark-threshold-analysis', action='store_true',
                   help='Run landmark threshold analysis')
    parser.add_argument('--skip-landmark-threshold-analysis', action='store_true',
                   help='Skip landmark threshold analysis even if config says to run it')
    parser.add_argument('--run-hierarchical-clustering', action='store_true',
                    help='Run hierarchical clustering analysis')
    
    # MISC OPTIONS
    parser.add_argument('--no-pca', action='store_true',
                       help='Do not load PCA model from analysis directory (replot mode)')

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()

    # Load config file first (needed for various settings)
    config_file = args.config
    config_obj = None
    
    if config_file:
        from cellprofiler_analyser.io.config_loader import load_config
        config_obj = load_config(config_file)

    # Determine output directory
    # Priority: 1) command line, 2) config file, 3) default
    if args.output:
        base_output = args.output
    elif config_obj and 'output_base_dir' in config_obj:
        base_output = config_obj['output_base_dir']
        print(f"Using output directory from config: {base_output}")
    else:
        base_output = './processed_data'

    if not args.no_timestamp and args.start_from == 'full':
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"{base_output}/{timestamp}_results"
    else:
        output_dir = base_output
    print(f"Output directory: {output_dir}")

    # Handle replot mode separately (doesn't need processor)
    if args.start_from == 'replot':
        print("REPLOT MODE: Recreating optimized visualizations from existing coordinates")
        print("=" * 80)
        
        success = run_replot_only(
            output_dir=output_dir,
            use_pca=not args.no_pca,
            verbose=args.verbose
        )
        
        if not success:
            sys.exit(1)
        return

    # Get input files (for full mode)
    input_files = None
    metadata_file = args.metadata
    
    if args.start_from == 'full':
        # Need input files for full mode
        if args.input:
            input_files = args.input
            print(f"Using input files from command line: {len(input_files)} files")
        elif config_obj:
            from cellprofiler_analyser.io.config_loader import get_input_files
            input_files = get_input_files(config_obj)
            if input_files:
                print(f"Using input files from config: {len(input_files)} files")
            else:
                print("ERROR: No input files provided via command line or config")
                print("Please either:")
                print("  - Use --input to specify files on command line")
                print("  - Add 'input_files' section to your config YAML")
                sys.exit(1)
        else:
            print("ERROR: No input files or config provided for full mode")
            sys.exit(1)

    # Create processor (needed for all non-replot modes)
    processor = EnhancedCellPaintingProcessor(
        input_file=input_files if input_files else [],
        metadata_file=metadata_file,
        config_file=config_file,
        output_dir=output_dir
    )

    print(f"DEBUG: Processor created successfully")
    print(f"DEBUG: args.start_from = {args.start_from}")

    # Check for threshold overrides
    override_any = any([
        args.missing_threshold is not None,
        args.correlation_threshold is not None,
        args.high_var_threshold is not None,
        args.low_var_threshold is not None
    ])

    if override_any:
        print("Command-line threshold overrides:")
        processor.set_quality_thresholds(
            missing_threshold=args.missing_threshold,
            correlation_threshold=args.correlation_threshold,
            high_variability_threshold=args.high_var_threshold,
            low_variability_threshold=args.low_var_threshold
        )

    # Get analysis flags from config
    from cellprofiler_analyser.io.config_loader import get_analysis_flags
    config_flags = get_analysis_flags(processor.config)
    
    # Determine analysis settings (for full mode)
    if args.run_landmark_analysis:
        run_landmark = True
        print("OVERRIDE: Running landmark analysis (from CLI flag)")
    else:
        run_landmark = config_flags['run_landmark_analysis']
        if run_landmark:
            print(f"Running landmark analysis from config: {run_landmark}")

    if args.run_landmark_threshold_analysis:
        run_threshold_analysis = True
        print("OVERRIDE: Running landmark threshold analysis (from CLI flag)")
    elif args.skip_landmark_threshold_analysis:
        run_threshold_analysis = False
        print("OVERRIDE: Skipping landmark threshold analysis (from CLI flag)")
    else:
        run_threshold_analysis = config_flags['run_landmark_threshold_analysis']
        if run_threshold_analysis:
            print(f"Running landmark threshold analysis from config: {run_threshold_analysis}")

    if args.run_hierarchical_clustering:
        run_clustering = True
        print("OVERRIDE: Running hierarchical clustering (from CLI flag)")
    else:
        run_clustering = config_flags['run_hierarchical_clustering']
        if run_clustering:
            print(f"Running hierarchical clustering from config: {run_clustering}")
    
    # Determine normalization baseline
    use_all_conditions = False
    if args.normalization_baseline == 'all-conditions':
        use_all_conditions = True
    elif args.normalization_baseline == 'dmso':
        use_all_conditions = False
    elif processor.config:
        from cellprofiler_analyser.io.config_loader import get_normalization_params
        norm_params = get_normalization_params(processor.config)
        use_all_conditions = (norm_params.get('normalization_type', 'control_based') == 'all_conditions')

    # Run the pipeline with the appropriate start point
    print(f"DEBUG: Calling run_full_pipeline with start_from={args.start_from}")
    
    success = processor.run_full_pipeline(
        use_all_conditions_baseline=use_all_conditions,
        start_from=args.start_from,
        previous_run_dir=args.previous_run_dir
    )
    
    print(f"DEBUG: run_full_pipeline returned {success}")

    if success:
            print(f"\nProcessing completed! Results in: {output_dir}")
            if args.start_from == 'well':
                print("Started from well-level data - skipped early processing stages")
            print("\nCheck comprehensive_summary.txt for detailed results")
    else:
        print("ERROR: Processing failed. Check the logs for details.")
        sys.exit(1)

    

if __name__ == "__main__":
    main()