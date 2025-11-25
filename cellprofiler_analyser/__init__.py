"""
Enhanced Cell Profiler Data Analyser

A comprehensive toolkit for processing Cell Profiler output following Broad Institute
methodology with DMSO normalization, RobustMAD scaling, and interactive visualizations.

Example usage:
    from cellprofiler-analyser import EnhancedCellPaintingProcessor
    
    processor = EnhancedCellPaintingProcessor(
        input_file="Image.parquet",
        metadata_file="metadata.csv",
        output_dir="./processed_data"
    )
    processor.run_full_pipeline()
"""

__version__ = "4.0.0"
__author__ = "Cell Painting Analysis Team"

# Import main classes for easy access
from .core.processor import EnhancedCellPaintingProcessor
from .core.data_loader import DataLoader
from .core.feature_selection import FeatureSelector
from .core.normalization import DataNormalizer
from .core.visualization import DataVisualizer

__all__ = [
    "EnhancedCellPaintingProcessor",
    "DataLoader",
    "FeatureSelector", 
    "DataNormalizer",
    "DataVisualizer"
]