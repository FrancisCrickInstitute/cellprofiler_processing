"""
Configuration loading utilities for unified config file
UPDATED: Now includes normalization parameters
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_config(config_file: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Load unified configuration from YAML file
    
    Args:
        config_file: Path to YAML config file
    
    Returns:
        dict: Configuration dictionary or None if file not found
    """
    if not config_file or not os.path.exists(config_file):
        logger.warning("No config file provided or file doesn't exist. Using defaults.")
        return None
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded unified configuration from {config_file}")
        
        # Validate config structure
        if 'plate_dict' in config:
            logger.info(f"Found plate mappings for {len(config['plate_dict'])} plates")
        
        if 'quality_control' in config:
            qc_params = config['quality_control']
            logger.info(f"Quality control thresholds: missing={qc_params.get('missing_threshold', 0.05)}, "
                       f"correlation={qc_params.get('correlation_threshold', 0.95)}")
        
        if 'normalization' in config:
            norm_params = config['normalization']
            method = norm_params.get('method', 'z_score')
            baseline = norm_params.get('baseline', 'dmso')
            logger.info(f"Normalization: {method} method with {baseline} baseline")
        
        if 'visualization' in config:
            viz_config = config['visualization']
            n_umap = len(viz_config.get('umap_parameters', []))
            n_tsne = len(viz_config.get('tsne_parameters', []))
            logger.info(f"Visualization parameters: {n_umap} UMAP sets, {n_tsne} t-SNE sets")
        
        return config
        
    except Exception as e:
        logger.warning(f"Error loading config file: {e}")
        return None


def get_quality_control_params(config: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Extract quality control parameters from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        dict: Quality control parameters with defaults
    """
    defaults = {
        'missing_threshold': 0.05,
        'correlation_threshold': 0.95,
        'high_variability_threshold': 15.0,
        'low_variability_threshold': 0.01
    }
    
    if config is None or 'quality_control' not in config:
        logger.info("Using default quality control parameters")
        return defaults
    
    qc_config = config['quality_control']
    
    # Update defaults with config values
    params = defaults.copy()
    for key in defaults.keys():
        if key in qc_config:
            params[key] = float(qc_config[key])
    
    logger.info(f"Quality control parameters: {params}")
    return params


def get_normalization_params(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract normalization parameters from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        dict: Normalization parameters with defaults
    """
    defaults = {
        'method': 'z_score',
        'normalization_type': 'control_based',
        'control_compound': 'DMSO',
        'control_column': 'Metadata_perturbation_name'
    }
    
    if config is None or 'normalization' not in config:
        logger.info("Using default normalization parameters")
        return defaults
    
    norm_config = config['normalization']
    
    # Update defaults with config values
    params = defaults.copy()
    for key in defaults.keys():
        if key in norm_config:
            params[key] = norm_config[key]
    
    # Validate method
    valid_methods = ['z_score', 'robust_mad']
    if params['method'] not in valid_methods:
        logger.warning(f"Invalid normalization method '{params['method']}', using 'z_score'")
        params['method'] = 'z_score'
    
    # Validate normalization_type
    valid_normalization_types = ['control_based', 'all_conditions']
    if params.get('normalization_type', 'control_based') not in valid_normalization_types:
        logger.warning(f"Invalid normalization_type '{params.get('normalization_type')}', using 'control_based'")
        params['normalization_type'] = 'control_based'
    
    logger.info(f"Normalization parameters: {params}")
    return params


def get_plate_dict(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract plate dictionary from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        dict: Plate mapping dictionary
    """
    if config is None:
        return {}
    
    return config.get('plate_dict', {})


def get_viz_parameters(config: Optional[Dict[str, Any]]) -> Tuple[list, list]:
    """
    Extract UMAP and t-SNE parameters from unified config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        tuple: (umap_parameters, tsne_parameters)
    """
    if config is None or 'visualization' not in config:
        # Default parameter sets
        logger.info("Using default visualization parameters")
        default_umap = [{'name': 'default', 'n_neighbors': 15, 'min_dist': 0.1}]
        default_tsne = [{'name': 'default', 'perplexity': 30}]
        return default_umap, default_tsne
    
    viz_config = config['visualization']
    umap_params = viz_config.get('umap_parameters', [])
    tsne_params = viz_config.get('tsne_parameters', [])
    
    # Ensure we have at least one parameter set
    if not umap_params:
        umap_params = [{'name': 'default', 'n_neighbors': 15, 'min_dist': 0.1}]
    if not tsne_params:
        tsne_params = [{'name': 'default', 'perplexity': 30}]
    
    return umap_params, tsne_params


# Deprecated functions for backward compatibility
def load_viz_config(viz_config_file: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    DEPRECATED: Use load_config() instead.
    Load visualization parameters from YAML file
    
    This function is kept for backward compatibility but will be removed.
    """
    logger.warning("load_viz_config() is deprecated. Use load_config() with unified config file.")
    
    if not viz_config_file or not os.path.exists(viz_config_file):
        logger.info("No separate visualization config file. Using unified config.")
        return None
    
    try:
        with open(viz_config_file, 'r') as f:
            viz_config = yaml.safe_load(f)
        
        n_umap = len(viz_config.get('umap_parameters', []))
        n_tsne = len(viz_config.get('tsne_parameters', []))
        logger.info(f"Loaded separate visualization config with {n_umap} UMAP and {n_tsne} t-SNE parameter sets")
        
        return viz_config
        
    except Exception as e:
        logger.warning(f"Error loading visualization config: {e}")
        return None

def get_input_files(config: Optional[Dict[str, Any]]) -> List[str]:
    """
    Extract input file paths from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        list: List of input file paths
    """
    if config is None:
        logger.error("No configuration found - cannot determine input files")
        return []
    
    # Check for input_files in config
    if 'input_files' in config:
        input_files = config['input_files']
        if isinstance(input_files, list):
            logger.info(f"Found {len(input_files)} input files in config")
            return input_files
        else:
            logger.warning(f"input_files should be a list, got {type(input_files)}")
            return [input_files] if input_files else []
    
    # Fallback to single input_file for backward compatibility
    if 'input_file' in config:
        logger.info("Found single input_file in config (using as list)")
        return [config['input_file']]
    
    logger.error("No input_files or input_file found in configuration")
    return []

def get_metadata_file(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Extract metadata file path from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        str: Path to metadata file, or None if not found
    """
    if config is None:
        logger.warning("No configuration found - cannot determine metadata file")
        return None
    
    # Check for metadata_file in config
    if 'metadata_file' in config:
        metadata_file = config['metadata_file']
        logger.info(f"Found metadata file in config: {metadata_file}")
        return metadata_file
    
    logger.warning("No metadata_file found in configuration")
    return None


def get_analysis_flags(config: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Extract analysis flags from configuration (handles both nested and top-level)
    
    Args:
        config: Configuration dictionary
    
    Returns:
        dict: Analysis flags with defaults
    """
    defaults = {
        'run_landmark_analysis': False,
        'run_hierarchical_clustering': False,
        'run_landmark_threshold_analysis': False
    }
    
    if config is None:
        logger.info("No config provided, using default analysis flags (all False)")
        return defaults
    
    flags = defaults.copy()
    
    # Check nested 'analysis' section first (preferred structure)
    if 'analysis' in config and isinstance(config['analysis'], dict):
        analysis_config = config['analysis']
        for key in defaults.keys():
            if key in analysis_config:
                flags[key] = bool(analysis_config[key])
                logger.info(f"Found {key} in analysis section: {flags[key]}")
    
    # Check top-level as fallback (backward compatibility)
    for key in defaults.keys():
        if key in config:
            flags[key] = bool(config[key])
            logger.info(f"Found {key} at top level: {flags[key]}")
    
    logger.info(f"Analysis flags: {flags}")
    return flags


def get_visualization_flags(config: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Extract visualization control flags from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        dict: Visualization flags with defaults
    """
    defaults = {
        'skip_embedding_generation': False,
        'reuse_existing_coordinates': False  # Alternative name, same meaning
    }
    
    if config is None:
        logger.info("No config provided, using default visualization flags")
        return defaults
    
    flags = defaults.copy()
    
    # Check nested 'visualization' section first (preferred structure)
    if 'visualization' in config and isinstance(config['visualization'], dict):
        viz_config = config['visualization']
        for key in defaults.keys():
            if key in viz_config:
                flags[key] = bool(viz_config[key])
                logger.info(f"Found {key} in visualization section: {flags[key]}")
    
    # Check top-level as fallback (backward compatibility)
    for key in defaults.keys():
        if key in config:
            flags[key] = bool(config[key])
            logger.info(f"Found {key} at top level: {flags[key]}")
    
    # Handle either flag name (they mean the same thing)
    if flags['reuse_existing_coordinates']:
        flags['skip_embedding_generation'] = True
    
    logger.info(f"Visualization flags: skip_embedding_generation={flags['skip_embedding_generation']}")
    return flags