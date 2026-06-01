"""
Utility functions for credit risk modeling pipeline.

This module provides general-purpose helper functions for:
- Logging configuration
- Path management
- Plotting and visualization
- Error handling and reporting
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Any

import matplotlib.pyplot as plt
import seaborn as sns

from src.constants import LOG_LEVEL, LOG_FORMAT, PROJECT_ROOT


def setup_logging(level: str = LOG_LEVEL, 
                  log_file: Optional[Path] = None) -> logging.Logger:
    """
    Configure logging for the entire project.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to

    Returns:
        Configured logger object

    Example:
        >>> logger = setup_logging(level='INFO')
        >>> logger.info("Application started")
    """
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level))

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)

    # Add handler to logger
    if not logger.handlers:  # Prevent duplicate handlers
        logger.addHandler(console_handler)

    # Add file handler if requested
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.StreamHandler(open(log_file, 'a'))
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_directory_exists(dirpath: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        dirpath: Path object for directory

    Returns:
        Verified Path object

    Raises:
        IOError: If directory cannot be created

    Example:
        >>> output_dir = ensure_directory_exists(Path('output'))
    """
    dirpath = Path(dirpath)

    try:
        dirpath.mkdir(parents=True, exist_ok=True)
        return dirpath
    except Exception as e:
        raise IOError(f"Cannot create directory {dirpath}: {str(e)}")


def setup_plotting_style(style: str = 'whitegrid') -> None:
    """
    Configure matplotlib and seaborn for consistent visualizations.

    Args:
        style: Seaborn style (whitegrid, darkgrid, white, dark, ticks)

    Example:
        >>> setup_plotting_style(style='whitegrid')
        >>> plt.figure(figsize=(10, 6))
    """
    sns.set_style(style)
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'figure.dpi': 100,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10
    })


def print_section_header(title: str, width: int = 70) -> None:
    """
    Print a formatted section header for console output.

    Args:
        title: Section title text
        width: Header width in characters

    Example:
        >>> print_section_header("DATA QUALITY REPORT")
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection_header(title: str, width: int = 70) -> None:
    """
    Print a formatted subsection header for console output.

    Args:
        title: Subsection title text
        width: Header width in characters

    Example:
        >>> print_subsection_header("Missing Values Analysis")
    """
    print("\n" + "-" * width)
    print(title.ljust(width))
    print("-" * width)


def format_number(value: Any, decimals: int = 2) -> str:
    """
    Format a number with proper precision and separators.

    Args:
        value: Numeric value to format
        decimals: Decimal places to show

    Returns:
        Formatted string

    Example:
        >>> format_number(1234567.89, decimals=2)
        '1,234,567.89'
    """
    try:
        if isinstance(value, (int, float)):
            return f"{value:,.{decimals}f}"
        return str(value)
    except Exception:
        return str(value)


def safe_divide(numerator: float, denominator: float, 
                default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default on division by zero.

    Args:
        numerator: Dividend
        denominator: Divisor
        default: Value to return if denominator is 0

    Returns:
        Division result or default

    Example:
        >>> ratio = safe_divide(10, 0)  # Returns 0.0
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


class Timer:
    """
    Context manager for measuring execution time.

    Example:
        >>> with Timer("data processing"):
        ...     process_data(df)
    """
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        print(f"▶ Starting: {self.task_name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time

        if exc_type is not None:
            print(f"✗ Failed: {self.task_name} (Error: {exc_type.__name__})")
        else:
            print(f"✓ Completed: {self.task_name} ({elapsed:.2f}s)")

        return False


def validate_inputs(**kwargs) -> bool:
    """
    Validate that all keyword arguments are provided and not None.

    Args:
        **kwargs: Key-value pairs to validate

    Returns:
        True if all values are non-None

    Raises:
        ValueError: If any value is None

    Example:
        >>> validate_inputs(df=data, model=clf)  # Raises if either is None
    """
    for key, value in kwargs.items():
        if value is None:
            raise ValueError(f"Required parameter '{key}' is None")

    return True


def pretty_print_dict(d: dict, indent: int = 0) -> str:
    """
    Pretty-print a dictionary with nested structure support.

    Args:
        d: Dictionary to print
        indent: Initial indentation level

    Returns:
        Formatted string

    Example:
        >>> print(pretty_print_dict({'model': {'type': 'LogisticRegression'}}))
    """
    lines = []
    indent_str = "  " * indent

    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{indent_str}{key}:")
            lines.append(pretty_print_dict(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{indent_str}{key}: [")
            for item in value:
                if isinstance(item, dict):
                    lines.append(pretty_print_dict(item, indent + 1))
                else:
                    lines.append(f"{indent_str}  {item}")
            lines.append(f"{indent_str}]")
        else:
            lines.append(f"{indent_str}{key}: {value}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    setup_plotting_style()
    print_section_header("UTILITY FUNCTIONS DEMO")

    logger = setup_logging(level='INFO')
    logger.info("Logging configured successfully")

    with Timer("sample task"):
        import time
        time.sleep(0.1)

    print(f"\nFormatted number: {format_number(1234567.89)}")
    print(f"Safe division: {safe_divide(10, 2)}")
    print(f"Safe division by zero: {safe_divide(10, 0, default=999.0)}")
