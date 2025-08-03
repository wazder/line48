# Analysis Tools

This directory contains various analysis and debugging scripts used during the LineLogic project development.

## Scripts Overview

### Performance Analysis
- **`compare_to_ground_truth.py`** - Compare model results against manual ground truth
- **`compare_results.py`** - Compare different model runs and parameter changes
- **`simple_analysis.py`** - Basic analysis of detection counts and totals

### Debugging Tools
- **`debug_comparison.py`** - Detailed debugging of comparison logic
- **`check_confidence_levels.py`** - Analyze confidence distributions in logs
- **`check_overlaps.py`** - Check for overlapping intervals in ground truth data

### Detailed Analysis
- **`analyze_misses.py`** - Analyze patterns in missed detections
- **`detailed_analysis.py`** - Comprehensive object tracking analysis

## Usage

Most scripts can be run directly with Python:

```bash
python analysis_tools/compare_to_ground_truth.py
python analysis_tools/compare_results.py
```

## Key Findings

- **Handbag detection** is the main challenge (44% accuracy)
- **Frame logic thresholds** are more important than model parameters
- **Class-specific thresholds** could improve performance significantly

## Recommendations

1. Lower frame thresholds for handbags and backpacks
2. Count "very_brief" detections for these classes
3. Use class-specific confidence thresholds
4. Consider model fine-tuning for handbag detection 