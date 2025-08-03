"""
Results export utilities for LineLogic
Handles CSV export of analysis results and parameters.
"""

import csv
from datetime import datetime


def export_results_csv(filepath, parameters, results_summary):
    """Export comprehensive results to CSV."""
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow(["LineLogic Analysis Results"])
        writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow([])
        
        # Write parameters section
        writer.writerow(["ANALYSIS PARAMETERS"])
        writer.writerow(["Parameter", "Value", "Description"])
        
        param_descriptions = {
            'video_path': 'Full path to source video',
            'video_name': 'Name of source video file',
            'total_frames': 'Total number of frames in video',
            'fps': 'Video frame rate (frames per second)',
            'confidence_threshold': 'YOLO confidence threshold (0.0-1.0)',
            'iou_threshold': 'NMS IoU threshold (0.0-1.0)',
            'image_size': 'Input image size for YOLO model',
            'use_frame_logic': 'Whether frame-based validation was used',
            'min_safe_time': 'Minimum time for safe predictions (seconds)',
            'min_uncertain_time': 'Minimum time for uncertain predictions (seconds)',
            'min_very_brief_time': 'Minimum time for very brief predictions (seconds)',
            'max_frames': 'Maximum frames to process (None = all frames)',
            'line_base_x': 'X position of the first detection line',
            'line_spacing': 'Distance between consecutive detection lines',
            'line_height': 'Y position where all detection lines are placed',
            'line_positions': 'List of X positions for all 4 detection lines',
            'timestamp': 'Analysis timestamp'
        }
        
        for param, value in parameters.items():
            description = param_descriptions.get(param, 'Parameter value')
            writer.writerow([param, value, description])
        
        writer.writerow([])  # Empty row
        
        # Write results section
        writer.writerow(["DETECTION RESULTS"])
        if parameters.get('use_frame_logic'):
            writer.writerow(["Class", "Safe", "Uncertain", "Very Brief", "Total", "Success Rate (%)"])
            total_detections = 0
            for class_name, counts in results_summary.items():
                safe = counts.get("safe", 0)
                uncertain = counts.get("uncertain", 0)
                very_brief = counts.get("very_brief", 0)
                total = counts.get("total", 0)
                total_detections += total
                
                # Calculate success rate (safe + uncertain as successful)
                success_rate = ((safe + uncertain) / total * 100) if total > 0 else 0
                
                writer.writerow([
                    class_name,
                    safe,
                    uncertain,
                    very_brief,
                    total,
                    f"{success_rate:.1f}%"
                ])
            
            # Add summary row
            writer.writerow([])
            writer.writerow(["SUMMARY", "", "", "", f"Total: {total_detections}", ""])
            
        else:
            writer.writerow(["Class", "IN", "OUT", "Total", "Direction Balance"])
            total_detections = 0
            for class_name, counts in results_summary.items():
                in_count = counts.get("in", 0)
                out_count = counts.get("out", 0)
                total = counts.get("total", 0)
                total_detections += total
                
                # Calculate direction balance
                balance = f"{in_count}:{out_count}" if total > 0 else "0:0"
                
                writer.writerow([
                    class_name,
                    in_count,
                    out_count,
                    total,
                    balance
                ])
            
            # Add summary row
            writer.writerow([])
            writer.writerow(["SUMMARY", "", "", f"Total: {total_detections}", ""])
        
        writer.writerow([])  # Empty row
        
        # Write analysis notes
        writer.writerow(["ANALYSIS NOTES"])
        writer.writerow(["Note", "Description"])
        writer.writerow(["Model", f"YOLO model used for detection"])
        writer.writerow(["Logic", f"{'Frame-based validation' if parameters.get('use_frame_logic') else 'Basic tracking'} used"])
        writer.writerow(["Lines", "4 virtual lines for crossing detection"])
        writer.writerow(["Classes", "person, backpack, handbag, suitcase"])
        writer.writerow(["Performance", "Results may vary based on video quality and object visibility"]) 