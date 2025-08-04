#!/usr/bin/env python3
"""
YOLO vs SAM Comparison Script
Runs both YOLO and SAM analysis on the same video and compares results.
"""

import os
import sys
import subprocess
import time
import pandas as pd
from datetime import datetime
import argparse

def run_yolo_analysis(video_path: str, detailed_info: bool = True, max_frames: int = None) -> dict:
    """
    Run YOLO analysis and return results.
    
    Args:
        video_path: Path to video file
        detailed_info: Whether to show detailed frame info
        max_frames: Maximum frames to process
        
    Returns:
        Dictionary with analysis results and metadata
    """
    # Running YOLO Analysis - logging removed for cleaner output
    
    # Build command
    cmd = [
        "python", "run_analysis.py",
        "--video", video_path,
        "--frame-logic",
        "--confidence", "0.25",
        "--iou", "0.45",
        "--imgsz", "1280"
    ]
    
    if detailed_info:
        cmd.append("--detailed-info")
    
    if max_frames:
        cmd.extend(["--max-frames", str(max_frames)])
    
    # Run analysis
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        processing_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"❌ YOLO analysis failed: {result.stderr}")
            return None
            
        print(f"✅ YOLO analysis completed in {processing_time:.1f}s")
        
        # Parse results from latest log files
        return parse_analysis_results("yolo", processing_time, result.stdout)
        
    except subprocess.TimeoutExpired:
        print("⏰ YOLO analysis timed out")
        return None
    except Exception as e:
        print(f"❌ YOLO analysis error: {e}")
        return None

def run_sam_analysis(video_path: str, sam_model: str = "vit_b", detailed_info: bool = True, max_frames: int = None) -> dict:
    """
    Run SAM analysis and return results.
    
    Args:
        video_path: Path to video file
        sam_model: SAM model type
        detailed_info: Whether to show detailed frame info
        max_frames: Maximum frames to process
        
    Returns:
        Dictionary with analysis results and metadata
    """
    # Running SAM Analysis - logging removed for cleaner output
    
    # Build command
    cmd = [
        "python", "run_sam_analysis.py",
        "--video", video_path,
        "--sam-model", sam_model,
        "--confidence", "0.25",
        "--iou", "0.45",
        "--imgsz", "1280"
    ]
    
    if detailed_info:
        cmd.append("--detailed-info")
    
    if max_frames:
        cmd.extend(["--max-frames", str(max_frames)])
    
    # Run analysis
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 60 min timeout
        processing_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"❌ SAM analysis failed: {result.stderr}")
            return None
            
        print(f"✅ SAM analysis completed in {processing_time:.1f}s")
        
        # Parse results from latest log files
        return parse_analysis_results("sam", processing_time, result.stdout)
        
    except subprocess.TimeoutExpired:
        print("⏰ SAM analysis timed out")
        return None
    except Exception as e:
        print(f"❌ SAM analysis error: {e}")
        return None

def parse_analysis_results(analysis_type: str, processing_time: float, stdout: str) -> dict:
    """
    Parse analysis results from log files and stdout.
    
    Args:
        analysis_type: "yolo" or "sam"
        processing_time: Processing time in seconds
        stdout: Standard output from analysis
        
    Returns:
        Dictionary with parsed results
    """
    results = {
        'analysis_type': analysis_type,
        'processing_time': processing_time,
        'timestamp': datetime.now().isoformat()
    }
    
    # Find latest results CSV file
    try:
        import glob
        if analysis_type == "yolo":
            pattern = "../logs/*_results_*.csv"
        else:
            pattern = "../logs/*_sam_results_*.csv"
            
        csv_files = glob.glob(pattern)
        if csv_files:
            latest_csv = max(csv_files, key=os.path.getctime)
            
            # Read CSV results
            df = pd.read_csv(latest_csv)
            
            # Extract class counts
            for _, row in df.iterrows():
                class_name = row.get('class', 'unknown')
                results[class_name] = {
                    'safe': row.get('safe_crossings', 0),
                    'uncertain': row.get('uncertain_crossings', 0),
                    'very_brief': row.get('very_brief_crossings', 0),
                    'discarded': row.get('discarded_crossings', 0),
                    'total': row.get('total_valid_crossings', 0)
                }
    
    except Exception as e:
        print(f"⚠️ Failed to parse {analysis_type} results: {e}")
    
    # Parse processing speed from stdout
    try:
        lines = stdout.split('\n')
        for line in lines:
            if 'Average FPS' in line or 'Speed:' in line:
                # Extract FPS value
                import re
                fps_match = re.search(r'(\d+\.?\d*)\s*FPS', line)
                if fps_match:
                    results['processing_fps'] = float(fps_match.group(1))
    except:
        pass
    
    return results

def create_comparison_report(yolo_results: dict, sam_results: dict, video_path: str, output_path: str):
    """
    Create detailed comparison report.
    
    Args:
        yolo_results: YOLO analysis results
        sam_results: SAM analysis results
        video_path: Path to analyzed video
        output_path: Path to save report
    """
    
    report = []
    report.append("# YOLO vs SAM Analysis Comparison Report")
    report.append("=" * 60)
    report.append(f"**Video:** {os.path.basename(video_path)}")
    report.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Processing Performance
    report.append("## ⚡ Processing Performance")
    report.append("")
    yolo_time = yolo_results.get('processing_time', 0)
    sam_time = sam_results.get('processing_time', 0)
    yolo_fps = yolo_results.get('processing_fps', 0)
    sam_fps = sam_results.get('processing_fps', 0)
    
    report.append(f"| Metric | YOLO | SAM | Difference |")
    report.append(f"|--------|------|-----|------------|")
    report.append(f"| Processing Time | {yolo_time:.1f}s | {sam_time:.1f}s | {sam_time-yolo_time:+.1f}s |")
    report.append(f"| Processing Speed | {yolo_fps:.1f} FPS | {sam_fps:.1f} FPS | {sam_fps-yolo_fps:+.1f} FPS |")
    
    speed_ratio = yolo_fps / sam_fps if sam_fps > 0 else 0
    report.append(f"| Speed Ratio | YOLO is **{speed_ratio:.1f}x** faster | | |")
    report.append("")
    
    # Detection Accuracy Comparison
    report.append("## 🎯 Detection Accuracy Comparison")
    report.append("")
    
    classes = ['person', 'backpack', 'handbag', 'suitcase']
    
    report.append("| Class | YOLO Total | SAM Total | Difference | Improvement |")
    report.append("|-------|------------|-----------|------------|-------------|")
    
    total_yolo = 0
    total_sam = 0
    
    for cls in classes:
        yolo_count = yolo_results.get(cls, {}).get('total', 0)
        sam_count = sam_results.get(cls, {}).get('total', 0)
        
        total_yolo += yolo_count
        total_sam += sam_count
        
        difference = sam_count - yolo_count
        improvement = (difference / yolo_count * 100) if yolo_count > 0 else 0
        
        improvement_str = f"{improvement:+.1f}%" if improvement != 0 else "0%"
        
        report.append(f"| {cls.capitalize()} | {yolo_count} | {sam_count} | {difference:+d} | {improvement_str} |")
    
    overall_improvement = ((total_sam - total_yolo) / total_yolo * 100) if total_yolo > 0 else 0
    report.append(f"| **TOTAL** | **{total_yolo}** | **{total_sam}** | **{total_sam-total_yolo:+d}** | **{overall_improvement:+.1f}%** |")
    report.append("")
    
    # Detailed Breakdown
    report.append("## 📊 Detailed Quality Breakdown")
    report.append("")
    
    for cls in classes:
        yolo_data = yolo_results.get(cls, {})
        sam_data = sam_results.get(cls, {})
        
        if yolo_data.get('total', 0) > 0 or sam_data.get('total', 0) > 0:
            report.append(f"### {cls.capitalize()}")
            report.append("")
            report.append("| Quality Level | YOLO | SAM | Difference |")
            report.append("|---------------|------|-----|------------|")
            
            for quality in ['safe', 'uncertain', 'very_brief', 'discarded']:
                yolo_val = yolo_data.get(quality, 0)
                sam_val = sam_data.get(quality, 0)
                diff = sam_val - yolo_val
                
                report.append(f"| {quality.capitalize()} | {yolo_val} | {sam_val} | {diff:+d} |")
            
            report.append("")
    
    # Summary and Recommendations
    report.append("## 💡 Summary & Recommendations")
    report.append("")
    
    if overall_improvement > 5:
        report.append("✅ **SAM shows significant improvement** in detection accuracy")
    elif overall_improvement > 0:
        report.append("🟡 **SAM shows slight improvement** in detection accuracy")
    else:
        report.append("🔴 **YOLO performs better** in this case")
    
    report.append("")
    
    if speed_ratio > 3:
        report.append(f"⚡ **YOLO is significantly faster** ({speed_ratio:.1f}x) - recommended for real-time applications")
    elif speed_ratio > 1.5:
        report.append(f"🏃 **YOLO is faster** ({speed_ratio:.1f}x) - good balance of speed and accuracy")
    else:
        report.append("🐌 **Processing speeds are comparable** - choose based on accuracy needs")
    
    report.append("")
    report.append("### Recommendations:")
    report.append("")
    
    if overall_improvement > 10 and speed_ratio < 5:
        report.append("- **Use SAM** for highest accuracy requirements")
        report.append("- **Use YOLO** for speed-critical applications")
    elif overall_improvement > 5:
        report.append("- **SAM recommended** for better accuracy")
        report.append("- Consider SAM if processing time is acceptable")
    else:
        report.append("- **YOLO recommended** for better speed with comparable accuracy")
        report.append("- SAM may not provide sufficient improvement to justify slower processing")
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"📊 Comparison report saved: {output_path}")
    
    # Also print summary to console
    print("\n" + "=" * 60)
    print("🏆 COMPARISON SUMMARY")
    print("=" * 60)
    print(f"📈 Detection Improvement: {overall_improvement:+.1f}%")
    print(f"⚡ Speed Difference: YOLO is {speed_ratio:.1f}x faster")
    print(f"📊 Report saved: {output_path}")

def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare YOLO vs SAM analysis")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--sam-model", type=str, default="vit_b", 
                       choices=["vit_b", "vit_l", "vit_h"], help="SAM model type")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process (for testing)")
    parser.add_argument("--detailed-info", action="store_true", 
                       help="Show detailed frame information")
    parser.add_argument("--output", type=str, 
                       default=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                       help="Output report filename")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    print("🎬 YOLO vs SAM Video Analysis Comparison")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"SAM Model: {args.sam_model}")
    if args.max_frames:
        print(f"Max Frames: {args.max_frames}")
    print("=" * 60)
    
    # Run YOLO analysis
    yolo_results = run_yolo_analysis(args.video, args.detailed_info, args.max_frames)
    if not yolo_results:
        print("❌ YOLO analysis failed, cannot proceed with comparison")
        return
    
    print("\n" + "=" * 60)
    
    # Run SAM analysis  
    sam_results = run_sam_analysis(args.video, args.sam_model, args.detailed_info, args.max_frames)
    if not sam_results:
        print("❌ SAM analysis failed, cannot proceed with comparison")
        return
    
    print("\n" + "=" * 60)
    
    # Create comparison report
    create_comparison_report(yolo_results, sam_results, args.video, args.output)

if __name__ == "__main__":
    main()