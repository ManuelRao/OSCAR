"""
Demo: Difference Map (diff_map)

This script demonstrates how to create a difference map between consecutive frames
from a video stream or camera. The difference map shows areas of change/motion.

Press 'q' to quit, 's' to save the current diff_map, 'd' to toggle debug views.
"""

import cv2 as cv
import numpy as np
import time
from src import math_func as mf


def main():
    # Open camera (change index if needed: 0, 1, 2, etc.)
    camera_index = 0
    cap = cv.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        print("Try changing camera_index in the script (0, 1, 2, ...)")
        return
    
    print(f"Camera {camera_index} opened successfully!")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current diff_map as image")
    print("  'd' - Toggle debug views (colored diff, overlay)")
    print("  'c' - Clear saved frame (reset diff_map)")
    print()
    
    prev_gray = None
    show_debug = True
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        
        # Convert to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Create diff_map if we have a previous frame
        if prev_gray is not None:
            # Calculate difference map
            diff_map = mf.picture_diference(gray, prev_gray)
            
            # Show the difference map (grayscale)
            diff_map = cv.normalize(diff_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            diff_map = cv.equalizeHist(diff_map)
            diff_map = cv.GaussianBlur(diff_map, (15, 15), 2)
            cv.imshow("Diff Map (Grayscale)", diff_map)
            
            # Optional: colored visualization for better visibility
            if show_debug:
                colored_diff = cv.applyColorMap(diff_map, cv.COLORMAP_JET)
                cv.imshow("Diff Map (Colored - JET)", colored_diff)
                
                # Overlay diff on original frame
                overlay = cv.addWeighted(frame, 0.3, colored_diff, 0.7, 0)
                cv.imshow("Overlay (Frame + Diff)", overlay)
            
            # Display statistics
            mean_diff = np.mean(diff_map)
            max_diff = np.max(diff_map)
            
            # Add text info to frame
            info_frame = frame.copy()
            cv.putText(info_frame, f"Frame: {frame_count}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(info_frame, f"Mean change: {mean_diff:.2f}", (10, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(info_frame, f"Max change: {max_diff:.2f}", (10, 90), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(info_frame, f"Debug: {'ON' if show_debug else 'OFF'} (press 'd')", (10, 120), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv.imshow("Original Frame", info_frame)
        else:
            # First frame - just show it
            cv.putText(frame, "Capturing first frame...", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.imshow("Original Frame", frame)
        
        # Store current frame as previous for next iteration
        prev_gray = gray.copy()
        
        # Handle keyboard input
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            if prev_gray is not None and 'diff_map' in locals():
                filename = f"diff_map_{saved_count:03d}.png"
                cv.imwrite(filename, diff_map)
                print(f"Saved: {filename}")
                saved_count += 1
            else:
                print("No diff_map to save yet (need 2+ frames)")
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug views: {'ON' if show_debug else 'OFF'}")
            if not show_debug:
                cv.destroyWindow("Diff Map (Colored - JET)")
                cv.destroyWindow("Overlay (Frame + Diff)")
        elif key == ord('c'):
            prev_gray = None
            print("Cleared previous frame - diff_map will reset")
    
    # Cleanup
    cap.release()
    cv.destroyAllWindows()
    print(f"Processed {frame_count} frames")
    print(f"Saved {saved_count} diff_maps")


if __name__ == "__main__":
    print("=" * 60)
    print("Diff Map Demo")
    print("=" * 60)
    main()
