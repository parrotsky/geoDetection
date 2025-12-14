#!/usr/bin/env python3
"""
Traditional Computer Vision approach to detect circular soil pit falls (karez features)
in aerial/satellite images using OpenCV techniques.

Detection methods:
1. Hough Circle Transform
2. Contour-based detection with circularity filtering
3. Edge detection + morphological operations
4. Template matching (optional)

Requirements:
    pip install opencv-python numpy matplotlib
"""

import sys
import os

try:
    import cv2
    import numpy as np
    from pathlib import Path
except ImportError as e:
    print(f"Error: Required library not installed: {e}")
    print("Please install requirements: pip install opencv-python numpy matplotlib")
    sys.exit(1)


def preprocess_image(img):
    """
    Preprocess the image for circle detection.
    
    Args:
        img: Input image (BGR format from OpenCV)
    
    Returns:
        gray: Grayscale image
        blurred: Gaussian blurred image
        edges: Edge detected image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    return gray, blurred, edges


def detect_circles_hough(img, blurred, min_radius=10, max_radius=100):
    """
    Detect circles using Hough Circle Transform.
    
    Args:
        img: Original image for drawing
        blurred: Blurred grayscale image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
    
    Returns:
        circles: Detected circles (x, y, radius)
        result_img: Image with circles drawn
    """
    # Hough Circle Transform
    # Parameters:
    # - dp: Inverse ratio of accumulator resolution
    # - minDist: Minimum distance between circle centers
    # - param1: Upper threshold for edge detection
    # - param2: Accumulator threshold (lower = more false positives)
    # - minRadius, maxRadius: Circle radius range
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,  # Minimum distance between circle centers
        param1=50,   # Upper threshold for edge detection
        param2=30,   # Accumulator threshold (lower = more detections)
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    result_img = img.copy()
    detected_circles = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # Draw circle outline
            cv2.circle(result_img, center, radius, (0, 255, 0), 2)
            # Draw center point
            cv2.circle(result_img, center, 2, (0, 0, 255), 3)
            detected_circles.append((i[0], i[1], i[2]))
    
    return detected_circles, result_img


def detect_circles_contour(img, gray, min_area=100, max_area=5000, circularity_threshold=0.7):
    """
    Detect circles using contour detection with circularity filtering.
    
    Args:
        img: Original image for drawing
        gray: Grayscale image
        min_area: Minimum contour area
        max_area: Maximum contour area
        circularity_threshold: Minimum circularity (4*pi*area/perimeter^2)
    
    Returns:
        circles: Detected circles (x, y, radius)
        result_img: Image with circles drawn
    """
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = img.copy()
    detected_circles = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Filter by circularity
        if circularity < circularity_threshold:
            continue
        
        # Fit a circle to the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Draw circle
        cv2.circle(result_img, center, radius, (255, 0, 0), 2)
        cv2.circle(result_img, center, 2, (0, 0, 255), 3)
        
        detected_circles.append((int(x), int(y), radius))
    
    return detected_circles, result_img


def detect_circles_edge_morphology(img, gray, min_radius=10, max_radius=100):
    """
    Detect circles using edge detection and morphological operations.
    
    Args:
        img: Original image for drawing
        gray: Grayscale image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
    
    Returns:
        circles: Detected circles (x, y, radius)
        result_img: Image with circles drawn
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Morphological operations to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = img.copy()
    detected_circles = []
    
    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if area < np.pi * min_radius * min_radius or area > np.pi * max_radius * max_radius:
            continue
        
        # Fit a circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        if radius < min_radius or radius > max_radius:
            continue
        
        center = (int(x), int(y))
        radius = int(radius)
        
        # Draw circle
        cv2.circle(result_img, center, radius, (0, 255, 255), 2)
        cv2.circle(result_img, center, 2, (255, 0, 255), 3)
        
        detected_circles.append((int(x), int(y), radius))
    
    return detected_circles, result_img


def combine_detections(img, hough_circles, contour_circles, edge_circles):
    """
    Combine results from multiple detection methods.
    
    Args:
        img: Original image
        hough_circles: Circles from Hough transform
        contour_circles: Circles from contour detection
        edge_circles: Circles from edge detection
    
    Returns:
        combined_img: Image with all detections
        all_circles: Combined list of circles
    """
    combined_img = img.copy()
    all_circles = []
    
    # Draw Hough circles in green
    for x, y, r in hough_circles:
        cv2.circle(combined_img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(combined_img, (x, y), 2, (0, 255, 0), 3)
        all_circles.append(('hough', x, y, r))
    
    # Draw contour circles in blue
    for x, y, r in contour_circles:
        cv2.circle(combined_img, (x, y), r, (255, 0, 0), 2)
        cv2.circle(combined_img, (x, y), 2, (255, 0, 0), 3)
        all_circles.append(('contour', x, y, r))
    
    # Draw edge circles in yellow
    for x, y, r in edge_circles:
        cv2.circle(combined_img, (x, y), r, (0, 255, 255), 2)
        cv2.circle(combined_img, (x, y), 2, (0, 255, 255), 3)
        all_circles.append(('edge', x, y, r))
    
    return combined_img, all_circles


def detect_circles(image_path, output_dir="detections", min_radius=15, max_radius=80):
    """
    Main function to detect circles using multiple CV techniques.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save detection results
        min_radius: Minimum circle radius in pixels
        max_radius: Maximum circle radius in pixels
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return False
    
    print(f"Image size: {img.shape[1]} x {img.shape[0]} pixels")
    
    # Preprocess
    print("Preprocessing image...")
    gray, blurred, edges = preprocess_image(img)
    
    # Method 1: Hough Circle Transform
    print("Method 1: Hough Circle Transform...")
    hough_circles, hough_img = detect_circles_hough(img, blurred, min_radius, max_radius)
    print(f"  Detected {len(hough_circles)} circles")
    
    # Method 2: Contour-based detection
    print("Method 2: Contour-based detection...")
    contour_circles, contour_img = detect_circles_contour(
        img, gray, 
        min_area=np.pi * min_radius * min_radius,
        max_area=np.pi * max_radius * max_radius
    )
    print(f"  Detected {len(contour_circles)} circles")
    
    # Method 3: Edge + Morphology
    print("Method 3: Edge detection + Morphology...")
    edge_circles, edge_img = detect_circles_edge_morphology(img, gray, min_radius, max_radius)
    print(f"  Detected {len(edge_circles)} circles")
    
    # Combine results
    print("Combining results...")
    combined_img, all_circles = combine_detections(
        img, hough_circles, contour_circles, edge_circles
    )
    print(f"Total unique detections: {len(all_circles)}")
    
    # Save results
    base_name = Path(image_path).stem
    cv2.imwrite(str(output_path / f"{base_name}_hough.jpg"), hough_img)
    cv2.imwrite(str(output_path / f"{base_name}_contour.jpg"), contour_img)
    cv2.imwrite(str(output_path / f"{base_name}_edge.jpg"), edge_img)
    cv2.imwrite(str(output_path / f"{base_name}_combined.jpg"), combined_img)
    
    # Save detection coordinates
    with open(output_path / f"{base_name}_detections.txt", 'w') as f:
        f.write("Method\tX\tY\tRadius\n")
        for method, x, y, r in all_circles:
            f.write(f"{method}\t{x}\t{y}\t{r}\n")
    
    print(f"\nResults saved to: {output_path.absolute()}")
    print(f"  - Hough: {base_name}_hough.jpg")
    print(f"  - Contour: {base_name}_contour.jpg")
    print(f"  - Edge: {base_name}_edge.jpg")
    print(f"  - Combined: {base_name}_combined.jpg")
    print(f"  - Coordinates: {base_name}_detections.txt")
    
    return True


def main():
    """Main function with command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python3 detect_circles_cv.py <image.jpg> [output_dir] [min_radius] [max_radius]")
        print("\nExample:")
        print("  python3 detect_circles_cv.py data/x_22_y_43.jpg")
        print("  python3 detect_circles_cv.py data/x_22_y_43.jpg detections 15 80")
        print("\nDefault values:")
        print("  output_dir: detections")
        print("  min_radius: 15 pixels")
        print("  max_radius: 80 pixels")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "detections"
    min_radius = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    max_radius = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist: {image_path}")
        sys.exit(1)
    
    success = detect_circles(image_path, output_dir, min_radius, max_radius)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

