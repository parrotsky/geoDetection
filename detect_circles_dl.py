#!/usr/bin/env python3
"""
Deep Learning approaches for detecting circular soil pit falls (karez features).

Supports:
1. YOLOv8 (recommended for direct detection)
2. DINOv3 feature extraction + detection head
"""

import sys
import os
import numpy as np
from pathlib import Path
import cv2

# Check for required libraries
HAS_YOLO = False
HAS_DINOV3 = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

try:
    from transformers import AutoImageProcessor, AutoModel
    import torch
    HAS_DINOV3 = True
except ImportError:
    print("Warning: transformers/torch not installed. Install with: pip install transformers torch")


def detect_yolov8(image_path, model_path=None, conf_threshold=0.25):
    """
    Detect circles using YOLOv8.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained YOLOv8 model (None for pre-trained)
        conf_threshold: Confidence threshold
    
    Returns:
        detections: List of detections (x, y, w, h, conf, class)
        result_img: Image with detections drawn
    """
    if not HAS_YOLO:
        print("Error: ultralytics not available")
        return None, None
    
    # Load model
    if model_path and os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"Loaded custom model: {model_path}")
    else:
        # Use pre-trained model (will need fine-tuning for circles)
        model = YOLO('yolov8n.pt')
        print("Using pre-trained YOLOv8n (needs fine-tuning for circle detection)")
    
    # Run inference
    results = model(image_path, conf=conf_threshold)
    
    # Get detections
    detections = []
    result_img = cv2.imread(image_path)
    
    if len(results) > 0:
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # Convert to center, width, height
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            width = int(x2 - x1)
            height = int(y2 - y1)
            
            detections.append({
                'center': (center_x, center_y),
                'size': (width, height),
                'confidence': float(conf),
                'class': cls
            })
            
            # Draw detection
            cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(result_img, f'{conf:.2f}', (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return detections, result_img


def detect_dinov3_simple(image_path, threshold=0.5):
    """
    Simple DINOv3-based detection using feature similarity.
    This is a simplified example - full implementation would require training.
    
    Args:
        image_path: Path to input image
        threshold: Similarity threshold
    
    Returns:
        detections: List of potential detections
        result_img: Image with detections drawn
    """
    if not HAS_DINOV3:
        print("Error: transformers/torch not available")
        return None, None
    
    print("Note: This is a simplified DINOv3 example.")
    print("Full implementation requires training a detection head on DINOv3 features.")
    
    # Load DINOv3
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    model.eval()
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process image
    inputs = processor(images=img_rgb, return_tensors="pt")
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state
    
    print(f"Extracted features shape: {features.shape}")
    print("To use DINOv3 for detection, you need to:")
    print("1. Extract features from labeled circle examples")
    print("2. Train a detection head (CNN or MLP) on these features")
    print("3. Use the trained head for inference")
    
    # Placeholder: would need trained detection head
    detections = []
    result_img = img.copy()
    
    return detections, result_img


def create_yolo_dataset_from_cv_detections(cv_detections_file, image_dir, output_dir="yolo_dataset"):
    """
    Convert CV detection results to YOLO format for training.
    
    Args:
        cv_detections_file: Path to CV detection results (.txt file)
        image_dir: Directory containing images
        output_dir: Output directory for YOLO dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    labels_dir = output_path / "labels"
    images_dir = output_path / "images"
    labels_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # Read CV detections
    # Format: Method\tX\tY\tRadius
    detections = {}
    with open(cv_detections_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                method, x, y, r = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
                # Find corresponding image
                # This is a simplified example - you'd need to map detections to images
                pass
    
    print("YOLO dataset creation requires:")
    print("1. Manual labeling of circles in images")
    print("2. Converting to YOLO format (normalized coordinates)")
    print("3. Creating train/val/test splits")
    print("\nUse tools like:")
    print("- LabelImg: https://github.com/HumanSignal/labelImg")
    print("- Roboflow: https://roboflow.com")
    print("- CVAT: https://cvat.org")


def main():
    """Main function with command-line interface."""
    if len(sys.argv) < 3:
        print("Usage: python3 detect_circles_dl.py <method> <image.jpg> [model_path] [options]")
        print("\nMethods:")
        print("  yolov8    - Use YOLOv8 for detection")
        print("  dinov3    - Use DINOv3 (requires trained model)")
        print("\nExamples:")
        print("  python3 detect_circles_dl.py yolov8 data/x_22_y_43.jpg")
        print("  python3 detect_circles_dl.py yolov8 data/x_22_y_43.jpg custom_model.pt")
        print("  python3 detect_circles_dl.py dinov3 data/x_22_y_43.jpg")
        sys.exit(1)
    
    method = sys.argv[1].lower()
    image_path = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist: {image_path}")
        sys.exit(1)
    
    output_dir = Path("dl_detections")
    output_dir.mkdir(exist_ok=True)
    
    if method == "yolov8":
        print("Using YOLOv8 for detection...")
        detections, result_img = detect_yolov8(image_path, model_path)
        
        if detections is not None:
            print(f"Detected {len(detections)} circles")
            
            # Save result
            base_name = Path(image_path).stem
            output_path = output_dir / f"{base_name}_yolov8.jpg"
            cv2.imwrite(str(output_path), result_img)
            print(f"Result saved to: {output_path}")
            
            # Save detections
            with open(output_dir / f"{base_name}_yolov8.txt", 'w') as f:
                f.write("Center_X\tCenter_Y\tWidth\tHeight\tConfidence\n")
                for det in detections:
                    cx, cy = det['center']
                    w, h = det['size']
                    conf = det['confidence']
                    f.write(f"{cx}\t{cy}\t{w}\t{h}\t{conf:.4f}\n")
    
    elif method == "dinov3":
        print("Using DINOv3 for detection...")
        detections, result_img = detect_dinov3_simple(image_path)
        
        if detections is not None:
            print(f"Detected {len(detections)} circles")
            # Save result if any detections
            if len(detections) > 0:
                base_name = Path(image_path).stem
                output_path = output_dir / f"{base_name}_dinov3.jpg"
                cv2.imwrite(str(output_path), result_img)
                print(f"Result saved to: {output_path}")
    
    else:
        print(f"Error: Unknown method '{method}'")
        print("Available methods: yolov8, dinov3")
        sys.exit(1)


if __name__ == "__main__":
    main()

