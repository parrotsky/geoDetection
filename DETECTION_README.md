# Circle Detection for Karez Soil Pit Falls

This directory contains scripts for detecting circular soil pit falls (karez system features) in aerial/satellite imagery using both traditional Computer Vision and Deep Learning approaches.

## Files Created

1. **`detect_circles_cv.py`** - Traditional CV detection using OpenCV
2. **`detect_circles_dl.py`** - Deep Learning detection (YOLOv8, DINOv3)
3. **`dl_detection_recommendations.md`** - Detailed recommendations for DL approaches
4. **`requirements.txt`** - Python dependencies

## Quick Start

### 1. Install Dependencies

```bash
# For traditional CV
pip install opencv-python numpy matplotlib

# For deep learning (optional)
pip install ultralytics  # For YOLOv8
pip install transformers torch torchvision  # For DINOv3

# Or install all
pip install -r requirements.txt
```

### 2. Test Traditional CV Detection

```bash
# Test on example tile
python3 detect_circles_cv.py data/x_22_y_43.jpg

# With custom parameters
python3 detect_circles_cv.py data/x_22_y_43.jpg detections 15 80
```

**Output:**
- `x_22_y_43_hough.jpg` - Hough Circle Transform results
- `x_22_y_43_contour.jpg` - Contour-based detection results
- `x_22_y_43_edge.jpg` - Edge detection results
- `x_22_y_43_combined.jpg` - All methods combined
- `x_22_y_43_detections.txt` - Detection coordinates

### 3. Deep Learning Detection

```bash
# YOLOv8 (requires trained model)
python3 detect_circles_dl.py yolov8 data/x_22_y_43.jpg custom_model.pt

# DINOv3 (example - requires training)
python3 detect_circles_dl.py dinov3 data/x_22_y_43.jpg
```

## Detection Methods

### Traditional CV (`detect_circles_cv.py`)

Uses three complementary methods:

1. **Hough Circle Transform**
   - Detects circular shapes using gradient information
   - Good for well-defined circles
   - Parameters: `min_radius`, `max_radius`, `minDist`, `param2`

2. **Contour-based Detection**
   - Finds contours and filters by circularity
   - Good for irregular circles
   - Parameters: `min_area`, `max_area`, `circularity_threshold`

3. **Edge Detection + Morphology**
   - Uses Canny edges and morphological operations
   - Good for detecting circular patterns
   - Parameters: `min_radius`, `max_radius`

**Advantages:**
- No training data needed
- Fast inference
- Good baseline results
- Can create labeled data for DL training

**Limitations:**
- May have false positives
- Requires parameter tuning
- Less robust to variations

### Deep Learning (`detect_circles_dl.py`)

#### YOLOv8 (Recommended)
- **Best for**: Direct detection, fast processing
- **Requirements**: Labeled training data (200-500 examples)
- **Speed**: Very fast (~10-50ms per image)
- **Accuracy**: 85-95% with good training data

#### DINOv3
- **Best for**: Limited labeled data, high accuracy
- **Requirements**: Feature extraction + detection head training
- **Speed**: Moderate (~100-200ms per image)
- **Accuracy**: 90-98% with proper training

## Recommended Workflow

### Phase 1: Baseline (Traditional CV)
1. Run `detect_circles_cv.py` on sample tiles
2. Manually verify and correct detections
3. Use results to create initial training dataset

### Phase 2: Deep Learning Training
1. **Option A - YOLOv8** (Easier):
   ```bash
   # Label 200-500 examples using LabelImg or similar
   # Train model
   yolo train data=karez_dataset.yaml model=yolov8n.pt epochs=100
   ```

2. **Option B - DINOv3** (More accurate, less data):
   - Extract DINOv3 features from labeled examples
   - Train lightweight detection head
   - Fine-tune end-to-end

### Phase 3: Production
1. Process all tiles with trained model
2. Post-process to filter false positives
3. Generate detection statistics/maps

## Parameters Tuning

### For Traditional CV

**Circle Size Range:**
- Based on your image, circles appear to be ~15-80 pixels in radius
- Adjust `min_radius` and `max_radius` accordingly

**Hough Circle Parameters:**
- `minDist`: Minimum distance between circle centers (default: 30)
- `param2`: Accumulator threshold - lower = more detections (default: 30)
- `param1`: Upper threshold for edge detection (default: 50)

**Contour Parameters:**
- `circularity_threshold`: 0.7 = fairly circular, 0.9 = very circular
- `min_area`: Minimum circle area in pixels²
- `max_area`: Maximum circle area in pixels²

### For Deep Learning

**YOLOv8:**
- `conf_threshold`: Confidence threshold (default: 0.25)
- `imgsz`: Image size for training (640 recommended)
- `epochs`: Training epochs (100-300)

## Expected Results

### Traditional CV
- **Precision**: ~70-85%
- **Recall**: ~60-80%
- **Speed**: ~50-200ms per 640x640 tile

### YOLOv8 (with training)
- **Precision**: ~85-95%
- **Recall**: ~80-90%
- **Speed**: ~10-50ms per tile

### DINOv3 (with training)
- **Precision**: ~90-98%
- **Recall**: ~85-95%
- **Speed**: ~100-200ms per tile

## Next Steps

1. **Test traditional CV** on `x_22_y_43.jpg` to see baseline results
2. **Review detections** and manually label correct ones
3. **Create training dataset** in YOLO format
4. **Train YOLOv8** model on your data
5. **Evaluate** on test set and iterate

## Troubleshooting

**OpenCV not found:**
```bash
pip install opencv-python
```

**No detections found:**
- Adjust `min_radius` and `max_radius` parameters
- Try different methods (hough vs contour)
- Check image quality/preprocessing

**Too many false positives:**
- Increase `param2` in Hough transform
- Increase `circularity_threshold` in contour method
- Use deep learning approach with training

## References

- OpenCV Circle Detection: https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
- YOLOv8 Documentation: https://docs.ultralytics.com/
- DINOv3 Paper: https://arxiv.org/abs/2304.07193

