# Deep Learning Detection Recommendations for Karez Circle Detection

## Problem Overview
Detecting circular soil pit falls (karez system features) in aerial/satellite imagery:
- **Target**: Circular depressions (light-colored circles with dark centers)
- **Size**: Approximately 15-80 pixels in radius (640x640 tile images)
- **Characteristics**: Relatively easy to recognize, distinct circular patterns
- **Scale**: Need to process many tiles (10,000+ tiles)

## Recommended Approaches

### 1. **DINOv3 (Vision Transformer) - RECOMMENDED FOR FEATURE EXTRACTION**

**Pros:**
- State-of-the-art self-supervised learning
- Excellent feature representation without labeled data
- Can be fine-tuned with minimal labeled examples
- Good for transfer learning
- Strong performance on aerial/satellite imagery

**Cons:**
- Not a detection model by itself (needs additional detection head)
- Larger model size (though smaller variants available)
- Requires fine-tuning for specific task

**Implementation Strategy:**
1. Use DINOv3 as a feature extractor
2. Add a detection head (e.g., simple CNN classifier or object detection head)
3. Fine-tune on labeled karez circles
4. Can use DINOv3 features for:
   - Circle/non-circle classification
   - Circle localization (with additional regression head)
   - Similarity matching (find similar patterns)

**Code Example:**
```python
# Using DINOv3 for feature extraction
from transformers import AutoImageProcessor, AutoModel
import torch

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

# Extract features
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    features = outputs.last_hidden_state
```

**Best For:** 
- When you have limited labeled data
- Need strong feature representations
- Want to leverage self-supervised learning

---

### 2. **Small Detection Models - RECOMMENDED FOR DIRECT DETECTION**

#### 2.1 **YOLOv8 Nano (YOLOv8n)** ⭐ **TOP RECOMMENDATION**

**Pros:**
- Very fast inference (real-time capable)
- Small model size (~6MB)
- Excellent for small object detection
- Easy to use and train
- Good balance of speed and accuracy
- Can detect multiple objects per image

**Cons:**
- Requires labeled training data
- May need fine-tuning for specific circle sizes

**Implementation:**
```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')  # or yolov8n-seg.pt for segmentation

# Fine-tune on your data
model.train(data='karez_dataset.yaml', epochs=100, imgsz=640)

# Inference
results = model('data/x_22_y_43.jpg')
```

**Best For:**
- Direct object detection
- Fast processing of many tiles
- When you can create labeled dataset

#### 2.2 **EfficientDet-D0/D1**

**Pros:**
- Efficient architecture
- Good for small objects
- Multiple scale variants (D0-D7)
- Good accuracy/speed trade-off

**Cons:**
- Slightly slower than YOLOv8n
- Requires more setup

**Best For:**
- When you need efficient detection
- Multiple object scales

#### 2.3 **RetinaNet (ResNet-50 backbone)**

**Pros:**
- Good for small object detection
- Focal loss handles class imbalance
- Well-established architecture

**Cons:**
- Slower than YOLOv8
- More complex setup

---

### 3. **Hybrid Approach: DINOv3 + Detection Head** ⭐ **BEST FOR ACCURACY**

Combine DINOv3 features with a lightweight detection head:

**Architecture:**
```
Input Image (640x640)
    ↓
DINOv3 Feature Extractor (frozen or fine-tuned)
    ↓
Feature Map
    ↓
Detection Head (CNN or Transformer)
    ↓
Circle Detections (bbox + confidence)
```

**Advantages:**
- Leverages DINOv3's strong features
- Lightweight detection head
- Can fine-tune end-to-end
- Good accuracy with limited data

**Implementation:**
```python
import torch
import torch.nn as nn
from transformers import AutoModel

class DINOv3Detector(nn.Module):
    def __init__(self):
        super().__init__()
        # DINOv3 backbone
        self.backbone = AutoModel.from_pretrained('facebook/dinov2-base')
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 5, 1)  # 4 for bbox + 1 for confidence
        )
    
    def forward(self, x):
        features = self.backbone(x).last_hidden_state
        detections = self.detection_head(features)
        return detections
```

---

## Comparison Table

| Method | Speed | Accuracy | Data Needed | Model Size | Best Use Case |
|--------|-------|----------|-------------|------------|---------------|
| **YOLOv8n** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | Small (~6MB) | Direct detection, many tiles |
| **DINOv3 + Head** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low | Medium | Limited data, high accuracy |
| **EfficientDet-D0** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | Small | Balanced approach |
| **RetinaNet** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | Medium | Small objects |

---

## Recommended Workflow

### Phase 1: Quick Start (Traditional CV)
1. Use `detect_circles_cv.py` to get baseline results
2. Manually label ~100-200 positive examples from results
3. Create initial training dataset

### Phase 2: Deep Learning Setup
1. **Start with YOLOv8n** (easiest, fastest):
   - Label 200-500 examples
   - Train YOLOv8n on your data
   - Evaluate on test set
   - If good enough → deploy

2. **If need better accuracy → DINOv3 + Detection Head**:
   - Use DINOv3 to extract features
   - Train lightweight detection head
   - Fine-tune end-to-end if needed

### Phase 3: Production
- Process all tiles with trained model
- Post-process to filter false positives
- Generate detection maps/statistics

---

## Implementation Scripts Needed

1. **Data Preparation Script**: Convert CV detections to YOLO format
2. **YOLOv8 Training Script**: Train on karez dataset
3. **DINOv3 Feature Extraction Script**: Extract features for training
4. **Inference Script**: Process all tiles with trained model
5. **Evaluation Script**: Calculate precision/recall/F1

---

## Quick Start Commands

### Install Dependencies
```bash
# For YOLOv8
pip install ultralytics

# For DINOv3
pip install transformers torch torchvision

# For traditional CV (already in detect_circles_cv.py)
pip install opencv-python numpy matplotlib
```

### Train YOLOv8
```bash
# 1. Prepare dataset in YOLO format
# 2. Create dataset.yaml
# 3. Train
yolo train data=karez_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### Use DINOv3
```python
from transformers import AutoImageProcessor, AutoModel
model = AutoModel.from_pretrained('facebook/dinov2-base')
```

---

## Final Recommendation

**For your use case (many tiles, relatively easy to detect circles):**

1. **Start with YOLOv8n** - Fast, easy, good results
2. **If accuracy insufficient → DINOv3 + Detection Head**
3. **Use traditional CV** (`detect_circles_cv.py`) for:
   - Initial exploration
   - Creating training labels
   - Validation/comparison

**Expected Performance:**
- YOLOv8n: 85-95% accuracy with 200-500 labeled examples
- DINOv3 + Head: 90-98% accuracy with 100-300 labeled examples
- Traditional CV: 70-85% accuracy (baseline)

