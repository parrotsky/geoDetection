# geoDetection
1. Start with traditional CV to get baseline results and create training labels
2. Train YOLOv8n if you need fast, direct detection (200–500 labeled examples)
3. Use DINOv3 + Detection Head if you have limited data but want higher accuracy
