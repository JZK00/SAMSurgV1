# SAMSurgV1

## SamSurgScratch

- `graph2vid`: Reads images and outputs them as a video.  
- `RNN_detect_train`: Trains an RNN model for change detection. 
- `yolo_labelpreprocess`: Preprocesses labels into a format compatible with YOLO.  
- `detect`: Detection script that combines RNN and YOLO.
- `batch_prompt`: Used for batch prompting under the sam2 framework; must be run in sam2/segment-anything-2/notebooks/video_predictor_example.ipynb.
- `read_bbox.py`: Reads bounding boxes.
- `calc_iou`: Calculates the mean IoU of all final results; must be run in sam2/segment-anything-2/notebooks/video_predictor_example.ipynb.
