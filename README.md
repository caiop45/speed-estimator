**Speed Estimator: Summary**

This script uses a YOLO-based model to detect vehicles in a video (https://mega.nz/file/N0Vm1KLb#oX-oLyW6lvoYTA-zsmBdI6rrr22tEwdydaaUG1Zk95Q), calculates each vehicle's speed from positional changes over time, and then uses Tesseract OCR to recognize any license plate text in the detected regions. The processed frames are saved as **`output.mp4`**, and the extracted information—vehicle speed, detection time, and OCR text—is logged to **`speed_data.csv`**.

**Key Components**  
1. **YOLO Detection**: Identifies and tracks vehicles to obtain bounding boxes.  
2. **Speed Calculation**: Uses changes in the vehicle’s position between frames, divided by elapsed time.  
3. **Tesseract OCR**: Extracts textual data (e.g., license plates) from detected regions.  
4. **CSV Logging**: Saves date, time, ID, speed, and recognized plate text to a file for further analysis.
