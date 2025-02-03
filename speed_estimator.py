import cv2
from time import time
import numpy as np
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
from datetime import datetime
import csv
import os
import pytesseract

# Tesseract executable. You can comment this line
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class SpeedEstimator(BaseSolution):
    """
    Main class extending BaseSolution to estimate object speed,
    perform OCR, and save the results to a CSV file.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_region()
        self.spd = {}
        self.trkd_ids = []
        self.trk_pt = {}
        self.trk_pp = {}
        self.logged_ids = set()
        self.csv_file = "speed_data.csv"
        self.initialize_csv()

    def initialize_csv(self):
        """Creates a CSV file with a header if it doesn't already exist."""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["id", "date", "time", "track_id", "class_name", "speed", "numberplate"])
        self.csv_id = 0

    def perform_ocr(self, image_array):
        """
        Performs OCR on the provided image using pytesseract and returns the extracted text.
        Raises exceptions if the image is empty or invalid.
        """
        if image_array is None:
            raise ValueError("The image is empty.")
        if isinstance(image_array, np.ndarray):
            img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(img_rgb, lang='eng')
        else:
            raise TypeError("The input image is not a valid numpy array.")
        return text

    def save_to_csv(self, date, time_str, track_id, class_name, speed, numberplate):
        """
        Saves the extracted data to a CSV file, incrementing the ID automatically.
        """
        self.csv_id += 1
        with open(self.csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([self.csv_id, date, time_str, track_id, class_name, speed, numberplate])
        print(f"Data saved to CSV: {date}, {time_str}, {track_id}, {class_name}, {speed}, {numberplate}")

    def estimate_speed(self, im0):
        """
        Estimates object speeds, tracks them, draws information on the image,
        and performs OCR for license plate reading.
        """
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)
        current_time = datetime.now()

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            speed_label = f"{int(self.spd[track_id])} km/h" if track_id in self.spd else self.names[int(cls)]
            label = f"ID: {track_id} {speed_label}"
            self.annotator.box_label(box, label=label, color=colors(track_id, True))

            if self.LineString([self.trk_pp[track_id], self.track_line[-1]]).intersects(self.r_s):
                direction = "known"
            else:
                direction = "unknown"

            if direction == "known" and track_id not in self.trkd_ids:
                self.trkd_ids.append(track_id)
                time_difference = time() - self.trk_pt[track_id]
                if time_difference > 0:
                    speed = np.abs(
                        self.track_line[-1][1].item() - self.trk_pp[track_id][1].item()
                    ) / time_difference
                    self.spd[track_id] = round(speed)

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]

            x1, y1, x2, y2 = map(int, box)
            cropped_image = np.array(im0)[y1:y2, x1:x2]
            ocr_text = self.perform_ocr(cropped_image)

            class_name = self.names[int(cls)]
            speed = self.spd.get(track_id)

            if track_id not in self.logged_ids and ocr_text.strip() and speed is not None:
                self.save_to_csv(
                    current_time.strftime("%Y-%m-%d"),
                    current_time.strftime("%H:%M:%S"),
                    track_id,
                    class_name,
                    speed if speed is not None else 0.0,
                    ocr_text
                )
                self.logged_ids.add(track_id)

        self.display_output(im0)
        return im0


# Open the video file and check for errors
cap = cv2.VideoCapture('mycarplate.mp4')
if not cap.isOpened():
    print("Error opening the video")
    exit()

# Define the points for the monitored region
region_points = [(0, 145), (1018, 145)]

# Initialize the speed estimator object
speed_obj = SpeedEstimator(
    region=region_points,
    model="best.pt",
    line_width=2
)

# Settings for saving the processed video
original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_width, frame_height = 1020, 500
fps_out = original_fps / 3
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps_out, (frame_width, frame_height))

count = 0

# Main loop to read and process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:  # Process only 1 out of every 3 frames
        continue

    frame = cv2.resize(frame, (frame_width, frame_height))
    result = speed_obj.estimate_speed(frame)
    cv2.imshow("RGB", result)
    out.write(result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
