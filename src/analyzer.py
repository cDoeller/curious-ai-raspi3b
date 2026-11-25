import cv2
import numpy as np

from config import HISTOGRAM_BINS_DEF, FIELDS_ROWS_DEF, FIELDS_COLS_DEF
from database import get_observation


# ==========================================
# SINGLE IMAGE OVERALL ANALYSIS
# ==========================================

def analyze_saturation(img):
    # Convert the image from BGR to HSV = Hue Saturation BrightnessValue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Calculate the average saturation (0.0 - 1.0).
    # array: [[ROW],[ROW],]; ROW: [PX],[PX],; PX: [H,S,V]
    # Array-Slicing-Operation: hsv[height, width, channel]
    # all rows, all pixels, only channel 1 (S); np-array
    # img[0, 1, 2] = img[0][1][2] = 1st row, 2nd pixel, 3rd channel
    # np.mean = avging all saturation values
    # / 255 = normalizing for numbers 0-1
    saturation = np.mean(hsv[:, :, 1]) / 255

    return float(saturation)

def analyze_entropy(img):
    # Low entropy: few different brightness values.
    # High entropy: a broader, more complex distribution of brightness.

    # Convert the image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the normalized brightness histogram.
    # hist = how often are certain brightness values present
    # [0] = channel (only), None = no mask / all pixels
    # [256] = amt of histogram bins, [0, 256] = analysis value range
    # 256 bins = every value gets its own bin 
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # .sum adds all entries of historgam 
    # divide every entry in histogram by total sum
    # = get probability for every brightness value
    probabilities = histogram / histogram.sum()

    # Ignore empty bins to avoid log2(0).
    # Boolean Indexing in NumPy
    probabilities = probabilities[probabilities > 0]

    # Calculate Shannon entropy in bits
    # 0 bit = black image, 1 bit = black / white halves img
    # 8 bit = equally distributed 256 brightness vals in img
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)

def analyze_color_histogram(img, bins):
    # Convert BGR to RGB for clearly named color channels.
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    histogram = {} # dict

    # Calculate a normalized histogram for each color channel.
    # channel = 0,1,2 name = enum
    for channel, name in enumerate(["red", "green", "blue"]):
        values = cv2.calcHist([rgb], [channel], None, [bins], [0, 256])
        # calculate relative amount (probability), not absolute counts
        # does not specify for a certain resolution
        values = values / values.sum()

        histogram[name] = values.flatten().tolist()

    return histogram

def analyze_brightness_histogram(img, bins):
    # Convert the image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate a normalized brightness histogram.
    histogram = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    # calculate relative amount (probability), not absolute counts
    # does not specify for a certain resolution
    histogram = histogram / histogram.sum()

    return histogram.flatten().tolist()

def analyze_single_image(img, histogram_bins=HISTOGRAM_BINS_DEF):
    # Collect all single-image analysis results.
    analysis = {
        "saturation_normalized": analyze_saturation(img),
        "entropy_shannon_bits": analyze_entropy(img),
        f"color_histogram_{histogram_bins}_bins_probabilities": analyze_color_histogram(img, histogram_bins),
        f"brightness_histogram_{histogram_bins}_bins_probabilities": analyze_brightness_histogram(img, histogram_bins)
    }

    return analysis

# ==========================================
# 4 x 4 SPATIAL METRICS ANALYSIS
# ==========================================

def analyze_n_fields(img, bins=HISTOGRAM_BINS_DEF, num_fields_rows=FIELDS_ROWS_DEF, num_fields_cols=FIELDS_COLS_DEF):
    if num_fields_rows < 1 or num_fields_cols < 1:
        raise ValueError("Number of field rows and columns must be at least 1.")
   
    # Get the image dimensions. Unpacking.
    height, width = img.shape[:2]

    # Store the analysis results for all fields.
    fields = []

    # Iterate through all rows and columns.
    for row in range(num_fields_rows):
        for col in range(num_fields_cols):

            # Calculate the boundaries of the current field.
            # // = Integer Division: no floating point results!
            y_start = row * height // num_fields_rows
            y_end = (row + 1) * height // num_fields_rows

            x_start = col * width // num_fields_cols
            x_end = (col + 1) * width // num_fields_cols

            # Extract the current field from the image.
            # third dimension (3x color ch) is added automatically
            field_img = img[y_start:y_end, x_start:x_end]

            # Analyze brightness and color distribution.
            brightness = analyze_brightness_histogram(field_img, bins)
            color = analyze_color_histogram(field_img, bins)

            # Store the results of the current field.
            fields.append({
                "field_id": f"r{row}_c{col}",
                "row": row,
                "col": col,
                "brightness_distribution": brightness,
                "color_distribution": color
            })

    return fields, [num_fields_rows, num_fields_cols]

# ==========================================
# MOVEMENT ANALYSIS
# ==========================================

def analyze_movement (img_sequence):
    return 

# ==========================================
# OBJECT DETECTION
# ==========================================

def detect_objects (img):
    return

if __name__ == "__main__":
    # test: get last image
    last_observation = get_observation(2)
    img_path = last_observation["image_path"] + ".jpg"

    # Load the image.
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    # Run single-image analysis.
    analysis = analyze_single_image(img)
    
    print(analysis)