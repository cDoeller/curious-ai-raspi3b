import cv2
import numpy as np
import math


import camera
from filesystem import get_now_timestamp

from config import HISTOGRAM_OVERALL_BINS_DEF, HISTOGRAM_FIELDS_BINS_DEF, FIELDS_ROWS_DEF, FIELDS_COLS_DEF, HISTORGAM_ROUND_DECIMALS, MOVEMENT_MASK_THRESH_DEF, MOTION_MASK_MIN_AREA_DEF, MOTION_MASK_MERGE_DISTANCE_DEF
from database import get_observation

# TODO 
# tell agent rounding params and params of data
# agent has to reverse engineer the id of each cell in spatial analysis
# movement vector: relative Bewegungsanalyse innerhalb einer Bildsequenz!
# save ephemeral images (movement: masks, bounding boxes / edges,  )

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def round_histogram(histogram, decimals=HISTORGAM_ROUND_DECIMALS):
    return [round(float(value), decimals) for value in histogram]

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

        histogram[name] = round_histogram(values.flatten())

    return histogram

def analyze_brightness_histogram(img, bins):
    # Convert the image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate a normalized brightness histogram.
    histogram = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    # calculate relative amount (probability), not absolute counts
    # does not specify for a certain resolution
    histogram = histogram / histogram.sum()

    return round_histogram(histogram.flatten())

def analyze_single_image(img, histogram_bins=HISTOGRAM_OVERALL_BINS_DEF):
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
def analyze_n_fields(
    img,
    bins=HISTOGRAM_FIELDS_BINS_DEF,
    num_fields_rows=FIELDS_ROWS_DEF,
    num_fields_cols=FIELDS_COLS_DEF
):
    if num_fields_rows < 1 or num_fields_cols < 1:
        raise ValueError("Number of field rows and columns must be at least 1.")

    # only info for size, not channel
    height, width = img.shape[:2]

    fields = []

    for row in range(num_fields_rows):
        for col in range(num_fields_cols):

            # get the boundaries of each cell in px
            y_start = row * height // num_fields_rows
            y_end = (row + 1) * height // num_fields_rows

            x_start = col * width // num_fields_cols
            x_end = (col + 1) * width // num_fields_cols

            # make image of cell boundaries
            field_img = img[y_start:y_end, x_start:x_end]

            # get histogram data of each cell
            brightness = analyze_brightness_histogram(field_img, bins)
            color = analyze_color_histogram(field_img, bins)

            # Store the histogram arrays after one another with decoding info
            fields.append([
                brightness,
                color["red"],
                color["green"],
                color["blue"]
            ])

    return {
        "grid": [num_fields_rows, num_fields_cols],
        "bins": bins,
        "channel_order": ["brightness", "red", "green", "blue"],
        "fields": fields
    }

# ==========================================
# MOVEMENT ANALYSIS
# ==========================================

# threshold ändert emfpindlichkeit, ggf anpassen bei höherem gain
# Maske i gehört zu den Bildern i und i+1, anzahl masken = anzahl frames -1
def create_motion_masks(img_sequence, threshold=MOVEMENT_MASK_THRESH_DEF, blur_kernel=(5, 5)):

    # Check if enough images are available for frame differencing.
    if img_sequence is None or len(img_sequence) < 2:
        print("[ANALYZER] [ERR] At least 2 images are required.")
        return []

    motion_masks = []

    for i in range(len(img_sequence) - 1):

        img_a = img_sequence[i]
        img_b = img_sequence[i + 1]

        # Skip failed camera captures.
        if img_a is None or img_b is None:
            print(f"[ANALYZER] [WARN] Skipping image pair {i}-{i+1}: missing image.")
            motion_masks.append(None) # keep the position, so mask index == pair index
            continue

        # Check if both images have the same dimensions.
        if img_a.shape != img_b.shape:
            print(f"[ANALYZER] [WARN] Skipping image pair {i}-{i+1}: different dimensions.")
            motion_masks.append(None)
            continue

        # Convert BGR images to grayscale.
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        # Reduce image noise before calculating differences
        # kernel = spatial radius of averaging effect for each pixel
        gray_a = cv2.GaussianBlur(gray_a, blur_kernel, 0)
        gray_b = cv2.GaussianBlur(gray_b, blur_kernel, 0)

        # Calculate absolute pixel differences between both images.
        difference = cv2.absdiff(gray_a, gray_b)

        # Keep only pixel differences above the threshold, everything else black
        _, motion_mask = cv2.threshold(
            difference,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

        motion_masks.append(motion_mask)

        print(f"[ANALYZER] Motion mask created for pair {i}-{i+1}.")

    return motion_masks


# find connected white pixel spaces in movement masks and define bounding box
# min_area allows for filtering out very small areas like pixel noise from high gain
# result: list of motion regions (box, area) for each mask [[mask1],[mask2],[mask3]]
def find_motion_regions(motion_masks, min_area=MOTION_MASK_MIN_AREA_DEF,  merge_distance=MOTION_MASK_MERGE_DISTANCE_DEF):

    all_regions = []

    # The "brush": a square of white pixels.
    # Bigger merge_distance = thicker brush = more areas grow together.
    # Return a new array of given shape and type, filled with ones
    # arg #1 = shape of the array, arg #2 = data type (unsigned int 8 bit (256 max))
    kernel = np.ones((merge_distance, merge_distance), np.uint8)

    for i, mask in enumerate(motion_masks):

        # make all white areas thicker, so areas that are
        # close together touch each other and become one area.
        # a pixel element is ‘1’ if atleast one pixel under the kernel is ‘1’.
        thick_mask = cv2.dilate(mask, kernel)

        # normal mask version
        # Find connected external contours in the binary motion mask (white areas)
        # contours, _ = cv2.findContours(
        #     mask,
        #     cv2.RETR_EXTERNAL,
        #     cv2.CHAIN_APPROX_SIMPLE
        # )

        # dilate version
        contours, _ = cv2.findContours(
            thick_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []

        for contour in contours:

            # Calculate the contour area in pixels. (square pixels)
            area = cv2.contourArea(contour)

            # Ignore regions smaller than the configured minimum.
            if area < min_area:
                continue

            # Get bounding box coordinates.
            x, y, width, height = cv2.boundingRect(contour)

            regions.append({
                "bbox": [x, y, width, height],
                "area": area
            })

        all_regions.append(regions)

        print(
            f"[ANALYZER] Image pair {i}: "
            f"{len(regions)} motion regions detected."
        )

    return all_regions

# optical flow: only the boxes of regions, only the white pixels of mask
def calculate_region_vectors(img_sequence, motion_masks, motion_regions):

    all_vectors = []

    for i, regions in enumerate(motion_regions):

        # original images that belong to this region array / "regions calculation"
        img_a = img_sequence[i]
        img_b = img_sequence[i + 1]

        # Skip failed camera captures and do not mess up index
        if img_a is None or img_b is None:
            all_vectors.append([])
            continue

        # grayscale images, slight blur against noise
        gray_a = cv2.GaussianBlur(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        gray_b = cv2.GaussianBlur(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        # get the mask that belongs to this image pair
        mask = motion_masks[i]  

        pair_vectors = []

        # loop through all regions in array of detected regions for this mask
        for region in regions:

            # destructure values from array
            x, y, width, height = region["bbox"]

            # Cut the box area out of both images and out of the mask
            # Slicing: [row_start:row_end, col_start:col_end] -> y first, then x!
            crop_a = gray_a[y:y + height, x:x + width]
            crop_b = gray_b[y:y + height, x:x + width]
            crop_mask = mask[y:y + height, x:x + width]

            # Very small boxes are too small for the flow algorithm
            # also helps agains noise!
            if width < 32 and height < 32:
                print(f"[ANALYZER] [WARN] Pair {i}: box {region['bbox']} too small for flow.")
                continue

            # flow[row, col, 0] = dx of this pixel, flow[row, col, 1] = dy.
            # The numbers are common default values for this function.
            flow = cv2.calcOpticalFlowFarneback(
                crop_a, crop_b, None,
                0.5,  # pyramid scale
                3,    # pyramid levels (helps with bigger movements)
                15,   # window size (bigger = smoother, less detail)
                3,    # iterations
                5, 1.2, 0
            )
            # print ("flow result:" , flow)

            # get every pixel that is white in the motion mask
            moving = crop_mask > 0

            # Too few moving pixels = no reliable result
            # helps agains noise / filters out small areas
            if np.count_nonzero(moving) < 20:
                continue

            # Mean of dx and dy over the moving pixels only
            dx = float(np.mean(flow[:, :, 0][moving]))
            dy = float(np.mean(flow[:, :, 1][moving]))

            # Length of the arrow (Pythagoras) and its direction in degrees.
            # The y-axis of an image points DOWN, so a negative angle = moving up.
            magnitude = math.sqrt(dx * dx + dy * dy)
            angle = math.degrees(math.atan2(dy, dx))

            pair_vectors.append({
                "bbox": region["bbox"],
                "vector": {
                    "dx": round(dx, 2),
                    "dy": round(dy, 2),
                    "magnitude": round(magnitude, 2),
                    "angle": round(angle, 1)
                }
            })

        all_vectors.append(pair_vectors)
        print(f"[ANALYZER] Image pair {i}: {len(pair_vectors)} vectors calculated.")

    return all_vectors

def analyze_movement (img_sequence, step):

    return 

# ==========================================
# OBJECT DETECTION
# ==========================================

def detect_objects (img):
    return

if __name__ == "__main__":

    # ==============================
    # CAMERA INITIALIZATION
    # ==============================

    print("[CAMERA] Initializing camera...")
    camera_device = camera.initialize_camera()

    try:

        # ==============================
        # CAMERA CALIBRATION
        # ==============================

        print("[CAMERA] Starting calibration...")
        calibration = camera.calibrate_gain(camera_device)

        print("[CAMERA] Calibration values:", calibration)

        # ==============================
        # CAPTURE IMAGE SEQUENCE
        # ==============================

        print("[CAMERA] Capturing image sequence...")

        frame_sequence, step = camera.capture_n_images(camera_device)

        print(f"[CAMERA] Captured {len(frame_sequence)} images.")
        print(f"[CAMERA] Step: {step}")

    finally:
        # Always release the camera after capture.
        camera_device.release()
        print("[CAMERA] Camera released.")

    # ==============================
    # MOVEMENT ANALYSIS TEST
    # ==============================

    print("[ANALYZER] Creating motion masks...")

    motion_masks = create_motion_masks(frame_sequence)
    # print ("motion masks: ", motion_masks)  

    motion_regions = find_motion_regions(motion_masks)
    print ("motion regions: ", motion_regions)

    motion_vectors = calculate_region_vectors(frame_sequence, motion_masks, motion_regions)
    print("motion vectors: ", motion_vectors)

    # string for filename
    now_time_string = get_now_timestamp()

    # if motion_regions are not empty like [[],[]]
    if any(motion_regions):
        # save motion masks
        for i, mask in enumerate(motion_masks):
            image_path = f"img-tst/motion_mask_{now_time_string}_{i}.jpg"
            print (image_path)

            success = cv2.imwrite(image_path, mask)

            if success:
                print(f"[ANALYZER] Saved: {image_path}")
            else:
                print(f"[ANALYZER] [ERR] Could not save: {image_path}")

        print(f"[ANALYZER] {len(motion_masks)} motion masks created.")

        # save og-imgs with bounding box of detected regions
        for i, regions in enumerate(motion_regions):

            img = frame_sequence[i].copy()

            for region in regions:

                x, y, width, height = region["bbox"]

                cv2.rectangle(
                    img,
                    (x, y),
                    (x + width, y + height),
                    (255,255,255),
                    1
                )

            image_path = f"img-tst/motion_bbox_{now_time_string}_{i}.jpg"

            cv2.imwrite(image_path, img)

            print(f"[ANALYZER] Saved: {image_path}")