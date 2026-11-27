import cv2
import numpy as np
import re

from config import (
    MODEL_PATH, LABEL_PATH,
    CONFIDENCE_THRESHOLD, NMS_THRESHOLD,
    FEATURE_MAPS, SCALES
)

# DETECTION PIPELINE (SSD MobileNet v2)
#
# ONCE AT START
# 1. Load labels: class id -> name (e.g. 47 -> "cup")
# 2. Load the model from the ONNX file
# 3. Build the anchors (fixed template boxes, same for every image)
#
# FOR EVERY IMAGE
# 4. Prepare input: resize to 300x300, normalize values, BGR -> RGB (blob)
# 5. Run the model (net.forward): raw numbers for every anchor
# 6. Reorder raw outputs into two tables (box corrections, class scores)
# 7. Decode boxes: anchor + correction = real box (relative 0-1)
# 8. Class scores -> probabilities (sigmoid)
# 9. For each class: keep confident boxes, convert them to pixels of the
#    original image, remove duplicates (NMS)
# 10. Return the list of detections
#
# SEPARATE, OPTIONAL (not part of the detection itself)
# 11. Draw boxes and labels on a copy of the image
# 12. Save the image / store results

# ==========================================
# SETUP HELPERS (called once at the bottom of this section)
# ==========================================

def load_labels(label_path):
    # Read the whole label file as one text string.
    with open(label_path, "r") as file:
        label_text = file.read()

    # build a dict with id: label_name from labels file
    # The file contains blocks like: id: 47 ... display_name: "cup".
    # The regex finds every (id, name) pair and we build a dict {47: "cup"}.
    # re.DOTALL lets ".*?" also match across line breaks.
    labels = {
        int(class_id): name
        for class_id, name in re.findall(
            r'id:\s*(\d+).*?display_name:\s*"([^"]+)"',
            label_text,
            re.DOTALL
        )
    }

    return labels


def build_anchors(feature_maps, scales):
    # Anchors are fixed template boxes laid over the image like a grid.
    # The model does not draw boxes, it only says how to adjust each template.
    # Format per anchor: [center_y, center_x, height, width], all relative (0-1).
    anchors = []

    for layer, feature_size in enumerate(feature_maps):

        scale = scales[layer]

        # Each stage uses a different set of shapes: (aspect ratio, size).
        # Ratio 1.0 = square, 2.0 = wide, 0.5 = tall.
        # The first stage uses only 3 shapes, the others use 6.
        if layer == 0:
            specs = [
                (1.0, 0.1),
                (2.0, scale),
                (0.5, scale)
            ]
        else:
            # Extra square anchor with a size between this and the next stage.
            # For the last stage there is no next one, so 1.0 is used.
            next_scale = (
                scales[layer + 1]
                if layer < 5
                else 1.0
            )

            specs = [
                (1.0, scale),
                (1.0, np.sqrt(scale * next_scale)),
                (2.0, scale),
                (0.5, scale),
                (3.0, scale),
                (1.0 / 3.0, scale)
            ]

        # Walk over every grid cell (row by row), and place all shapes in its center.
        # This loop order MUST match the model's output order, otherwise
        # corrections get applied to the wrong anchors.
        for y in range(feature_size):
            for x in range(feature_size):

                # Cell center as a fraction of the image (0 = top/left, 1 = bottom/right).
                center_y = (y + 0.5) / feature_size
                center_x = (x + 0.5) / feature_size

                for ratio, anchor_scale in specs:

                    # Turn size + aspect ratio into height and width.
                    # sqrt keeps the area roughly the same for all ratios.
                    anchor_h = anchor_scale / np.sqrt(ratio)
                    anchor_w = anchor_scale * np.sqrt(ratio)

                    anchors.append([
                        center_y,
                        center_x,
                        anchor_h,
                        anchor_w
                    ])

    anchors = np.array(anchors, dtype=np.float32)

    # compare with box predictions (1917, must be equal)
    print("[OBJECT DETECTION] Anchors: ", len(anchors))

    return anchors

# ==========================================
# RUNS ONCE WHEN THE FILE IS IMPORTED IN MAIN
# ==========================================

labels = load_labels(LABEL_PATH)                 # step 1
net = cv2.dnn.readNetFromONNX(MODEL_PATH)        # step 2
anchors = build_anchors(FEATURE_MAPS, SCALES)    # step 3


# ==========================================
# PER-IMAGE HELPERS
# ==========================================

def reorder_outputs(outputs):
    # We collect the results of all 6 stages in two lists, then glue them together.
    # all_boxes: rows of 4 correction numbers, all_scores: rows of class scores.
    all_boxes = []
    all_scores = []

    for i in range(6):

        # Assumption: outputs come in pairs (boxes first, classes second).
        # [0] removes the batch dimension (we only have one image).
        box_output = outputs[i * 2][0]
        class_output = outputs[i * 2 + 1][0]

        # Shape is (channels, grid_h, grid_w). The channels hold the numbers
        # of all anchors per grid cell: anchors * 4 for boxes, anchors * classes for scores.
        box_channels = box_output.shape[0]
        class_channels = class_output.shape[0]

        # Derive how many anchors sit on each grid cell and how many classes exist.
        num_anchors = box_channels // 4
        num_classes = class_channels // num_anchors

        # Move channels to the end, so we get (grid_h, grid_w, channels),
        # then cut the channels into groups of 4 (one group per anchor).
        # Result: one row of 4 numbers for every anchor of every grid cell.
        box_values = box_output.transpose(1, 2, 0)
        box_values = box_values.reshape(-1, num_anchors, 4)

        # Same idea for the class scores: one row of 91 numbers per anchor.
        class_values = class_output.transpose(1, 2, 0)
        class_values = class_values.reshape(-1, num_anchors, num_classes)

        # Flatten to a plain table (one row per anchor) and store it.
        all_boxes.append(box_values.reshape(-1, 4))
        all_scores.append(class_values.reshape(-1, num_classes))

    # Glue the 6 stages together: one big table with 1917 rows,
    # row n belongs to anchor n (same order as the anchors built below).
    box_predictions = np.concatenate(all_boxes)
    class_predictions = np.concatenate(all_scores)

    # compare with anchor length (1917, must be equal)
    print("[OBJECT DETECTION] Predictions: ", len(box_predictions))

    return (box_predictions, class_predictions)


def decode_boxes(box_predictions, anchors):
    # Move the center: correction / 10 * anchor size + anchor center.
    # The divisor 10 is a fixed constant of this model type (training convention).
    y_center = (
        box_predictions[:, 0] / 10.0 * anchors[:, 2]
        + anchors[:, 0]
    )

    x_center = (
        box_predictions[:, 1] / 10.0 * anchors[:, 3]
        + anchors[:, 1]
    )

    # Resize: exp(correction / 5) * anchor size.
    # exp makes sure the size is never negative, 5 is a fixed constant.
    box_h = np.exp(
        box_predictions[:, 2] / 5.0
    ) * anchors[:, 2]

    box_w = np.exp(
        box_predictions[:, 3] / 5.0
    ) * anchors[:, 3]

    # Convert "center + size" into corners: [x1, y1, x2, y2] (relative 0-1).
    # Corners are what we need later for drawing and for pixel coordinates.
    boxes = np.stack([
        x_center - box_w / 2,
        y_center - box_h / 2,
        x_center + box_w / 2,
        y_center + box_h / 2
    ], axis=1)

    # Boxes can stick out of the image, so cut them off at the borders 0 and 1.
    boxes = np.clip(boxes, 0.0, 1.0)

    return (boxes)
    

def to_probabilities(class_predictions):
    # The raw class numbers (logits) can be any value, also negative.
    # Sigmoid squeezes them into 0-1, so they can be read as probabilities.
    scores = 1.0 / (1.0 + np.exp(-class_predictions))

    # Debug: show the 15 classes with the highest score anywhere in the image.
    # Useful to see what the model "thinks" even below the threshold.
    print ("\n[OBJECT DETECTION]")
    print("--- TOP CLASS SCORES ---")

    top_indices = np.argsort(scores.max(axis=0))[::-1][:15]

    for class_id in top_indices:
        print(
            labels.get(class_id, f"class_{class_id}"),
            round(float(scores[:, class_id].max()), 4)
        )

    return scores


def filter_and_nms(boxes, scores, width, height):
    detections = []

    # Go through every class separately. Start at 1 because class 0 is "background".
    for class_id in range(1, scores.shape[1]):

        class_scores = scores[:, class_id]

        # Row numbers of all anchors that are confident enough for this class.
        selected = np.where(
            class_scores >= CONFIDENCE_THRESHOLD
        )[0]

        # Nothing found for this class, continue with the next one.
        if len(selected) == 0:
            continue

        class_boxes = boxes[selected]

        # Convert relative boxes (0-1) into pixel boxes of the ORIGINAL image.
        # Because the image was just squashed to 300x300, we can simply multiply
        # with the original width / height. Format: [x, y, width, height].
        pixel_boxes = []

        for box in class_boxes:
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)

            pixel_boxes.append([
                x1, y1,
                max(1, x2 - x1),
                max(1, y2 - y1)
            ])

        # NMS: many anchors hit the same object. From each group of strongly
        # overlapping boxes only the one with the best score is kept.
        # Returns the positions (in pixel_boxes) of the boxes that survive.
        indices = cv2.dnn.NMSBoxes(
            pixel_boxes,
            class_scores[selected].tolist(),
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD
        )

        # Store each surviving box with its class and score.
        # selected[index] maps back to the row in the big score table.
        for index in np.array(indices).flatten():

            box_index = selected[index]

            detections.append({
                "label": labels.get(class_id, f"class_{class_id}"),
                "class_id": class_id,
                "score": float(class_scores[box_index]),
                "bounding_box": pixel_boxes[index]
            })

    return (detections)


# ==========================================
# MAIN FUNCTION 
# ==========================================

def detect_objects(img):
    if img is None:
        print("[OBJECT DETECTION] [ERR] No image given.")
        return []

    # save og img dimensions
    height, width = img.shape[:2]

    # blobFromImage turns the image into the exact format the model expects.
    # Resize to 300x300 (squashed, no cropping), pixel values (255) (p - 127.5) / 127.5
    # so they lie between -1 and 1, and swapRB because OpenCV uses BGR, model uses RGB.
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1.0 / 127.5,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False
    )

    # Give the blob to the network and run it (this is the slow part).
    # The model has 12 outputs: for each of the 6 stages one for box
    # corrections and one for class scores. We ask for all of them (basically like a big table).
    # corrections: 4 Zahlen: Wie muss ich diese Schablone verschieben und strecken, damit sie zum Objekt passt?
    # scores: 91 Zahlen: Wie sicher bin ich für jede der Klassen (Person, Tasse, Stuhl, ...)? Klasse 0 ist "Hintergrund".
    net.setInput(blob)

    output_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_names)

    box_predictions, class_predictions = reorder_outputs(outputs)
    boxes = decode_boxes(box_predictions, anchors)
    scores = to_probabilities(class_predictions)
    detections =  filter_and_nms(boxes, scores, width, height)

    print ("\n[OBJECT DETECTION]")
    print("Detections:", len(detections))

    for detection in detections:
        label = detection["label"]

        print(
            label,
            round(detection["score"], 3),
            detection["bounding_box"]
        )

    return detections


# ==========================================
# OPTIONAL: DRAWING (step 11)
# ==========================================

def draw_detections(img, detections):
    detection_img = img.copy()
    
    # Draw a white rectangle and a "label: score" text for each detection.
    # The text is placed above the box, but never outside the top of the image.
    for detection in detections:

        class_id = detection["class_id"]
        score = detection["score"]
        x, y, w, h = detection["bounding_box"]

        label = detection["label"]

        cv2.rectangle(
            detection_img,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            1
        )

        text = f"{label}: {score:.2f}"

        cv2.putText(
            detection_img,
            text,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    return detection_img
