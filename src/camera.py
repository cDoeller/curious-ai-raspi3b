import subprocess
import cv2
import numpy as np

from config import (CAMERA_DEVICE, 
                    CAMERA_RES_W, 
                    CAMERA_RES_H, 
                    CAMERA_FPS, 
                    CAMERA_TARGET_BRIGHTNESS_DEF, 
                    CAMREA_FIXED_EXPOSURE_DEF, 
                    CAMERA_WB_TEMP_DEF, 
                    CAMERA_IMG_SEQU_AMT_DEF, 
                    CAMERA_IMG_SEQU_STEP_DEF, 
                    ROUND_DECIMALS)

# TODO Calibration not working properly when sunlight strong. 
# maybe auto exposure enable and save vals, then change analysis blur based on vals?

def initialize_camera():
    camera = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    if not camera.isOpened():
            print(f"[CAMERA] [ERR] no camera device at {CAMERA_DEVICE}.")
            camera.release()
            return None

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RES_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RES_H)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    print("[CAMERA] successfully initialized.")
    return camera

# Automatically collects all arguments from the function call
# in a Python dictionary (dict) named ctrls
# ctrls = {“brightness”: 120, ‘contrast’: 50, “auto_exposure”: 1}
# f = f-string, formatted string, replaces variables directly in the string
def v4l2_set(**ctrls):
    args = ",".join(f"{k}={v}" for k, v in ctrls.items())
    subprocess.run(["v4l2-ctl", "-d", CAMERA_DEVICE, f"--set-ctrl={args}"],
                   check=True, capture_output=True)

# measure mean value of brightness in image, taking n_avg samples
def measure(camera, n_discard=10, n_avg=3):
    for _ in range(n_discard):
        camera.read()
    means, clipped = [], []
    for _ in range(n_avg):
        ok, frame = camera.read()
        if not ok:
            print ("[CAMERA][ERR] No frame reading camera in measure func")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        means.append(gray.mean()) # Calculates an average (mean) of array elements
        clipped.append((gray > 250).mean()) # NumPy-Broadcasting, mean of above 250 vals in array
    return float(np.mean(means)), float(np.mean(clipped))


def calibrate_gain(camera, target_mean=CAMERA_TARGET_BRIGHTNESS_DEF, exposure=CAMREA_FIXED_EXPOSURE_DEF, wb_temp=CAMERA_WB_TEMP_DEF,
                   gain_min=0, gain_max=64, max_clipped=0.02):
    """
    Locks in exposure and white balance, then uses a binary search to find the gain
    whose mean brightness is closest to the target.
    Returns a dict containing gain, mean, and clipped.
    """
    # alles fixe fixieren
    v4l2_set(white_balance_automatic=0,
             white_balance_temperature=wb_temp,
             auto_exposure=1,
             exposure_time_absolute=exposure,
             backlight_compensation=0)

    cache = {}

    def probe(g):
        if g not in cache:
            v4l2_set(gain=g)
            cache[g] = measure(camera) # get means and clipped for gain and save in cache
        return cache[g] # return the values for means and clipped for this gain

    # binary search
    # smallest gain whose average value is >= the target 
    # (brightness increases monotonically with gain)
    lo, hi = gain_min, gain_max
    while lo < hi:
        mid = (lo + hi) // 2
        if probe(mid)[0] < target_mean: # if calc mean of mid gain smaller than target mean
            lo = mid + 1 # next search step: middle to right side
        else:
            hi = mid # otherwise: search continues in left side

    # one step below found lo (lo - 1), to see if it is closer to the target
    candidates = {lo, max(gain_min, lo - 1)} # SET (only once), dont go below allowed min
    best = min(candidates, key=lambda g: abs(probe(g)[0] - target_mean)) # get lowest gain with delta target mean

    # If there are too many burned-out pixels: Lower the gain as much as possible
    while probe(best)[1] > max_clipped and best > gain_min:
        best -= 1

    mean, clipped = probe(best)
    v4l2_set(gain=best)  # set final value

    if best == gain_min and mean > target_mean + 10:
        print("[CAMERA] [WARN] calibration at minimum gain.")
    if best == gain_max and mean < target_mean - 10:
        print("[CAMERA] [WARN] calibration at maximum gain")

    # set gain
    print (f"[CAMERA] setting gain: {best}")
    v4l2_set(gain=best)

    return {"gain": best, 
            "mean_brightness_gray": round(mean, ROUND_DECIMALS), 
            "target_brightness_gray": target_mean, 
            "clipped": round (clipped, ROUND_DECIMALS),
            "exposure_ms": exposure/100, 
            "wb_temp_K": wb_temp,
            "resolution_w": CAMERA_RES_W,
            "resolution_h": CAMERA_RES_H,
            "fps": CAMERA_FPS,
            }

def skip_frames (camera, skip_amount=10):
    for _ in range (skip_amount):
        camera.read()

def capture_n_images (camera, img_amount=CAMERA_IMG_SEQU_AMT_DEF, step=CAMERA_IMG_SEQU_STEP_DEF):
    # skip n frames to refresh camera
    print ("[CAMERA] skipping frames..")
    skip_frames(camera)

    print (f"[CAMERA] capturing {img_amount} frames..")
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # small buffer allows for most recent frame capture

    frames = []
    n_failed = 0

    for i in range (img_amount):
        # timing: caputre every step-frame (step=1 means every frame)
        for _ in range(step - 1):
            if not camera.grab():
                print ("[CAMERA] [ERR] Kamera liefert keine Frames")
            
        success, frame = camera.read()

        if success:
            frames.append(frame)
            print (f"[CAMERA] image {i} capture in sequence success.")
        else:
            frames.append(None)
            n_failed += 1
            print ("[CAMERA] [ERR] image capture in image sequence failed.")

    if n_failed:
        print(f"[CAMERA] [ERR] {n_failed}/{img_amount} frames failed in sequence.")
    else:
        print(f"[CAMERA] {img_amount} sequence frames successfully captured.")

    return frames, step


if __name__ == "__main__":
    print ("[CAMERA] initializing camera..")
    camera = initialize_camera()

    # calculate gain
    print ("[CAMERA] starting calibration..")
    calibration = calibrate_gain(camera)
    print("[CAMERA] calibration values ", calibration)

    # capture n images every x frame
    print ("[CAMERA] capturing frame sequence..")
    frame_sequence = capture_n_images (camera, step=2)

    # (kamera setzt hiernach evtl. zurück)
    print ("[CAMERA] done.")
    camera.release()