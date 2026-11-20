'''
white balance = fixed (4500K)
exposure = fixed (150ms)
gain = variable, fixed after calibration
'''

import subprocess
import cv2
import numpy as np

DEV = "/dev/video0"


# Automatically collects all arguments from the function call
# in a Python dictionary (dict) named ctrls
# ctrls = {“brightness”: 120, ‘contrast’: 50, “auto_exposure”: 1}
# f = f-string, formatted string, replaces variables directly in the string
def v4l2_set(**ctrls):
    args = ",".join(f"{k}={v}" for k, v in ctrls.items())
    subprocess.run(["v4l2-ctl", "-d", DEV, f"--set-ctrl={args}"],
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


def calibrate_gain(camera, target_mean=120.0, exposure=150, wb_temp=4500,
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
        print(f"WARNUNG: Selbst bei Gain {gain_min} ist das Bild zu hell "
              f"(mean={mean:.1f}). Szene abdunkeln oder brightness/gamma senken.")
    if best == gain_max and mean < target_mean - 10:
        print(f"WARNUNG: Selbst bei Gain {gain_max} ist das Bild zu dunkel "
              f"(mean={mean:.1f}). Mehr Licht nötig.")

    return {"gain": best, "mean": mean, "clipped": clipped}

def capture_image (camera, filename):
    success, frame = camera.read()
 
    if success:
        cv2.imwrite(filename, frame)
        print("SUCCESS: Image saved")
        print("Image shape:", frame.shape)
        return 1
    else:
        print("ERROR: Could not capture image") 
        return 0

def skip_frames (skip_amount=10):
    for _ in range (skip_amount):
        camera.read()

def capture_n_images (camera, img_amount=10, step=1):
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # small buffer allows for most recent frame capture

    frames = []
    n_failed = 0

    for _ in range (img_amount):
        # timing: caputre every step-frame (step=1 means every frame)
        for _ in range(step - 1):
            if not camera.grab():
                print ("[CAMERA] [ERR] Kamera liefert keine Frames")
            
        success, frame = camera.read()

        if success:
            frames.append(frame)
            print ("[CAMERA] image capture in sequence success.")
        else:
            frames.append(None)
            n_failed += 1
            print ("[CAMERA] [ERR] image capture in image sequence failed.")

    if n_failed:
        print(f"[CAMERA] [ERR] {n_failed}/{img_amount} Frames fehlgeschlagen.")
    else:
        print(f"[CAMERA] {img_amount} Frames erfolgreich aufgenommen.")

    return frames


if __name__ == "__main__":
    camera = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)

    # calculate gain
    calibration = calibrate_gain(camera, target_mean=120)
    print(calibration)

    # set gain
    v4l2_set(gain=calibration.get("gain"))

    # skip frames to adjust
    skip_frames()

    # capture n images every x frame
    frame_sequence = capture_n_images (camera, step=2)

    # save test image last
    filename = "/img-tst/image_seq_test__last_of_ten"
    cv2.imwrite(filename, frame_sequence[-1])


    # (kamera setzt hiernach evtl. zurück)
    camera.release()