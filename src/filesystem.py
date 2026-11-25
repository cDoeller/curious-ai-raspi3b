import cv2

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

def get_today_date():
    now_berlin = datetime.now(ZoneInfo("Europe/Berlin"))
    return (now_berlin.strftime("%Y-%m-%d"))

def get_now_timestamp():
    now_berlin = datetime.now(ZoneInfo("Europe/Berlin"))
    return (now_berlin.strftime("%Y-%m-%d-%H-%M-%S"))

def make_today_folder():
    folder = Path(__file__).resolve().parent.parent / "media" / get_today_date()
    folder.mkdir(parents=True,exist_ok=True)
    return folder

def save_image_in_archive(frame):
    filename = str(make_today_folder()) + "/" + get_now_timestamp()

    success = cv2.imwrite(filename + ".jpg", frame)

    if (success):
        print ("[FILE SYSTEM] image saved in archive.")
        return filename
    else:
        print ("[FILE SYSTEM] [ERR] error saving image.")


if __name__ == "__main__":
    print(get_now_timestamp())
    print(get_today_date())
    print (make_today_folder())