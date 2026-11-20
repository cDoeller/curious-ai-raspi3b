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

if __name__ == "__main__":
    print(get_now_timestamp())
    print(get_today_date())
    print (make_today_folder())