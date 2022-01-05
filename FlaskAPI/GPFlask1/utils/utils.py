import os
import shutil
import cv2


def get_record_path(main_root, sub_root):
    return os.path.join(main_root, sub_root)


def clear_folder(folder_path, exception_file=""):
    for f in os.listdir(folder_path):
        if f in exception_file:
            continue
        os.remove(os.path.join(folder_path, f))


def copy_file(source, destination):
    shutil.copy(source, destination)


def video2array(file_path):
    cap = cv2.VideoCapture(file_path)
    len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        for i in range(0, len_frames):
            _, frame = cap.read()
    except Exception as e:
        print(e)
    finally:
        cap.release()
    return
