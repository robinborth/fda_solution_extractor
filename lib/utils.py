import cv2
from tqdm import tqdm
from PIL import Image


def frame_generator(
    video_path: str,
    sample_rate: int = 25,
    max_frames: int | None = None,
):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # algorithm
    success = cap.grab()  # get the next frame
    fno = 0
    with tqdm(total=total_frames) as pbar:
        while success:
            if fno % sample_rate == 0:
                _, img = cap.retrieve()
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                yield frame
            success = cap.grab()
            pbar.update(1)
            fno += 1
            if max_frames is not None and (fno // sample_rate) > max_frames - 1:
                break


def slice_frame(
    frame,
    *,
    left: int = 35,
    right: int = 1305,
    top: int = 250,
    bottom: int = 1890,
):
    return frame[top:bottom, left:right, :]


def show_frame(frame):
    img = Image.fromarray(frame, "RGB")
    return img
