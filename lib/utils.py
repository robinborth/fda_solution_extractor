import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
from typing import Generator


def frame_generator(
    video_path: str,
    sample_rate: int = 25,
    max_frames: int | None = None,
) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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


def get_random_frame(video_path: str) -> Image:
    return list(
        frame_generator(
            video_path=video_path,
            sample_rate=1000,
            max_frames=2,
        )
    )[-1]


def slice_frame_generator(
    frame: np.ndarray,
    *,
    pleft: int = 0,
    pright: int = 0,
    ptop: int = 0,
    pbottom: int = 0,
):
    height, width, _ = frame.shape
    return frame[ptop : height - pbottom, pleft : width - pright, :]


def get_image(frame):
    return Image.fromarray(frame, "RGB")


def save_frames(slides: list[np.ndarray], path: str = "out.pdf") -> None:
    images = []
    for slide in slides:
        img = get_image(slide)
        images.append(img)
    images[0].save(path, save_all=True, append_images=images[1:])
