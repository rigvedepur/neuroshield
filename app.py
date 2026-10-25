import tempfile
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from python_model import pose_frame

st.set_page_config(page_title="NeuroShield", page_icon="🧠", layout="wide")

WINDOW_SEC = 3.0          # analysis window
BAND = (2.0, 12.0)        # Hz range typical of convulsive rhythmic movement


class SeizureAnalyzer:
    """Heuristic: sustained, rhythmic, high-energy motion in the 2-12 Hz band."""

    def __init__(self, fps=30.0, motion_thresh=1.5, rhythm_thresh=0.5):
        self.fps = fps
        self.motion_thresh = motion_thresh
        self.rhythm_thresh = rhythm_thresh
        self.prev = None
        self.energy = deque(maxlen=int(fps * WINDOW_SEC))

    def update(self, frame_bgr):
        gray = cv2.GaussianBlur(
            cv2.cvtColor(cv2.resize(frame_bgr, (160, 120)), cv2.COLOR_BGR2GRAY), (5, 5), 0
        )
        score, flag = 0.0, False
        if self.prev is not None:
            self.energy.append(float(np.mean(cv2.absdiff(gray, self.prev))))
        self.prev = gray
        if len(self.energy) >= self.energy.maxlen:
            e = np.array(self.energy)
            e = e - e.mean()
            spec = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
            freqs = np.fft.rfftfreq(len(e), 1 / self.fps)
            total = spec[1:].sum() + 1e-9
            in_band = spec[(freqs >= BAND[0]) & (freqs <= BAND[1])].sum()
            score = float(in_band / total)
            flag = np.mean(self.energy) > self.motion_thresh and score > self.rhythm_thresh
        return score, flag


def annotate(img, flag):
    color = (0, 0, 255) if flag else (0, 180, 0)
    text = "ALERT: possible seizure activity" if flag else "Monitoring"
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def read_frames(cap):
    """Yield BGR frames from an opened cv2.VideoCapture."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


class LiveProcessor:
    def __init__(self):
        self.analyzer = SeizureAnalyzer(fps=30.0)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        _, flag = self.analyzer.update(img)
        img = pose_frame(img)
        return av.VideoFrame.from_ndarray(annotate(img, flag), format="bgr24")


st.title("🧠 NeuroShield")

mode = st.radio(
    "Input source",
    ["Live camera", "Upload video"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "Live camera":
    webrtc_streamer(
        key="neuroshield-live",
        video_processor_factory=LiveProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1920, "min": 1280},
                "height": {"ideal": 1080, "min": 720},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },
        video_html_attrs={"style": {"width": "100%"}, "controls": False, "autoPlay": True},
        async_processing=True,
    )
else:
    up = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if up and st.button("Analyze video", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(up.read())
        cap = cv2.VideoCapture(tmp.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        an = SeizureAnalyzer(fps)
        view, bar, status = st.empty(), st.progress(0), st.empty()
        flags = []
        for i, frame in enumerate(read_frames(cap)):
            _, flag = an.update(frame)
            if flag:
                flags.append(i / fps)
            if i % 3 == 0:
                shown = annotate(pose_frame(frame), flag)
                view.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
            bar.progress(min(i / total, 1.0))
        cap.release()
        bar.progress(1.0)
        if flags:
            status.error(
                f"Possible seizure-like activity detected between "
                f"{flags[0]:.1f}s and {flags[-1]:.1f}s ({len(flags)} flagged frames)."
            )
        else:
            status.success("No seizure-like activity detected.")
