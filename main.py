"""Display the aircraft video with optional detections and latest RC telemetry."""

import argparse
import json
import math
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from collections import deque
from threading import Thread

import cv2
from ultralytics import RTDETR, YOLO

# Local lab settings. Replace None with the full path to use that file by default.
RC_IP = "192.168.1.190"
MODEL_PATH = None  # e.g. Path("C:/models/fire_rtdetr.engine")
MEDIAMTX_PATH = None  # e.g. Path("C:/tools/mediamtx.exe")
# Example with overrides (PowerShell, from this folder):
# .\.venv\Scripts\python.exe main.py --detector engine --model "C:/models/fire_rtdetr.engine" --rc "192.168.1.190" --mediamtx "C:/tools/mediamtx.exe"

def load_model(detector: str, weights: str):
    return RTDETR(weights) if detector == "rtdetr" else YOLO(weights)


def detector_for(path: Path) -> str:
    name = path.stem.lower().replace("-", "").replace("_", "")
    if "rtdetr" in name:
        return "rtdetr"
    if "yolo" in name:
        return "yolo"
    raise ValueError("Specify --detector yolo or --detector rtdetr for this filename")


def receive(rc: str, latest: deque) -> None:
    while True:
        try:
            with socket.create_connection((rc, 8081), timeout=4) as sock, sock.makefile("rb") as stream:
                for line in stream:
                    try:
                        packet = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if isinstance(packet, dict):
                        latest.append((time.monotonic(), packet))
        except OSError as error:
            print(f"Telemetry: {error}. Retrying...", flush=True)
        latest.clear()
        time.sleep(1)


def reading(value: object, unit: str) -> str:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.1f} {unit}"
    return f"-- {unit}"


def labels(packet: dict) -> tuple[str, str]:
    velocity = packet.get("speed") or {}
    x, y = velocity.get("x"), velocity.get("y")
    speed = math.hypot(x, y) if all(isinstance(v, (int, float)) for v in (x, y)) else None
    return f"Velocidade: {reading(speed, 'm/s')}", f"Altura: {reading(packet.get('altitude'), 'm')}"


def draw_telemetry(frame, packet: dict, fps: float) -> None:
    height, width = frame.shape[:2]
    speed, altitude = labels(packet)
    lines = (speed, altitude, f"FPS: {fps:.1f}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = max(3, min(16, width // 80))
    widest = max(cv2.getTextSize(line, font, 1, 1)[0][0] for line in lines)
    scale = min(1.4, max(0.5, height / 900), (width - 2 * margin - 4) / widest,
                (height - 2 * margin) / (40 * len(lines)))
    if scale <= 0:
        return
    stroke = max(1, round(2 * scale))
    outline = stroke + 2
    (_, text_height), baseline = cv2.getTextSize("Ag", font, scale, outline)
    y = margin + text_height
    for line in lines:
        line_width = cv2.getTextSize(line, font, scale, outline)[0][0]
        position = (width - margin - line_width, y)
        cv2.putText(frame, line, position, font, scale, (0, 0, 0), outline, cv2.LINE_AA)
        cv2.putText(frame, line, position, font, scale, (255, 255, 255), stroke, cv2.LINE_AA)
        y += text_height + baseline + margin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rc", default=RC_IP, help="Device IP shown on the controller")
    parser.add_argument("--video", help="Override the direct DJI RTSP source")
    parser.add_argument(
        "--mediamtx", type=Path, default=MEDIAMTX_PATH, metavar="EXE_PATH",
        help="Path to mediamtx.exe on this PC. The viewer starts it to receive camera video "
             "from the RC. If MediaMTX is already running, use --video with its RTSP URL instead.",
    )
    parser.add_argument("--detector", choices=["yolo", "rtdetr", "engine"])
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to your detection model (.pt or .engine)")
    args = parser.parse_args()
    if args.mediamtx and not args.mediamtx.is_file():
        parser.error("--mediamtx must point to the MediaMTX executable")
    model = None
    if args.model:
        path = Path(args.model)
        if not path.is_file():
            parser.error(f"Model file does not exist: {path}")
        detector = args.detector
        if detector == "engine" and path.suffix.lower() != ".engine":
            parser.error("--detector engine requires a .engine file")
        if detector in (None, "engine"):
            try:
                detector = detector_for(path)
            except ValueError as error:
                parser.error(str(error))
        model = load_model(detector, str(path))
    elif args.detector == "engine":
        parser.error("--detector engine requires --model PATH")
    elif args.detector:
        weights = "yolo11n.pt" if args.detector == "yolo" else "rtdetr-l.pt"
        model = load_model(args.detector, weights)

    url = args.video or f"rtsp://{args.rc}:8554/streaming/live/1"
    latest = deque(maxlen=1)
    cap = cv2.VideoCapture()
    relay = None
    config_dir = None
    try:
        if args.mediamtx:
            stream = "lb_d00dglmh"
            config_dir = tempfile.TemporaryDirectory()
            config = Path(config_dir.name) / "mediamtx.yml"
            config.write_text(
                f"webrtc: yes\nwebrtcAddress: :8889\nrtsp: yes\nrtspAddress: :8554\n"
                f"paths:\n  {stream}:\n    source: publisher\n",
                encoding="utf-8",
            )
            relay = subprocess.Popen([str(args.mediamtx.resolve()), str(config)], cwd=config_dir.name)
            url = args.video or f"rtsp://127.0.0.1:8554/{stream}"
        Thread(target=receive, args=(args.rc, latest), daemon=True).start()
        while True:
            if relay is not None and relay.poll() is not None:
                raise SystemExit("MediaMTX stopped. Check its log above for the cause.")
            if not cap.isOpened():
                cap.open(url, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000,
                ])
            start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cv2.destroyAllWindows()
                print(f"No frame from {url}. Retrying in two seconds (Ctrl+C to stop).", flush=True)
                time.sleep(2)
                continue
            if model is not None:
                frame = model.predict(frame, verbose=False)[0].plot()
            sample = next(iter(latest), None)
            packet = sample[1] if sample and time.monotonic() - sample[0] < 3 else {}
            fps = 1.0 / max(time.perf_counter() - start, 1e-9)
            draw_telemetry(frame, packet, fps)
            cv2.namedWindow("Lyrebird | press Q to quit", cv2.WINDOW_NORMAL)
            cv2.imshow("Lyrebird | press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        if relay is not None:
            relay.terminate()
            relay.wait()
        if config_dir is not None:
            config_dir.cleanup()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
