# Lyrebird live viewer

Windows viewer for a DJI controller running Lyrebird. It receives video through MediaMTX, reads telemetry from the controller on TCP port 8081, and displays horizontal speed (m/s), height above takeoff (m), and notebook processing FPS in the upper-right corner. Optional YOLO or RT-DETR models draw their detected classes on the video.

## 1. Prepare the notebook

Open PowerShell in this repository. Install uv and have the Windows MediaMTX executable available.

```powershell
uv venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

## 2. Connect the controller

Open Lyrebird on the RC, confirm the live camera in Flight Deck, and connect the RC and notebook to the same Wi-Fi. Read the controller IP from its Wi-Fi details. This lab preset publishes the stream named `lb_d00dglmh`. For a different aircraft stream name, update the `stream` setting in `main.py` (and the path in `mediamtx.yml` if launching the relay manually).

## 3. Start the viewer

With MediaMTX closed, run the command below using your executable path and controller IP. The viewer starts the relay. Press **Q** in the video window to quit.

```powershell
.\.venv\Scripts\python.exe main.py --rc 192.168.1.190 --mediamtx "C:/tools/mediamtx.exe"
```

To add detections, append `--model "C:/models/best.pt" --detector rtdetr` or select `--detector yolo`. TensorRT `.engine` files require a compatible NVIDIA GPU and TensorRT runtime. Store reusable paths in `MEDIAMTX_PATH` and `MODEL_PATH` in `main.py`.

If MediaMTX is already running with the included `mediamtx.yml`, use `--video "rtsp://127.0.0.1:8554/lb_d00dglmh"` in place of `--mediamtx`.
