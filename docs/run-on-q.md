# VENTUNO Q: from .pt to running edgeble-infer (ExecuTorch + Hexagon), step by step

All host steps run on the laptop in docker; the board needs no compiler and no sudo.
Paths: `N` = `notes/hackthon/qnn/executorch`, `S` = the session scratch dir (volatile; `restore.sh` recreates it).

## 1. .pt -> .pte (laptop, x86 container)

1. Image: `executorch-x86:1.5.1-q` (Dockerfile `N/Dockerfile.executorch-x86-qcs8275`: ubuntu 22.04,
   py3.10, torch 2.14 cpu, torchao 0.18, executorch 1.5.1, ultralytics 8.3.156; plus the chipset
   patch QCS8275 = 82 / Hexagon V75 in `qc_schema.py`, `utils.py`, `qc_compiler_spec.fbs`).
2. QAIRT SDK extracted on the laptop and mounted at `/qairt` (x86 AOT libs are used at export).
3. Calibration frames: `python3 make_calib.py <out_dir> <video1> <video2>` -> 70 letterboxed
   640x640 float32 NCHW RGB 0..1 `.raw` files.
4. Sealed export:
   ```
   docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -e QNN_SDK_ROOT=/qairt \
     -e LD_LIBRARY_PATH=/qairt/lib/x86_64-linux-clang -v $S:/work -v $S/qairt:/qairt \
     executorch-x86:1.5.1-q python3 /work/et-work/export_pte.py --pt /work/model/9v-bat-2.pt \
     --out /work/et-work/out/9v-bat-2_QCS8275_8a8w --soc QCS8275 --calib /work/calib_nchw \
     --act-observer Histogram            # add --online-prepare to compile on the device instead
   ```
   Output: `.pte` (int8, one QNN delegate, float32 in/out), `.json` (shapes, names), `.txt` (labels).
   The script prints float / int8-sim / ONNX detections on the first frames: boxes must match
   within a few px, scores within ~0.05.
## 2. ExecuTorch runtime for aarch64 (laptop, once)

`build-aarch64.sh` in the `executorch-x86:1.5.1-cross` image: executorch v1.5.1 source cloned as a
directory named `executorch` (with submodules), cross-compiled with gcc-aarch64-linux-gnu,
`-DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT=/qairt` + module/tensor/data-loader/flat-tensor
extensions + portable/quantized/optimized kernels. Install tree = `lib/*.a`,
`lib/executorch/backends/qualcomm/libqnn_executorch_backend.so`, `include/`,
`lib/cmake/ExecuTorch/`. A copy is inside `notes/hackthon/qnn/q900-et.tgz` (`executorch/`).

## 3. edgeble-infer for the Q (laptop, arm64 container under qemu)

`scratch/qbuild/Dockerfile`: `FROM --platform=linux/arm64 ubuntu:24.04`, apt cmake g++ ninja
libopencv-dev libmodbus-dev libeigen3-dev libjpeg-dev; COPY `qairt/include`, the executorch install,
the edgeble-infer tree (branch `platform-qcom-executorch`); `./build.sh -p qcom -z /opt/qairt -e
/opt/executorch`. Build takes ~25 min emulated. Result image `edgeble-infer-q:0.1` (1.3 GB).
Ship: `docker save edgeble-infer-q:0.1 | gzip -1 > edgeble-infer-q.tgz` (470 MB), `adb push` (or scp),
`docker load -i` on the Q. **To change edgeble-infer**: edit on the laptop, rebuild the image
(only the last layers rebuild), ship again; or edit inside the container on the Q (`docker run -it
edgeble-infer-q:0.1 bash`, the source and toolchain are in `/opt/edgeble-infer`) and `docker commit`.

## 4. Run on the Q

Board side files (`/home/arduino/jagan`): `qnn-libs/` (QNN host libs + V75 stub/skel/DSP libs + rpc
symlinks), `model/` (.pte + .json + coco_80_labels_list.txt), `video/` (case clips), `q-run.sh`.
```
~/jagan/q-run.sh ~/jagan/video/case1-good-lights3.mkv        # a clip
~/jagan/q-run.sh usb                                          # the USB camera
```
`q-run.sh` = `docker run --network host --device /dev/fastrpc-cdsp --device /dev/dri --device
/dev/video* -v qnn-libs:/qnn-libs -v model:/model -v video:/video -v dump:/dump -e LD_LIBRARY_PATH=
<install lib>:/opt/executorch/lib:/opt/executorch/lib/executorch/backends/qualcomm:/qnn-libs
-e EDGEBLE_QNN_LIBS=/qnn-libs -e ADSP_LIBRARY_PATH="/qnn-libs;/usr/lib/rfsa/adsp;/dsp"
edgeble-infer-q:0.1 ./rknn_yolo11_demo /model/<model>.pte <input> VIDEO|USB in:resin:track`.
Picture: `http://<Q ip>:8080` (Wi-Fi) or `adb forward tcp:8080 tcp:8080` -> `http://localhost:8080`.
For the motor/lights add `--device /dev/gpiochip2 --device /dev/ttyACM1 --group-add gpiod --group-add
dialout` and `-e EDGEBLE_RESIN_CONTROLLER=1` (default in q-run.sh is 0 = off).

## 5. Q hardware map

- USB camera: `/dev/video0` (Realtek 0bda:5842 "USB Camera"); edgeble-infer finds it with `usb USB`.
- RS485 adapter (QinHeng 1a86:55d3): `/dev/ttyACM1` (`/dev/ttyACM0` is the board's MCU). The Modbus
  outputs: `EDGEBLE_RESIN_DO_DEVICE=/dev/ttyACM1`, slave 1, 9600 8N1, coils 1..3 (`mbpoll -m rtu -a 1
  -b 9600 -P none -t 0 -r 1 -1 /dev/ttyACM1 1 1 1`).
- Stepper (A4988 STEP/DIR): Linux `gpiochip2` (f100000.pinctrl); header pin -> line: 19 -> 26 (DIR),
  23 -> 27 (STEP); 3.3 V pin 1, GND pin 39. Other free lines: pin 7 -> 83, 13 -> 109, 15 -> 90,
  16 -> 105, 18 -> 106, 22 -> 107, 26 -> 88, 29 -> 89, 31 -> 80, 32 -> 77, 33 -> 81, 37 -> 108.
  `gpioset gpiochip2 27=1`, `gpioget gpiochip2 26`. The RK3588 controller code uses gpiochip3 lines
  1/2: make the chip and lines env-configurable (e.g. `EDGEBLE_RESIN_GPIO=/dev/gpiochip2:26:27`).
- Hexagon: V75, QNN SoC model 82 as the host runtime sees it; QAIRT 2.37.1 DSP libs refuse it,
  2.50.0 under test.
