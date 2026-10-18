# VENTUNO Q notes (2026-10-18, hotel, SF)

- Dragonwing IQ8 / QCS8275 "Monaco Monza", Ubuntu 24.04.4, kernel 6.8.0-1078-qcom, 8 cores, 14 GB, 55 GB eMMC.
- Hexagon **V75** (qnn-platform-validator unit test PASSED with V75 stub + unsigned skel from QAIRT 2.37.1). QNN SoC model 82 (QCS8300 family in QnnTypes.h; QCS8275 not in the 2.37 SDK). ExecuTorch 1.5.1 lacks the chipset: patched image `executorch-x86:1.5.1-q` adds QCS8275 = 82 / V75 / vtcm 8 (qc_schema.py, utils.py maps, qc_compiler_spec.fbs).
- Access: ADB over USB-C (power on the 12 V barrel jack), user `arduino` (groups fastrpc, docker, gpiod, video, dialout, netdev), **no sudo password known** (set by the track team; saved Wi-Fi "Founders Inc-Guest"). No compiler, no cmake, no network. Docker daemon usable by the user -> edgeble-infer ships as an arm64 image (`edgeble-infer-q`), run script `~/jagan/q-run.sh`.
- HDMI flaps on the hotel TV (login screen on/off every second); a desktop session needs the password. Picture for the demo: edgeble-infer stream `:8080` via `adb forward tcp:8080 tcp:8080` -> http://localhost:8080 on the laptop.
- Files on the Q: `~/jagan/qnn-libs` (libQnnHtp, libQnnSystem, V75 stub+skel, HtpPrepare, calculator stub/skel, libcdsprpc/libadsprpc symlinks), `~/jagan/model/9v-bat-2_QCS8275_8a8w.pte` + labels, `~/jagan/video/`, `~/jagan/bin/qnn-platform-validator`.
- Devices: /dev/fastrpc-cdsp (NPU), /dev/dri, /dev/video32-33 (no USB cam yet), /dev/ttyACM0 (MCU), /dev/ttyHS1 = arduino-router <-> STM32 bridge (MessagePack RPC).
- GPIO for the stepper: 40-pin RPi header is driven by the SoC through 1.8->3.3 V level translators, Linux `gpiochip2` (f100000.pinctrl, 134 lines), line number = MD_GPIO number (user button = GPIO_79 -> `gpiomon gpiochip2 79`). Free GPIO pins: 7=83, 13=109, 15=90, 16=105, 18=106, 22=107, 26=88, 29=89, 31=80, 32=77, 33=81, 37=108. Avoid 8/10/11/36 (Bluetooth UART). 3V3 on pins 1/17, 5V on 2/4, GND 6/9/14/20/25/30/34/39. Full pinout: https://docs.arduino.cc/resources/pinouts/ABX00181-full-pinout.pdf
- RK3588 motor controller (lib/io/resin_motorlight_controller.cpp): gpiochip3 DIR line 1 low, STEP line 2, half-cycle 4000 us default (EDGEBLE_RESIN_STEP_US); Modbus RTU outputs on /dev/ttyACM0 (USB RS485). On the Q: STEP/DIR on two header GPIO lines of gpiochip2, the RS485 adapter on a USB-A port (will enumerate as ttyACM1 or ttyUSB0 next to the MCU's ttyACM0).

## 2026-10-18 evening: runtime vs SoC

- QNN 2.37.1 host runtime reports the device as socModel 82 (QCS8300), arch V75, vtcm 8 MB, dlbc, signed PD (probe `tools/qsoc.cpp`: deviceGetPlatformInfo). But the 2.37.1 **DSP side** refuses: `Dsp startup: Unsupported SoC model (SnapdragonModel): 82` -> backendCreate fails (4000) for every model, sealed or not. The 2.37.1 strings know the name QCS8275 but the public header has no id for it.
- Exports tried: QCS8275 (=82, V75) context, online-prepare variant, SA8620P (=67): all moot while the backend cannot start.
- Fix in progress: QAIRT **2.50.0** community SDK (public URL in ExecuTorch main docs, `v2.50.0.260828.zip`, ~1 GB) -> its aarch64 host libs + hexagon-v75 DSP libs as `~/jagan/qnn-libs-250`; `install-250.sh` pushes and re-runs the probe. If a 2.37-built .pte is rejected by 2.50, re-export with the 2.50 SDK mounted at /qairt (same image, same script).
- Password: `/usr/local/bin/arduino-passwd` (sudoers NOPASSWD for uid 1000) sets the user's password without the old one; Jagan set one. GNOME keyring still carries the old one (cancel its prompt). HDMI still flaps after login; parked.
- JPhone Wi-Fi activation failed on the board; stays USB-only.

- Resolution (2026-10-18): neither QAIRT 2.37.1 nor 2.50.0 starts a Hexagon context on this board ("Unsupported
  SoC model 82" from the DSP). The QNN 2.30 libraries inside Arduino's `ghcr.io/arduino/app-bricks/ei-qnn-models-runner:0.12.1`
  container do (API 2.30 against ExecuTorch's 2.28 expectation, a warning only). Extract them to `~/jagan/qnn-libs-ei`,
  add real copies of `libcdsprpc.so.1`, `libadsprpc.so.1`, `libdmabufheap.so.0`, and pass `/dev/dma_heap/*` into the
  container. backendCreate, deviceCreate and contextCreate all return 0; the sealed detector runs at 30 fps.
