# Venue runbook (VENTUNO Q), in order

## Setup
1. Power: 12 V on the barrel jack. USB-A: the USB camera and the RS485 adapter. HDMI optional (a monitor that holds 1080p60; the hotel TV did not). Stepper on the 40-pin header, odd row: pin 1 3.3 V, pin 19 DIR, pin 23 STEP, pin 39 GND.
2. Network: the Q joins the phone hotspot by itself (saved). Laptop on the same hotspot. The Q's address: `adb shell hostname -I` over USB-C if unknown.
3. Laptop: `ssh arduino@<Q>` works with the laptop key. Tunnel for the browser view:
   `ssh -N -L 8082:localhost:8082 -L 8080:localhost:8080 -L 8090:localhost:8090 -L 8100:localhost:8100 -L 8110:localhost:8110 arduino@<Q>`
   Page: `http://localhost:8082/?panes=a` (one pane) or `?panes=a,b,c,d`.
4. On the Q, once: `~/jagan/tools/hdmi.sh start a` if a monitor is connected (auto-login is on).

## Rig
```
~/jagan/tools/lights.sh 3            # all strips on (finds the RS485 adapter by vendor id)
~/jagan/tools/motor.sh start 8000    # slow; faster / slower / speed <us> / stop
~/jagan/tools/night.sh on|off        # two strips off + camera auto  |  back to day
```

## Inference
```
NAME=a PORT=8080 ~/jagan/q-run.sh usb                               # live camera, sealed model
NAME=a PORT=8080 ~/jagan/q-run.sh ~/jagan/video/demo-mixed.mkv      # the mixed clip instead
NAME=b PORT=8090 ~/jagan/q-run.sh usb -e EDGEBLE_GT=/gt -v /home/arduino/jagan/bundle:/gt   # + judge
NAME=c PORT=8100 MODEL=~/jagan/model/9v-bat-2_QCS8275_T1r-A.pte ~/jagan/q-run.sh usb   # retrained head
~/jagan/tools/q-grid.sh start usb        # all four panes at once: a plain, b judge, c head, d head + judge (one camera, shared by tools/cam-tee.sh)
~/jagan/tools/q-grid.sh reset usb        # same, with the GO/NG counters cleared first
~/jagan/tools/q-grid.sh stop
```
Each NAME is one instance with its own stream port and log: `~/jagan/reports-<NAME>/<date>/logs/results_<date>.log`.
Judge verdicts are on the instance's stdout (`~/jagan/run-<NAME>.log`, lines `[VERIFY] track=...`):
CONFIRMED, AMBIGUOUS (abstain), FLIP->label, LOW-VALIDITY bypass. Stop one: `docker rm -f edgeble-infer-<NAME>`.
The judge bundle is `~/jagan/bundle/` (embedding `.pte`, `embedding-mean.f32`, `reference/bank.evbk`,
`verify.conf` thresholds). Rebuild the bank from a good-light dump (`~/jagan/dump-<NAME>/`, `.rgb` + `.det`):
```
docker run --rm --device /dev/fastrpc-cdsp --device /dev/fastrpc-cdsp-secure $(for d in /dev/dma_heap/*; do printf -- "--device %s " $d; done) \
  --group-add $(getent group fastrpc | cut -d: -f3) -v ~/jagan/delta5.v/edgeble-correction:/opt/edgeble-correction:ro \
  -v ~/jagan/qnn-libs-ei:/qnn-libs:ro -v ~/jagan/bundle:/gt -v ~/jagan/dump-d:/dump:ro \
  -e LD_LIBRARY_PATH=/opt/edgeble-correction/lib:/opt/executorch/lib:/opt/executorch/lib/executorch/backends/qualcomm:/qnn-libs \
  -e EDGEBLE_QNN_LIBS=/qnn-libs -e ADSP_LIBRARY_PATH="/qnn-libs;/usr/lib/rfsa/adsp;/dsp" edgeble-infer-q:0.4 \
  /opt/edgeble-correction/bin/ec-bank /gt /dump 640 640 /gt/coco_80_labels_list.txt /gt/reference/bank.evbk 0-3599,9000-99999 0.6 400 1920 1080
```
(frame ranges = the good-light part of the dump; 1920 1080 = the camera frame the `.det` boxes are in.)
Load: `qcnpuperf_cli` (NPU), `htop`, `docker stats`.

## Pitch sequence (camera)
1. Day: `night.sh off`, good and bad batteries on the ring, pane a: green for good, red for the bad ones.
2. Night rule: `night.sh on`. Good batteries start to come out red on pane a.
3. Pane b (judge): same frames, corrections counted on the pane, genuine mismatches still red.
4. Learn and prove: the retrained head on pane c, logs in `docs/` and the repo `logs/`.

## If something is off
- No picture on a pane: that instance is not running; start it (above). The page polls snapshots, a tunnel drop is harmless.
- Strips do not react: `lights.sh` prints the device it found; the adapter must enumerate as vendor 1a86.
- Motor buzzes without turning: speed too low, use 8000 or faster.
- Camera not found: `v4l2-ctl --list-devices` inside the container; the engine takes `usb` or `/dev/videoN`.
