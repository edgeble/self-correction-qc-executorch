#!/bin/bash
# VENTUNO Q: run edgeble-infer (ExecuTorch + QNN on the Hexagon) from the shipped container image
#   [NAME=a] [PORT=8080] [MODEL=<.pte>] ~/jagan/q-run.sh <video|usb> [more docker args, e.g. -e EDGEBLE_RESIN_GHOST=0]
# One instance per NAME: container edgeble-infer-<NAME>, stream :PORT, dashboard :PORT+1,
# results in ~/jagan/reports-<NAME>/<date>/logs/. Defaults: NAME=a, PORT=8080.
set -e
H=/home/arduino/jagan
IMG=${IMG:-edgeble-infer-q:0.4}
NAME=${NAME:-a}; PORT=${PORT:-8080}
M=${MODEL:-$H/model/9v-bat-2_QCS8275_8a8w.pte}
IN=${1:-$H/video/case1-good-lights3.mkv}; shift || true
docker rm -f edgeble-infer-$NAME >/dev/null 2>&1 || true
mkdir -p $H/reports-$NAME $H/dump-$NAME $H/infer-dash; [ -f $H/infer-dash/ss_flag.txt ] || echo 0 > $H/infer-dash/ss_flag.txt
if [ "$IN" = usb ]; then SRC="usb USB"; else SRC="/video/$(basename "$IN") VIDEO"; fi
exec docker run --name edgeble-infer-$NAME --rm --network host \
  --device /dev/fastrpc-cdsp --device /dev/fastrpc-cdsp-secure --device /dev/dri \
  $(for d in /dev/dma_heap/*; do printf -- "--device %s " $d; done) \
  $(for v in /dev/video*; do echo --device $v; done 2>/dev/null) \
  --group-add $(getent group fastrpc | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  -v $H/qnn-libs-ei:/qnn-libs:ro -v $H/model:/model:ro -v $H/video:/video:ro -v $H/dump-$NAME:/dump \
  -v $H/delta5.v/edgeble-correction:/opt/edgeble-correction:ro -v $H/delta5.v/rknn_yolo11_demo:/opt/edgeble-infer/install/qcom_linux_aarch64/rknn_yolo11_demo \
  -v $H/reports-$NAME:/agent/reports -v $H/infer-dash:/agent/infer/dash \
  -e LD_LIBRARY_PATH=/opt/edgeble-correction/lib:/opt/edgeble-infer/install/qcom_linux_aarch64/rknn_yolo11_demo/lib:/opt/executorch/lib:/opt/executorch/lib/executorch/backends/qualcomm:/qnn-libs \
  -e EDGEBLE_QNN_LIBS=/qnn-libs -e ADSP_LIBRARY_PATH="/qnn-libs;/usr/lib/rfsa/adsp;/dsp" \
  -e EDGEBLE_STREAM_PORT=$PORT -e EDGEBLE_DASH_PORT=$((PORT+1)) \
  -e EDGEBLE_PERF=1 -e EDGEBLE_RUN_LOG=0 -e EDGEBLE_RESIN_CONTROLLER=0 "$@" \
  $IMG ./rknn_yolo11_demo /model/$(basename $M) $SRC in:resin:track
