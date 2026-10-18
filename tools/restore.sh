#!/bin/bash
# restore the wiped scratch state: QAIRT SDK copy, calibration frames, then export 9v-bat-2
S=/tmp/claude-1000/-misc-core-runner-edgeble-os/6aad6421-537d-458b-a86d-e3004c34cb2d/scratchpad
N=/misc/core/runner/edgeble-os/notes/hackthon/qnn
if [ ! -d $S/qairt/lib/x86_64-linux-clang ]; then
  docker create --name qairt-tmp radxazifeng278/qairt-npu-9075:v1.1 >/dev/null 2>&1
  V=$(docker run --rm radxazifeng278/qairt-npu-9075:v1.1 bash -c 'ls /root/qairt' 2>/dev/null | head -1)
  rm -rf $S/qairt; docker cp qairt-tmp:/root/qairt/$V $S/qairt; docker rm qairt-tmp >/dev/null; echo "sdk $V"
fi
[ $(ls $S/calib_nchw 2>/dev/null | wc -l) -ge 70 ] || docker run --rm -u $(id -u):$(id -g) -v $S:/work -v $N:/model:ro executorch-x86:1.5.1 python3 /work/et-work/make_calib.py /work/calib_nchw "/model/video_20261013_131212-raw.mkv" "/model/Video Project 2.mp4"
echo "calib $(ls $S/calib_nchw | wc -l)"
SOC=${1:-QCS9100}
run() { docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -e QNN_SDK_ROOT=/qairt -e LD_LIBRARY_PATH=/qairt/lib/x86_64-linux-clang -v $S:/work -v $S/qairt:/qairt executorch-x86:1.5.1 python3 "$@"; }
run /work/et-work/export_pte.py --pt /work/model/9v-bat-2.pt --out /work/et-work/out/9v-bat-2_${SOC}_8a8w --soc $SOC --calib /work/calib_nchw --act-observer Histogram > $S/et-work/export-9v2-sealed-$SOC.log 2>&1; echo "exit $?" >> $S/et-work/export-9v2-sealed-$SOC.log
echo done
