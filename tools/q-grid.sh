#!/bin/bash
# the 2x2 demo on the Q: four edgeble-infer instances on one clip (or the camera), one per pane
#   q-grid.sh start [clip|usb]     a: model as installed      b: + judge (correction library, bundle ~/jagan/bundle)
#                                  c: swapped head (A.pte + B.ptd)   d: judge + swapped head
#   q-grid.sh stop | status | reset [clip|usb]   (reset = clear the GO/NG counters and start again)
H=/home/arduino/jagan; IN=${2:-$H/video/demo-mixed.mkv}
[ "$IN" = usb ] && IN=cam   # usb/cam = the camera through tools/cam-tee.sh (one reader, every pane gets the frames)
SEALED=$H/model/9v-bat-2_QCS8275_8a8w.pte; PAIR=$H/model/9v-bat-2_QCS8275_T1-A.pte
case "$1" in
start)
  [ "$IN" = cam ] && $H/tools/cam-tee.sh start
  for n in a b c d; do docker rm -f edgeble-infer-$n >/dev/null 2>&1; done
  NAME=a PORT=8080 MODEL=$SEALED nohup $H/q-run.sh "$IN" -e EDGEBLE_RESIN_GHOST=0 > $H/run-grid-a.log 2>&1 &
  NAME=b PORT=8090 MODEL=$SEALED nohup $H/q-run.sh "$IN" -e EDGEBLE_RESIN_GHOST=0 -e EDGEBLE_GT=/gt -v $H/bundle:/gt > $H/run-grid-b.log 2>&1 &
  NAME=c PORT=8100 MODEL=$PAIR   nohup $H/q-run.sh "$IN" -e EDGEBLE_RESIN_GHOST=0 > $H/run-grid-c.log 2>&1 &
  NAME=d PORT=8110 MODEL=$PAIR   nohup $H/q-run.sh "$IN" -e EDGEBLE_RESIN_GHOST=0 -e EDGEBLE_GT=/gt -v $H/bundle:/gt > $H/run-grid-d.log 2>&1 &
  sleep 20; $H/tools/stream-keeper.sh start; $0 status ;;
reset) $0 stop >/dev/null; docker run --rm -v $H:/j edgeble-infer-q:0.4 bash -c "rm -rf /j/reports-a/* /j/reports-b/* /j/reports-c/* /j/reports-d/*"; echo "counters cleared"; $0 start "$IN" ;;
stop) for n in a b c d; do docker rm -f edgeble-infer-$n >/dev/null 2>&1; done; $H/tools/stream-keeper.sh stop; echo stopped ;;
status) for n in a b c d; do echo "$n: $(docker ps --filter name=edgeble-infer-$n --format '{{.Status}}') $(grep -a '^fps' $H/run-grid-$n.log 2>/dev/null | tail -n 1)"; done; uptime | sed 's/.*load/load/' ;;
*) sed -n 2,5p "$0" ;;
esac
