#!/bin/bash
# VENTUNO Q light strips over the RS485 adapter (/dev/ttyACM1, Modbus RTU slave 1, coils DO1..DO3)
#   lights.sh 3        all on      lights.sh 1   only DO1      lights.sh 0   all off
#   lights.sh a b c    explicit, e.g. lights.sh 0 0 1
case "$#:$1" in
1:3) set -- 1 1 1 ;; 1:2) set -- 1 1 0 ;; 1:1) set -- 1 0 0 ;; 1:0) set -- 0 0 0 ;;
3:*) ;; *) sed -n 2,4p "$0"; exit 2 ;;
esac
docker run --rm --device /dev/ttyACM1 -v /home/arduino/jagan:/j edgeble-infer-q:0.1 /j/tools/coils /dev/ttyACM1 "$1" "$2" "$3"
