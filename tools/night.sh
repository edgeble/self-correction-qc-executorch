#!/bin/bash
# the night rule on the Q: night.sh on  = two strips off + camera on automatic exposure (false positives)
#                          night.sh off = all strips on + the camera back to the day lock (166 / 4600 K)
T=/home/arduino/jagan/tools; CAM=/dev/video0
case "$1" in
on)  $T/lights.sh 1 | tail -n 1; v4l2-ctl -d $CAM -c auto_exposure=3 -c white_balance_automatic=1 2>/dev/null || docker run --rm --device $CAM edgeble-infer-q:0.3 v4l2-ctl -d $CAM -c auto_exposure=3 -c white_balance_automatic=1
     echo "night: one strip, camera auto" ;;
off) $T/lights.sh 3 | tail -n 1; v4l2-ctl -d $CAM -c auto_exposure=1 -c exposure_time_absolute=166 -c white_balance_automatic=0 -c white_balance_temperature=4600 2>/dev/null || docker run --rm --device $CAM edgeble-infer-q:0.3 v4l2-ctl -d $CAM -c auto_exposure=1 -c exposure_time_absolute=166 -c white_balance_automatic=0 -c white_balance_temperature=4600
     echo "day: all strips, camera locked" ;;
*) sed -n 2,3p "$0" ;;
esac
