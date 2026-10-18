#!/bin/bash
# VENTUNO Q stepper control (A4988 on gpiochip2: DIR line 26 = pin 19, STEP line 27 = pin 23)
#   motor.sh start [half_us]   run continuously (default 8000 us half-cycle = slow)
#   motor.sh faster | slower    change speed by 20 % while running (1000..100000 us)
#   motor.sh speed <half_us>    set an exact speed
#   motor.sh stop | status
D=/home/arduino/jagan/tools; F=/tmp/motor-speed; P=/tmp/motor.pid
case "$1" in
start)  echo ${2:-8000} > $F; [ -f $P ] && kill $(cat $P) 2>/dev/null
        nohup bash -c "while :; do $D/step /dev/gpiochip2 26 27 50 \$(cat $F) 0 >/dev/null || exit; done" >/dev/null 2>&1 &
        echo $! > $P; echo "motor running, half-cycle $(cat $F) us" ;;
stop)   [ -f $P ] && kill $(cat $P) 2>/dev/null; rm -f $P; echo "motor stopped" ;;
faster) s=$(cat $F 2>/dev/null || echo 8000); s=$((s - s/5)); [ $s -lt 1000 ] && s=1000; echo $s > $F; echo "half-cycle $s us" ;;
slower) s=$(cat $F 2>/dev/null || echo 8000); s=$((s + s/5)); [ $s -gt 100000 ] && s=100000; echo $s > $F; echo "half-cycle $s us" ;;
speed)  echo ${2:-8000} > $F; echo "half-cycle $(cat $F) us" ;;
status) [ -f $P ] && kill -0 $(cat $P) 2>/dev/null && echo "running, half-cycle $(cat $F) us" || echo "stopped" ;;
*) sed -n 2,7p "$0" ;;
esac
