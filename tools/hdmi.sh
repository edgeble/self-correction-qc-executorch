#!/bin/bash
# show the status page on the Q's HDMI desktop session (user arduino, Wayland), all from ssh:
#   hdmi.sh start [panes]   e.g. hdmi.sh start a   or   hdmi.sh start a,b,c,d      hdmi.sh stop
export XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 DISPLAY=:0 HOME=/home/arduino DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
case "$1" in
stop) pkill -f "firefox.*kiosk"; pkill -f status4.py; echo stopped ;;
*) pkill -f status4.py 2>/dev/null; pkill -f status.py 2>/dev/null; sleep 0.5
   setsid nohup python3 /home/arduino/jagan/tools/status4.py >/tmp/status.log 2>&1 < /dev/null &
   gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null; gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null
   gdbus call --session --dest org.gnome.ScreenSaver --object-path /org/gnome/ScreenSaver --method org.gnome.ScreenSaver.SetActive false >/dev/null 2>&1
   pkill -f "firefox.*kiosk" 2>/dev/null; sleep 1
   URL="http://localhost:8082/"; [ -n "$2" ] && URL="http://localhost:8082/?panes=$2"
   setsid nohup firefox --kiosk "$URL" >/tmp/firefox.log 2>&1 < /dev/null &
   sleep 2; echo "status page on :8082 ($URL), firefox kiosk on the HDMI session" ;;
esac
