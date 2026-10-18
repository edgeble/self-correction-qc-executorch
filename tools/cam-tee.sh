#!/bin/bash
# one reader of the USB camera on the host, served as MJPEG over TCP to every pane (the camera allows one reader)
#   cam-tee.sh start|stop|status      port 9000, 1920x1080 MJPG 30 fps
case "$1" in
stop) pkill -f "[c]am-tee-gst"; echo stopped ;;
status) pgrep -af "[c]am-tee-gst" >/dev/null && echo "cam-tee running" || echo "cam-tee NOT running" ;;
*) pkill -f "[c]am-tee-gst" 2>/dev/null; sleep 0.5
   setsid nohup bash -c 'exec -a cam-tee-gst gst-launch-1.0 -q v4l2src device=/dev/video0 ! image/jpeg,width=1920,height=1080,framerate=30/1 ! queue leaky=downstream max-size-buffers=4 ! multipartmux ! tcpserversink host=127.0.0.1 port=9000 sync=false' > /home/arduino/jagan/cam-tee.log 2>&1 < /dev/null &
   sleep 2; $0 status ;;
esac
