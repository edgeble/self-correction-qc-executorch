#!/bin/bash
# keep one MJPEG viewer open per pane so the engine copies frames (the server is idle without a viewer)
# and the page's /snapshot polls get live frames.   stream-keeper.sh start|stop
case "$1" in
stop) pkill -f "stream-keeper-viewer"; echo stopped ;;
*) pkill -f "stream-keeper-viewer" 2>/dev/null
   for p in 8080 8090 8100 8110; do
     setsid nohup bash -c "exec -a stream-keeper-viewer-$p bash -c 'while true; do curl -s -N -m 86400 http://localhost:$p/ > /dev/null; sleep 2; done'" > /dev/null 2>&1 < /dev/null &
   done; echo keepers started ;;
esac
