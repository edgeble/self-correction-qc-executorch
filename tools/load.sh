#!/bin/bash
# one line per second: Hexagon (q6 / HVX / HMX), CPU, RAM, temperature, running engine instances
qcnpuperf_cli 2>/dev/null | while read -r line; do
  npu=$(echo "$line" | sed -E 's/.*q6_utilization=([0-9.]+).*hvx_utilization=([0-9.]+).*hmx_utilization=([0-9.]+).*/q6 \1%  hvx \2%  hmx \3%/')
  cpu=$(top -bn1 | awk '/^%Cpu/{printf "%.0f%%", 100-$8}')
  mem=$(free -m | awk '/Mem/{printf "%d/%d MB", $3, $2}')
  t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -n 1 | awk '{printf "%.0f C", $1/1000}')
  n=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c edgeble-infer)
  printf '\r\033[K%s  NPU: %s   CPU: %s   RAM: %s   %s   instances: %s' "$(date +%T)" "$npu" "$cpu" "$mem" "$t" "$n"
done
