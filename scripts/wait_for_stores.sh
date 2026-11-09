#!/usr/bin/env bash
# Service containers start without health checks because the images lack a common probe tool.
set -euo pipefail
for url in http://localhost:6333/readyz http://localhost:8080/v1/.well-known/ready "http://localhost:9200/_cluster/health?wait_for_status=yellow&timeout=5s"; do
  for i in $(seq 1 90); do
    curl -fs "$url" > /dev/null && { echo "ready: $url"; break; }
    [ "$i" = 90 ] && { echo "timeout: $url"; exit 1; }
    sleep 2
  done
done
