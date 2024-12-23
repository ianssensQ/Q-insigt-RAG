#!/bin/bash
set -e

host="$1"
port="$2"

shift 2

echo "Waiting for RabbitMQ at $host:$port..."

while ! python -c "import socket; s = socket.socket(); s.connect(('$host', $port))" 2>/dev/null; do
  echo "RabbitMQ is not ready yet. Retrying in 2 seconds..."
  sleep 2
done

echo "RabbitMQ is ready!"
exec "$@"
