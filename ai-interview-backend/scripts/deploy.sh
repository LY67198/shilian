#!/bin/bash

# 试炼 (MockPilot) production deployment script
# Run from ai-interview-backend/ directory
# Usage: ./scripts/deploy.sh [--skip-frontend]

set -e

SKIP_FRONTEND=false
if [ "$1" = "--skip-frontend" ]; then
    SKIP_FRONTEND=true
fi

COMPOSE_FILES="${COMPOSE_FILES:-"-f docker-compose.yml -f docker-compose.prod.yml"}"
API_PORT="${API_PORT:-8001}"
HEALTH_ENDPOINT="http://localhost:${API_PORT}/api/v1/config/health"
TIMEOUT=120

echo "=== 试炼 (MockPilot) Deployment ==="
echo "Compose files: $COMPOSE_FILES"
echo "API port: $API_PORT"

# -------- Build frontends --------
if [ "$SKIP_FRONTEND" = false ]; then
    echo ""
    echo "--- Building user-facing frontend ---"
    (cd ../ai-interview-frontend && npm install && npm run build)
    echo "User frontend build complete."

    echo ""
    echo "--- Building admin panel ---"
    (cd ../ai-interview-admin && npm install && npm run build)
    echo "Admin panel build complete."
else
    echo "Skipping frontend build (--skip-frontend)."
fi

# -------- Start services --------
echo ""
echo "--- Building and starting containers ---"
docker compose $COMPOSE_FILES up -d --build

echo ""
echo "--- Running database migrations ---"
docker compose $COMPOSE_FILES exec -T app alembic upgrade head

echo ""
echo "--- Container status ---"
docker compose $COMPOSE_FILES ps

# -------- Health check --------
echo ""
echo "--- Health check ---"
timeout=$TIMEOUT
while [ "$timeout" -gt 0 ]; do
    if curl -sf "$HEALTH_ENDPOINT" >/dev/null 2>&1; then
        echo "Application is healthy."
        break
    fi
    sleep 5
    timeout=$((timeout - 5))
    echo "  Waiting... ${timeout}s left"
done

if [ "$timeout" -le 0 ]; then
    echo "ERROR: Application did not become healthy."
    echo "Recent logs:"
    docker compose $COMPOSE_FILES logs --tail=80
    exit 1
fi

echo ""
echo "=== Deployment completed ==="
echo "User frontend:  http://localhost"
echo "Admin panel:    http://localhost/admin"
echo "API (direct):   http://localhost:${API_PORT}"
echo "Swagger:        http://localhost/client/docs"
