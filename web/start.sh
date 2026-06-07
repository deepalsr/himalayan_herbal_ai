#!/bin/bash

# Himalayan Herbal AI - Web Interface Startup Script
# ===================================================

set -e

PROJECT_DIR="/Users/dipalshrestha/himalayan-herbal-ai"
cd "$PROJECT_DIR"

echo "================================"
echo "Himalayan Herbal AI - Web Setup"
echo "================================"
echo ""

# Check if main API is running
echo "1. Checking main API server (port 8000)..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "   ✓ Main API is running"
else
    echo "   ✗ Main API is NOT running"
    echo "   Starting main API..."
    export KMP_DUPLICATE_LIB_OK=TRUE
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &
    sleep 3
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "   ✓ Main API started successfully"
    else
        echo "   ✗ Failed to start main API"
        exit 1
    fi
fi

echo ""
echo "2. Starting API server (serves frontend + API on port 8000)..."

cd "$PROJECT_DIR"

export KMP_DUPLICATE_LIB_OK=TRUE
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/api.log 2>&1 &
API_PID=$!
sleep 3

if kill -0 $API_PID 2>/dev/null; then
    echo "   ✓ Server started (PID: $API_PID)"
else
    echo "   ✗ Failed to start server"
    cat /tmp/api.log
    exit 1
fi

echo ""
echo "================================"
echo "✓ Ready!"
echo "================================"
echo ""
echo "App:  http://localhost:8000"
echo "API:  http://localhost:8000/docs"
echo ""
echo "Logs: /tmp/api.log"