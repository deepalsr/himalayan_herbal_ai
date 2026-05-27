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
echo "2. Setting up web backend (port 5000)..."

# Install backend dependencies
if [ ! -d "web/backend/venv" ]; then
    echo "   Creating virtual environment..."
    python -m venv web/backend/venv
fi

echo "   Activating virtual environment..."
source web/backend/venv/bin/activate

echo "   Installing dependencies..."
pip install -q -r web/backend/requirements.txt

echo "   ✓ Backend dependencies installed"

echo ""
echo "3. Starting web backend server..."
cd web/backend

# Kill any existing Flask process on port 5000
pkill -f "flask run" || true
sleep 1

python app.py > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
sleep 2

if kill -0 $BACKEND_PID 2>/dev/null; then
    echo "   ✓ Backend server started (PID: $BACKEND_PID)"
else
    echo "   ✗ Failed to start backend server"
    cat /tmp/backend.log
    exit 1
fi

cd "$PROJECT_DIR"

echo ""
echo "================================"
echo "✓ Web Interface Ready!"
echo "================================"
echo ""
echo "Frontend:  http://localhost:5000"
echo "Backend:   http://localhost:5000/api"
echo "Main API:  http://localhost:8000"
echo ""
echo "Open your browser to http://localhost:5000"
echo ""
echo "Logs:"
echo "  - Frontend:  /tmp/backend.log"
echo "  - Main API:  /tmp/api.log"
echo ""
