#!/bin/bash

PORT=${1:-8080}

echo "🚀 Starting Pixelcrush web server on port $PORT..."
echo ""
echo "   Open http://localhost:$PORT/web/ in your browser"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

# Try python3 first, then python
if command -v python3 &> /dev/null; then
    python3 -m http.server "$PORT"
elif command -v python &> /dev/null; then
    python -m http.server "$PORT"
else
    echo "❌ Python not found. Please install Python or use another HTTP server."
    echo "   Alternative: npx serve . -p $PORT"
    exit 1
fi

