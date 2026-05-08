#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  SYSCOM — Vehicle Detection System
#  Starts the Flask server with RTSP→HTTP conversion + YOLO/SAHI
# ═══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════"
echo "  SYSCOM Vehicle Detection — Startup"
echo "════════════════════════════════════════════"

# ── Check FFmpeg ─────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    echo "[OK] FFmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "[WARN] FFmpeg NOT found. Install it for better RTSP support:"
    echo "       Ubuntu/Debian: sudo apt install ffmpeg"
    echo "       macOS:         brew install ffmpeg"
    echo "       The server will fall back to OpenCV-RTSP reader."
fi

# ── Check Python deps ─────────────────────────────────────────
echo ""
echo "Checking Python dependencies..."
pip install -q -r requirements.txt --break-system-packages 2>/dev/null || \
pip install -q -r requirements.txt 2>/dev/null || true

echo ""
echo "Starting server on http://localhost:5000"
echo "Open your browser at: http://localhost:5000"
echo "RTSP source: rtsp://admin:***@169.254.18.91:554/ISAPI/Streaming/channels/1"
echo "════════════════════════════════════════════"
echo ""

python3 server.py
