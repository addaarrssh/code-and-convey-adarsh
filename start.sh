#!/bin/bash
set -e

# Navigate to the web_app directory where app.py lives
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/web_app"

echo "=========================================================="
echo "🚀 Starting Code & Convey Competition Web Server"
echo "📂 Project root: $DIR"
echo "=========================================================="

# Check Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH."
    exit 1
fi

# Optional: activate virtual environment if it exists
if [ -d "$DIR/venv" ]; then
    echo "⚡ Activating virtual environment..."
    source "$DIR/venv/bin/activate"
fi

# Run Flask server
exec python3 app.py
