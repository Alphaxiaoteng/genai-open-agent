#!/usr/bin/env bash
# Enterprise Multi-Agent Collaboration Platform - Demo Script
# 
# Usage:
#   ./scripts/demo.sh                     # offline mode (MockLLM)
#   ./scripts/demo.sh --live              # live mode (OpenAI-compatible provider)
#   ./scripts/demo.sh --workspace PATH    # audit custom directory
#   ./scripts/demo.sh --web               # launch web dashboard + execute squad
#   ./scripts/demo.sh --enterprise        # start enterprise server mode
#
# Expected results:
#   - All agents score >= 0.85 (architecture, security, tests, review)
#   - Blackboard shows 4 namespaces with cross-agent findings
#   - Status: PASS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# Default configuration
MODE="offline"
WORKSPACE=""
VERBOSE=false
LAUNCH_WEB=false
ENTERPRISE_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --live)
            MODE="live"
            shift
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --web)
            LAUNCH_WEB=true
            shift
            ;;
        --enterprise)
            ENTERPRISE_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================================="
echo "Enterprise Multi-Agent Collaboration Platform"
echo "=============================================================="
echo "Mode: $MODE"
[[ -n "$WORKSPACE" ]] && echo "Workspace: $WORKSPACE"
[[ "$LAUNCH_WEB" == true ]] && echo "Dashboard: ENABLED"
echo ""

# Check if uvicorn is installed
if [[ "$LAUNCH_WEB" == true || "$ENTERPRISE_MODE" == true ]]; then
    if ! python3 -c "import uvicorn" 2>/dev/null; then
        echo "Installing required dependencies..."
        pip3 install fastapi uvicorn pydantic websockets > /dev/null 2>&1
        echo "Dependencies installed."
    fi
fi

if [[ "$LAUNCH_WEB" == true || "$ENTERPRISE_MODE" == true ]]; then
    echo "🚀 Starting Enterprise Multi-Agent Platform Server..."
    echo "📊 Dashboard URL: http://localhost:8000"
    echo ""
    
    # Run the Python server that integrates WebSockets with squad execution
    python3 << 'PYTHON_SCRIPT'
import sys
import os
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from examples.demo import run_demo
from squad.cli_web import run_audit_with_web, collect_source_files
from squads.code_audit import build_code_audit_squad, sample_workspace
from squad.llm import build_llm
from squad.tools import build_default_registry
from squad.memory import Blackboard

def main():
    # Build LLM provider
    provider, mode = build_llm()
    
    # Create workspace
    root = tempfile.mkdtemp(prefix="agent-squad-enterprise-")
    paths = sample_workspace(root)
    
    # Collect source files
    paths = collect_source_files(root, include=("py", "md"))
    
    # Build registry and blackboard
    registry = build_default_registry(workdir=root)
    blackboard = Blackboard()
    
    # Build squad
    squad = build_code_audit_squad(blackboard=blackboard, llm=provider, tools=registry, max_rounds=3)
    
    # Event handler for real-time updates
    def on_event(event):
        print(f"[REAL-TIME] {event}")
    
    # Run audit with enhanced event handling
    result = run_audit_with_web(paths, squad=squad, blackboard=blackboard, on_event=on_event)
    
    return result

import tempfile
result = main()
print("Execution complete!")
PYTHON_SCRIPT
    
    # Launch the FastAPI web server in a separate process
    python3 -c "
from squad.web_server import run_server
run_server(host='0.0.0.0', port=8000)
" &
    SERVER_PID=$!
    
    echo "Server started with PID: $SERVER_PID"
    echo "Opening dashboard in default browser..."
    sleep 2
    
    # Try to open browser
    if command -v open &> /dev/null; then
        open http://localhost:8000 2>/dev/null || true
    elif command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:8000 2>/dev/null || true
    fi
    
    echo ""
    echo "Press Ctrl+C to stop the server when done"
    wait $SERVER_PID
else
    # Run the demo using examples/demo.py
    if [[ "$VERBOSE" == true ]]; then
        python3 examples/demo.py --live $([[ "$MODE" == "live" ]] && echo "--live") --workspace "$WORKSPACE" 2>&1 | tee /dev/tty
    else
        if [[ "$MODE" == "live" ]]; then
            python3 examples/demo.py --live --workspace "$WORKSPACE" 2>&1
        else
            python3 examples/demo.py --workspace "$WORKSPACE" 2>&1
        fi
    fi
fi

echo ""
echo "=============================================================="
echo "Demo complete."
echo "=============================================================="

