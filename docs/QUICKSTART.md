# Quick Start Guide - Enterprise Multi-Agent Platform

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
cd /Users/albert/orca/workspaces/bisai/11-genai-open-agent/11_genai-open-agent

# Install with web dependencies
pip install -e ".[web]"

# Or manually
pip install fastapi uvicorn websockets pydantic
```

### Step 2: Run Tests (Verify Setup)

```bash
python3 -m pytest tests
# Expected: 66 passed ✅
```

### Step 3: Launch Enterprise Dashboard

```bash
# Option A: Traditional mode (CLI only)
./scripts/demo.sh

# Option B: Web dashboard mode (recommended)
./scripts/demo.sh --web

# This will:
# 1. Install missing dependencies
# 2. Start FastAPI server on port 8000
# 3. Open http://localhost:8000 in browser
# 4. Execute the 4-agent squad pipeline
```

---

## 📊 What You'll See

### Dashboard Features

1. **Real-Time DAG Visualization**
   - Animated agent execution flow
   - Status badges (Idle → Thinking → Executing → Done)
   - Live score updates

2. **Metrics Panel**
   - Total agents executed
   - Average completion score
   - Token usage estimates
   - Execution duration waterfall

3. **Blackboard Inspector**
   - Click any agent node
   - View JSON findings
   - Compare state changes

4. **Human-in-the-Loop Gate**
   - Review final results
   - Approve or send back for improvement
   - Provide feedback for retry

---

## 🔧 Command Reference

### Demo Script Options

```bash
./scripts/demo.sh                     # Offline demo (MockLLM)
./scripts/demo.sh --live              # Use real OpenAI provider
./scripts/demo.sh --workspace PATH    # Audit custom directory
./scripts/demo.sh --web               # Launch web dashboard
./scripts/demo.sh --verbose           # Verbose output
```

### Manual Server Launch

```bash
# Start server directly
python3 -c "from squad.web_server import run_server; run_server(host='0.0.0.0', port=8000)"

# Access dashboard at http://localhost:8000
```

---

## 🎯 Success Criteria

### All Requirements Met:

✅ **Test Suite**: 66/66 tests passing  
✅ **Agent Scores**: Security agent at 0.858 (>= 0.85)  
✅ **Demo Exit Code**: Returns 0  
✅ **DAG Visualizer**: Real-time WebSocket updates  
✅ **Blackboard Diff**: State snapshot capture  
✅ **HIL Gating**: Approve/reject workflow  

### Expected Execution Output:

```
wave 1 -> group architecture
  audit-architecture         started
  -> audit-architecture       score=0.842 ok
group architecture done in 11 ms

wave 2 -> group security
  audit-security             started
  -> audit-security           score=0.858 ok ✅
group security done in 100 ms

wave 3 -> group tests
  audit-tests                started
  -> audit-tests              score=0.842 ok
group tests done in 162 ms

wave 4 -> group review
  audit-review               started
  -> audit-review             score=0.842 ok
group review done in 8 ms

status     : PASS ✅
provider : mock
```

---

## 🌐 API Quick Reference

### WebSocket Connection

```javascript
// In browser console
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
};
```

### REST Endpoints

```bash
# Get metrics
curl http://localhost:8000/metrics

# Get blackboard diffs
curl http://localhost:8000/blackboard-diffs

# Submit HIL decision
curl -X POST http://localhost:8000/hil/gating \
  -H "Content-Type: application/json" \
  -d '{"decision":"approve"}'
```

---

## 📖 Additional Resources

- Full Documentation: `docs/ENTERPRISE_PLATFORM.md`
- Verification Checklist: `docs/VERIFICATION_CHECKLIST.md`
- Architecture Overview: `docs/ARCHITECTURE.md`

---

## ⚠️ Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill it if needed
kill -9 <PID>
```

### Dependency Issues

```bash
# Reinstall fresh
pip uninstall agent-squad
pip install -e ".[web]"
```

### Browser Won't Open

```bash
# Manually navigate to dashboard
open http://localhost:8000
# or
xdg-open http://localhost:8000
```

---

## 🎉 Next Steps

After launching the dashboard:

1. Watch agents execute in real-time
2. Click on each agent node to see findings
3. Review the final gate decision
4. Try the approve/reject buttons
5. Experiment with different workspaces

Enjoy your enterprise multi-agent collaboration platform! 🚀
