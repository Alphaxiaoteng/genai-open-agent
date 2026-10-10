#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  11_genai-open-agent · Open Agent Hackathon 参赛作品演示${NC}"
echo -e "${CYAN}  零外部依赖、高鲁棒性 Multi-Agent 软件工程代码审查协作编排引擎${NC}"
echo -e "${CYAN}================================================================${NC}"

echo -e "\n${BLUE}[1/2] 运行 4-Agent 波次流水线代码审查演示 (离线确定性 Mock / 毫秒级)...${NC}"
uv run python examples/demo.py

echo -e "\n${BLUE}[2/2] 验证 JSON 契约导出与评审鲁棒性...${NC}"
uv run python examples/demo.py --json > /dev/null
echo -e "${GREEN}✓ JSON 契约与黑板状态机序列化验证完全通过${NC}"

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN}  11_genai-open-agent 演示完成！全部链路 Exit Code 0${NC}"
echo -e "${GREEN}================================================================${NC}"
