#!/bin/bash
# Event Scout — Deploy & Test Orchestrator
# ==========================================
# Usage:
#   ./deploy.sh test       — Run Tier 1 smoke tests against production
#   ./deploy.sh verify     — Run Tier 2 functional tests against production
#   ./deploy.sh frontend   — Deploy frontend to Vercel + verify
#   ./deploy.sh backend    — Push to Railway + wait for build + smoke test
#   ./deploy.sh full       — Backend + frontend + verify all
#   ./deploy.sh status     — Quick health check
#
# Environment:
#   TEST_PASSWORD — Required for tier 2+ tests

set -euo pipefail

PROD_URL="https://event-scout-production.up.railway.app"
FRONTEND_URL="https://event-scout-delta.vercel.app"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
info() { echo "     $1"; }

wait_for_backend() {
    local url="$1"
    local max_attempts="${2:-30}"
    echo "Waiting for backend at $url ..."
    for i in $(seq 1 "$max_attempts"); do
        if curl -sf "${url}/health/" > /dev/null 2>&1; then
            ok "Backend is live!"
            return 0
        fi
        echo "  Attempt $i/$max_attempts ..."
        sleep 10
    done
    fail "Backend did not respond after $max_attempts attempts"
    return 1
}

cmd_status() {
    echo "=== Event Scout Status ==="
    echo ""

    # Backend health
    HEALTH=$(curl -sf "${PROD_URL}/health/" 2>/dev/null) || { fail "Backend unreachable"; return 1; }
    VERSION=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('version','?'))" 2>/dev/null)
    DB=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('database','?'))" 2>/dev/null)
    USERS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_users','?'))" 2>/dev/null)
    ok "Backend: v${VERSION} | DB: ${DB} | Users: ${USERS}"

    # Frontend
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$FRONTEND_URL" 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ]; then
        ok "Frontend: HTTP $HTTP_CODE"
    else
        fail "Frontend: HTTP $HTTP_CODE"
    fi

    # Git branch
    BRANCH=$(cd "$SCRIPT_DIR" && git branch --show-current 2>/dev/null)
    COMMIT=$(cd "$SCRIPT_DIR" && git log --oneline -1 2>/dev/null)
    info "Branch: $BRANCH"
    info "Commit: $COMMIT"
}

cmd_test() {
    echo "=== Tier 1 Smoke Tests ==="
    python3 "${SCRIPT_DIR}/tests/test_suite.py" --target "$PROD_URL" --frontend "$FRONTEND_URL" --tier 1
}

cmd_verify() {
    echo "=== Tier 2 Functional Tests ==="
    if [ -z "${TEST_PASSWORD:-}" ]; then
        fail "TEST_PASSWORD env var is required for Tier 2 tests"
        exit 1
    fi
    python3 "${SCRIPT_DIR}/tests/test_suite.py" --target "$PROD_URL" --frontend "$FRONTEND_URL" --tier 2
}

cmd_frontend() {
    echo "=== Frontend Deploy (Vercel) ==="

    # Deploy
    echo "Deploying to Vercel..."
    cd "$SCRIPT_DIR"
    npx vercel --prod --yes

    # Verify
    echo ""
    echo "Verifying frontend..."
    sleep 5
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$FRONTEND_URL")
    if [ "$HTTP_CODE" = "200" ]; then
        ok "Frontend deployed: HTTP $HTTP_CODE"
    else
        fail "Frontend returned HTTP $HTTP_CODE"
        return 1
    fi

    # Check key features in deployed HTML
    FEATURES=$(curl -s "$FRONTEND_URL" | grep -c 'resilientFetch\|renderUIComponents\|loadPipelineStatus' || true)
    if [ "$FEATURES" -ge 3 ]; then
        ok "Key features present ($FEATURES matches)"
    else
        warn "Only $FEATURES feature matches found — check deployment"
    fi
}

cmd_backend() {
    echo "=== Backend Deploy (Railway) ==="

    cd "$SCRIPT_DIR"
    BRANCH=$(git branch --show-current)

    # Push current branch to Railway remote
    echo "Pushing $BRANCH to event-scout/master..."
    git push event-scout "${BRANCH}:master"
    ok "Code pushed"

    # Wait for Railway to rebuild
    echo ""
    wait_for_backend "$PROD_URL" 30

    # Smoke test
    echo ""
    echo "Running smoke tests..."
    python3 "${SCRIPT_DIR}/tests/test_suite.py" --target "$PROD_URL" --tier 1
}

cmd_full() {
    echo "=========================================="
    echo "  FULL DEPLOY — Backend + Frontend"
    echo "=========================================="
    echo ""

    cmd_backend
    echo ""
    cmd_frontend
    echo ""

    if [ -n "${TEST_PASSWORD:-}" ]; then
        echo "Running Tier 2 verification..."
        cmd_verify
    else
        warn "TEST_PASSWORD not set — skipping Tier 2 verify"
        echo "Running Tier 1 smoke test..."
        cmd_test
    fi

    echo ""
    ok "Full deploy complete!"
}

# Main
case "${1:-help}" in
    status)   cmd_status ;;
    test)     cmd_test ;;
    verify)   cmd_verify ;;
    frontend) cmd_frontend ;;
    backend)  cmd_backend ;;
    full)     cmd_full ;;
    *)
        echo "Event Scout Deploy & Test"
        echo ""
        echo "Usage: ./deploy.sh <command>"
        echo ""
        echo "Commands:"
        echo "  status     Quick health check"
        echo "  test       Tier 1 smoke tests against production"
        echo "  verify     Tier 2 functional tests (needs TEST_PASSWORD)"
        echo "  frontend   Deploy frontend to Vercel + verify"
        echo "  backend    Push to Railway + wait + smoke test"
        echo "  full       Backend + frontend + verify"
        ;;
esac
