#!/bin/bash
# Event Scout — Daily Pre-Event Health Check
# ============================================
# Run this every morning before the event to verify all systems.
#
# Usage: ./check.sh

set -euo pipefail

PROD_URL="https://event-scout-production.up.railway.app"
FRONTEND_URL="https://event-scout-delta.vercel.app"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${NC}  $1"; }
fail() { echo -e "  ${RED}[!!]${NC}  $1"; FAILURES=$((FAILURES + 1)); }
warn() { echo -e "  ${YELLOW}[--]${NC}  $1"; }

FAILURES=0
echo "=========================================="
echo "  Event Scout — Daily Health Check"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# 1. Backend Health
echo "1. Backend Health"
HEALTH=$(curl -sf "${PROD_URL}/health/" 2>/dev/null) || { fail "Backend UNREACHABLE at ${PROD_URL}"; HEALTH=""; }
if [ -n "$HEALTH" ]; then
    VERSION=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'v{d.get(\"version\",\"?\")} | DB: {d.get(\"database\",\"?\")} | Users: {d.get(\"total_users\",\"?\")} | Gemini: {d.get(\"gemini_configured\")} | OpenRouter: {d.get(\"openrouter_configured\")}')" 2>/dev/null)
    ok "$VERSION"
fi

# 2. Frontend
echo ""
echo "2. Frontend"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$FRONTEND_URL" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    # Check timeout value
    TIMEOUT_VAL=$(curl -s "$FRONTEND_URL" | grep -o 'TIMEOUT_MS: [0-9]*' | head -1 || echo "unknown")
    ok "HTTP $HTTP_CODE | $TIMEOUT_VAL"
else
    fail "Frontend returned HTTP $HTTP_CODE"
fi

# 3. CORS Preflight
echo ""
echo "3. CORS Preflight"
CORS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -X OPTIONS "${PROD_URL}/health/" \
    -H "Origin: ${FRONTEND_URL}" \
    -H "Access-Control-Request-Method: POST" \
    -H "Access-Control-Request-Headers: X-API-Key,Content-Type" 2>/dev/null || echo "000")
if [ "$CORS_STATUS" = "200" ]; then
    ok "CORS preflight: HTTP $CORS_STATUS"
else
    fail "CORS preflight returned HTTP $CORS_STATUS"
fi

# 4. SSL Certificate
echo ""
echo "4. SSL Certificate"
SSL_EXPIRY=$(echo | openssl s_client -connect event-scout-production.up.railway.app:443 -servername event-scout-production.up.railway.app 2>/dev/null | openssl x509 -noout -dates 2>/dev/null | grep notAfter | cut -d= -f2)
if [ -n "$SSL_EXPIRY" ]; then
    ok "SSL expires: $SSL_EXPIRY"
else
    warn "Could not check SSL certificate"
fi

# 5. API Response Time
echo ""
echo "5. API Response Time"
RESP_TIME=$(curl -s -o /dev/null -w "%{time_total}" "${PROD_URL}/health/" 2>/dev/null || echo "99")
RESP_MS=$(echo "$RESP_TIME * 1000" | bc 2>/dev/null | cut -d. -f1 || echo "?")
if [ "${RESP_MS:-999}" -lt 3000 ]; then
    ok "Health endpoint: ${RESP_MS}ms"
else
    warn "Health endpoint slow: ${RESP_MS}ms"
fi

# 6. Git Status
echo ""
echo "6. Git Status"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BRANCH=$(cd "$SCRIPT_DIR" && git branch --show-current 2>/dev/null || echo "unknown")
COMMIT=$(cd "$SCRIPT_DIR" && git log --oneline -1 2>/dev/null || echo "unknown")
DIRTY=$(cd "$SCRIPT_DIR" && git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
ok "Branch: $BRANCH | Commit: $COMMIT"
if [ "$DIRTY" -gt 0 ]; then
    warn "$DIRTY uncommitted changes"
fi

# Summary
echo ""
echo "=========================================="
if [ "$FAILURES" -eq 0 ]; then
    echo -e "  ${GREEN}ALL CHECKS PASSED${NC} — Ready for the event!"
else
    echo -e "  ${RED}${FAILURES} CHECK(S) FAILED${NC} — Investigate before event!"
fi
echo "=========================================="

exit "$FAILURES"
