#!/bin/bash

# ============================================
# Event Scout Railway Deployment Test Script
# ============================================
# Tests your Railway deployment with real API calls
#
# Usage:
#   chmod +x test_railway_deployment.sh
#   ./test_railway_deployment.sh <RAILWAY_URL> <APP_API_KEY>
#
# Example:
#   ./test_railway_deployment.sh https://myapp.railway.app ZPtQtufMnSQh-fKmyOs_Z5qr...
# ============================================

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ "$#" -ne 2 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo "Usage: $0 <RAILWAY_URL> <APP_API_KEY>"
    echo "Example: $0 https://myapp.railway.app ZPtQtufMnSQh-fKmyOs_Z5qr..."
    exit 1
fi

RAILWAY_URL=$1
APP_API_KEY=$2

# Remove trailing slash if present
RAILWAY_URL=${RAILWAY_URL%/}

echo "============================================"
echo "Event Scout Railway Deployment Test"
echo "============================================"
echo "Railway URL: $RAILWAY_URL"
echo "API Key: ${APP_API_KEY:0:10}..."
echo ""

# Test 1: Health Check
echo -e "${YELLOW}Test 1: Health Check${NC}"
HEALTH_RESPONSE=$(curl -s "$RAILWAY_URL/health/")
echo "$HEALTH_RESPONSE" | python3 -m json.tool

if echo "$HEALTH_RESPONSE" | grep -q '"status": "healthy"'; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
    exit 1
fi
echo ""

# Test 2: User Registration
echo -e "${YELLOW}Test 2: User Registration${NC}"
TEST_EMAIL="test_$(date +%s)@example.com"
TEST_PASSWORD="SecurePassword123!"

REGISTER_RESPONSE=$(curl -s -X POST "$RAILWAY_URL/register/" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"Test User\",
    \"email\": \"$TEST_EMAIL\",
    \"password\": \"$TEST_PASSWORD\"
  }")

echo "$REGISTER_RESPONSE" | python3 -m json.tool

if echo "$REGISTER_RESPONSE" | grep -q '"status": "success"'; then
    echo -e "${GREEN}✓ User registration passed${NC}"
    USER_ID=$(echo "$REGISTER_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['user_id'])")
    echo "User ID: $USER_ID"
else
    echo -e "${RED}✗ User registration failed${NC}"
    exit 1
fi
echo ""

# Test 3: User Login
echo -e "${YELLOW}Test 3: User Login${NC}"
LOGIN_RESPONSE=$(curl -s -X POST "$RAILWAY_URL/login/" \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"$TEST_EMAIL\",
    \"password\": \"$TEST_PASSWORD\"
  }")

echo "$LOGIN_RESPONSE" | python3 -m json.tool

if echo "$LOGIN_RESPONSE" | grep -q '"status": "success"'; then
    echo -e "${GREEN}✓ User login passed${NC}"
else
    echo -e "${RED}✗ User login failed${NC}"
    exit 1
fi
echo ""

# Test 4: Add Contact
echo -e "${YELLOW}Test 4: Add Contact${NC}"
ADD_CONTACT_RESPONSE=$(curl -s -X POST "$RAILWAY_URL/add_contact/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $APP_API_KEY" \
  -d "{
    \"contact\": {
      \"name\": \"Jane Doe\",
      \"email\": \"jane.doe@example.com\",
      \"phone\": \"+1234567890\",
      \"linkedin\": \"https://linkedin.com/in/janedoe\",
      \"company_name\": \"Acme Corp\"
    },
    \"user_id\": \"$USER_ID\"
  }")

echo "$ADD_CONTACT_RESPONSE" | python3 -m json.tool

if echo "$ADD_CONTACT_RESPONSE" | grep -q '"status": "success"'; then
    echo -e "${GREEN}✓ Add contact passed${NC}"
    CONTACT_ID=$(echo "$ADD_CONTACT_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['contact_id'])")
    echo "Contact ID: $CONTACT_ID"
else
    echo -e "${RED}✗ Add contact failed${NC}"
    exit 1
fi
echo ""

# Test 5: List Contacts
echo -e "${YELLOW}Test 5: List Contacts${NC}"
LIST_CONTACTS_RESPONSE=$(curl -s "$RAILWAY_URL/list_contacts/?user_id=$USER_ID" \
  -H "X-API-Key: $APP_API_KEY")

echo "$LIST_CONTACTS_RESPONSE" | python3 -m json.tool | head -20

CONTACT_COUNT=$(echo "$LIST_CONTACTS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['total_contacts'])")

if [ "$CONTACT_COUNT" -eq 1 ]; then
    echo -e "${GREEN}✓ List contacts passed (found $CONTACT_COUNT contact)${NC}"
else
    echo -e "${RED}✗ List contacts failed (expected 1, found $CONTACT_COUNT)${NC}"
    exit 1
fi
echo ""

# Test 6: Search Contacts
echo -e "${YELLOW}Test 6: Search Contacts${NC}"
SEARCH_RESPONSE=$(curl -s -X POST "$RAILWAY_URL/search/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $APP_API_KEY" \
  -d "{
    \"query\": \"Acme Corp\",
    \"user_id\": \"$USER_ID\"
  }")

echo "$SEARCH_RESPONSE" | python3 -m json.tool

if echo "$SEARCH_RESPONSE" | grep -q '"status": "success"'; then
    echo -e "${GREEN}✓ Search contacts passed${NC}"
else
    echo -e "${RED}✗ Search contacts failed${NC}"
    exit 1
fi
echo ""

# Test 7: Delete Contact
echo -e "${YELLOW}Test 7: Delete Contact${NC}"
DELETE_RESPONSE=$(curl -s -X DELETE "$RAILWAY_URL/contact/$CONTACT_ID?user_id=$USER_ID" \
  -H "X-API-Key: $APP_API_KEY")

echo "$DELETE_RESPONSE" | python3 -m json.tool

if echo "$DELETE_RESPONSE" | grep -q '"status": "success"'; then
    echo -e "${GREEN}✓ Delete contact passed${NC}"
else
    echo -e "${RED}✗ Delete contact failed${NC}"
    exit 1
fi
echo ""

# Test 8: Verify Volume Persistence (requires manual restart)
echo -e "${YELLOW}Test 8: Volume Persistence Check${NC}"
echo "To test volume persistence:"
echo "1. Go to Railway Dashboard → Deployments"
echo "2. Click 'Restart' on your deployment"
echo "3. Wait 30 seconds for restart"
echo "4. Run this command:"
echo ""
echo "   curl -X POST \"$RAILWAY_URL/login/\" \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"email\": \"$TEST_EMAIL\", \"password\": \"$TEST_PASSWORD\"}'"
echo ""
echo "If login succeeds, volumes are working correctly!"
echo ""

# Summary
echo "============================================"
echo -e "${GREEN}All tests passed!${NC}"
echo "============================================"
echo ""
echo "Your Railway deployment is working correctly."
echo ""
echo "Test User Credentials:"
echo "  Email: $TEST_EMAIL"
echo "  Password: $TEST_PASSWORD"
echo "  User ID: $USER_ID"
echo ""
echo "Next steps:"
echo "  1. Test volume persistence (see Test 8 above)"
echo "  2. Build and deploy frontend"
echo "  3. Connect frontend to this Railway URL"
echo "============================================"
