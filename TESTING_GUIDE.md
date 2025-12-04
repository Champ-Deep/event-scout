# Multi-User Contact Management API - Testing Guide

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
# OR
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

---

## Test Flow: Complete Multi-User Scenario

### Step 1: Health Check

```bash
curl -X GET http://localhost:8000/health/
```

**Expected Response:**
```json
{
  "status": "healthy",
  "multi_user": true,
  "total_users": 0
}
```

---

### Step 2: Register First User (Alice)

```bash
curl -X POST http://localhost:8000/register/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Alice Johnson",
    "email": "alice@example.com",
    "password": "alice123"
  }'
```

**Expected Response:**
```json
{
  "status": "success",
  "message": "User registered successfully",
  "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "Alice Johnson",
  "email": "alice@example.com"
}
```

**📝 SAVE THE user_id - You'll need it for all future requests!**

---

### Step 3: Register Second User (Bob)

```bash
curl -X POST http://localhost:8000/register/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Bob Smith",
    "email": "bob@example.com",
    "password": "bob456"
  }'
```

**Expected Response:**
```json
{
  "status": "success",
  "message": "User registered successfully",
  "user_id": "b2c3d4e5-f6g7-8901-bcde-fg2345678901",
  "name": "Bob Smith",
  "email": "bob@example.com"
}
```

**📝 SAVE Bob's user_id as well!**

---

### Step 4: Login as Alice

```bash
curl -X POST http://localhost:8000/login/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alice@example.com",
    "password": "alice123"
  }'
```

**Expected Response:**
```json
{
  "status": "success",
  "message": "Login successful",
  "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "Alice Johnson",
  "email": "alice@example.com"
}
```

---

### Step 5: Alice Adds Her First Contact

Replace `ALICE_USER_ID` with Alice's actual user_id:

```bash
curl -X POST http://localhost:8000/add_contact/ \
  -H "Content-Type: application/json" \
  -H "x-api-key: 1234" \
  -H "x-user-id: ALICE_USER_ID" \
  -d '{
    "name": "Charlie Brown",
    "email": "charlie@techcorp.com",
    "phone": "555-1234",
    "linkedin": "linkedin.com/in/charlie",
    "company_name": "TechCorp"
  }'
```

**Expected Response:**
```json
{
  "status": "success",
  "message": "Contact added",
  "contact_id": "c3d4e5f6-g7h8-9012-cdef-gh3456789012",
  "qr_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

---

### Step 6: Alice Adds Second Contact

```bash
curl -X POST http://localhost:8000/add_contact/ \
  -H "Content-Type: application/json" \
  -H "x-api-key: 1234" \
  -H "x-user-id: ALICE_USER_ID" \
  -d '{
    "name": "Diana Prince",
    "email": "diana@startupco.com",
    "phone": "555-5678",
    "linkedin": "linkedin.com/in/diana",
    "company_name": "StartupCo"
  }'
```

---

### Step 7: Bob Adds His Contact

Replace `BOB_USER_ID` with Bob's actual user_id:

```bash
curl -X POST http://localhost:8000/add_contact/ \
  -H "Content-Type: application/json" \
  -H "x-api-key: 1234" \
  -H "x-user-id: BOB_USER_ID" \
  -d '{
    "name": "Eve Adams",
    "email": "eve@bigcompany.com",
    "phone": "555-9999",
    "linkedin": "linkedin.com/in/eve",
    "company_name": "BigCompany"
  }'
```

---

### Step 8: List Alice's Contacts (Should show 2 contacts)

```bash
curl -X GET http://localhost:8000/list_contacts/ \
  -H "x-api-key: 1234" \
  -H "x-user-id: ALICE_USER_ID"
```

**Expected Response:**
```json
{
  "status": "success",
  "total_contacts": 2,
  "contacts": [
    {
      "id": "...",
      "name": "Charlie Brown",
      "email": "charlie@techcorp.com",
      "phone": "555-1234",
      "linkedin": "linkedin.com/in/charlie",
      "company_name": "TechCorp",
      "qr_base64": "..."
    },
    {
      "id": "...",
      "name": "Diana Prince",
      "email": "diana@startupco.com",
      "phone": "555-5678",
      "linkedin": "linkedin.com/in/diana",
      "company_name": "StartupCo",
      "qr_base64": "..."
    }
  ]
}
```

---

### Step 9: List Bob's Contacts (Should show 1 contact - DATA ISOLATION TEST!)

```bash
curl -X GET http://localhost:8000/list_contacts/ \
  -H "x-api-key: 1234" \
  -H "x-user-id: BOB_USER_ID"
```

**Expected Response:**
```json
{
  "status": "success",
  "total_contacts": 1,
  "contacts": [
    {
      "id": "...",
      "name": "Eve Adams",
      "email": "eve@bigcompany.com",
      "phone": "555-9999",
      "linkedin": "linkedin.com/in/eve",
      "company_name": "BigCompany",
      "qr_base64": "..."
    }
  ]
}
```

**✅ Bob cannot see Alice's contacts - Data isolation is working!**

---

### Step 10: Search Alice's Contacts

```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -H "x-api-key: 1234" \
  -H "x-user-id: ALICE_USER_ID" \
  -d '{
    "query": "TechCorp"
  }'
```

**Expected Response:** Should return Charlie Brown from TechCorp

---

### Step 11: Search Bob's Contacts for "Charlie" (Should find nothing!)

```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -H "x-api-key: 1234" \
  -H "x-user-id: BOB_USER_ID" \
  -d '{
    "query": "Charlie"
  }'
```

**Expected Response:** Should return empty results (Bob doesn't have Charlie in his contacts)

**✅ Search isolation is working!**

---

### Step 12: Get Specific Contact

Replace `CONTACT_ID` with an actual contact_id from Alice's contacts:

```bash
curl -X GET http://localhost:8000/contact/CONTACT_ID \
  -H "x-api-key: 1234" \
  -H "x-user-id: ALICE_USER_ID"
```

---

### Step 13: Update Contact

```bash
curl -X PUT http://localhost:8000/contact/CONTACT_ID \
  -H "Content-Type: application/json" \
  -H "x-api-key: 1234" \
  -H "x-user-id: ALICE_USER_ID" \
  -d '{
    "phone": "555-0000",
    "company_name": "TechCorp Updated"
  }'
```

---

### Step 14: Delete Contact

```bash
curl -X DELETE http://localhost:8000/contact/CONTACT_ID \
  -H "x-api-key: 1234" \
  -H "x-user-id: ALICE_USER_ID"
```

---

### Step 15: Conversational Search (Using Gemini AI)

```bash
curl -X POST http://localhost:8000/converse/ \
  -H "Content-Type: application/json" \
  -H "x-api-key: 1234" \
  -H "x-user-id: ALICE_USER_ID" \
  -d '{
    "query": "Who do I know at TechCorp?",
    "top_k": 4
  }'
```

---

## Error Test Cases

### Test 1: Try to register with duplicate email

```bash
curl -X POST http://localhost:8000/register/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Alice Duplicate",
    "email": "alice@example.com",
    "password": "different123"
  }'
```

**Expected:** Error - "Email already registered"

---

### Test 2: Login with wrong password

```bash
curl -X POST http://localhost:8000/login/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alice@example.com",
    "password": "wrongpassword"
  }'
```

**Expected:** 401 - "Invalid credentials"

---

### Test 3: Access without user_id header

```bash
curl -X GET http://localhost:8000/list_contacts/ \
  -H "x-api-key: 1234"
```

**Expected:** 422 - Missing required header x-user-id

---

### Test 4: Access with invalid user_id

```bash
curl -X GET http://localhost:8000/list_contacts/ \
  -H "x-api-key: 1234" \
  -H "x-user-id: invalid-uuid-12345"
```

**Expected:** 404 - "User not found"

---

## Key Points Verified

✅ **User Registration** - Each user gets a unique UUID
✅ **User Login** - Password verification with bcrypt
✅ **Data Isolation** - Users can only see their own contacts
✅ **Per-User FAISS Index** - Each user has separate vector database
✅ **Search Isolation** - Search results are user-specific
✅ **CRUD Operations** - All operations are scoped to user_id
✅ **Authentication Flow** - Frontend stores user_id after login
✅ **Security** - Passwords hashed, API key required

---

## Architecture Summary

```
User Registration/Login
        ↓
  Frontend stores user_id
        ↓
  All API requests include:
  - x-api-key: 1234 (existing security)
  - x-user-id: <uuid> (user identification)
        ↓
  Backend routes to user-specific:
  - FAISS index: users/<user_id>/faiss.index
  - Metadata: users/<user_id>/metadata.pickle
        ↓
  Complete data isolation per user
```

---

## File Structure

```
users/
├── <user_id_1>/
│   ├── faiss.index          # User 1's vector index
│   └── metadata.pickle       # User 1's contact metadata
├── <user_id_2>/
│   ├── faiss.index          # User 2's vector index
│   └── metadata.pickle       # User 2's contact metadata
└── ...

users.json                    # User accounts (email, hashed password, UUID)
saved_qr/                     # QR codes for all contacts
```

---

## Frontend Integration

The frontend should:

1. **On Registration/Login**: Store the `user_id` in localStorage or sessionStorage
2. **On Every API Call**: Include `x-user-id` header with the stored UUID
3. **On Logout**: Clear the stored `user_id`

```javascript
// Example Frontend Code
// After login/register:
localStorage.setItem('user_id', response.user_id);

// For all API calls:
fetch('http://localhost:8000/add_contact/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'x-api-key': '1234',
    'x-user-id': localStorage.getItem('user_id')
  },
  body: JSON.stringify(contactData)
});
```
