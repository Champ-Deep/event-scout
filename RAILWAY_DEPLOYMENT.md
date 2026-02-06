# Railway Deployment Guide

**Complete step-by-step instructions for deploying Event Scout to Railway with persistent storage.**

---

## Prerequisites

- [ ] GitHub account with repository access
- [ ] Railway account (sign up at [railway.app](https://railway.app))
- [ ] Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))
- [ ] Generated production `APP_API_KEY` (see below)

---

## Phase 1: Pre-Deployment Setup

### Step 1: Generate Production API Key

Run this command locally:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Save this key securely** - you'll need it for Railway environment variables.

Example output:
```
ZPtQtufMnSQh-fKmyOs_Z5qrgyKdQ1jAwqSI06skGM4
```

### Step 2: Get Gemini API Key

1. Visit: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Select a Google Cloud project (or create new)
4. Copy the API key (starts with `AIzaSy...`)

---

## Phase 2: Railway Project Setup

### Step 1: Create New Railway Project

1. **Log in to Railway**: https://railway.app/dashboard
2. **Click "New Project"**
3. **Select "Deploy from GitHub repo"**
4. **Authorize GitHub**: Grant Railway access to your repositories
5. **Select Repository**: Choose `Round_Table_Conferrance_app_Authenticated`
6. **Railway will automatically**:
   - Detect the `Dockerfile`
   - Start building the container
   - Assign a random project name

### Step 2: Configure Environment Variables

1. **In Railway Dashboard**:
   - Click on your project
   - Click "Variables" tab (left sidebar)

2. **Add Environment Variables**:

   Click "New Variable" and add these **two variables**:

   | Variable Name | Value |
   |---------------|-------|
   | `APP_API_KEY` | `<your-generated-key-from-step-1>` |
   | `GEMINI_API_KEY` | `<your-gemini-key-from-step-2>` |

3. **Click "Deploy"** to apply environment variables

**⚠️ CRITICAL: Without these variables, the app will use insecure defaults!**

---

## Phase 3: Configure Persistent Volumes

**Why Volumes Matter**: Railway containers are ephemeral - they restart/redeploy frequently. Without volumes, all user data will be lost on every restart.

### Step 1: Add Volume for User Data

1. **In Railway Dashboard**:
   - Click on your service
   - Click "Settings" → "Volumes" section

2. **Click "New Volume"**:
   - **Mount Path**: `/app/users`
   - **Size**: `1 GB` (can increase later)
   - Click "Add"

### Step 2: Add Volume for QR Codes

1. **Click "New Volume"** again:
   - **Mount Path**: `/app/saved_qr`
   - **Size**: `1 GB`
   - Click "Add"

### Step 3: Add Volume for Legacy Data

1. **Click "New Volume"** again:
   - **Mount Path**: `/app/data`
   - **Size**: `1 GB`
   - Click "Add"

### Step 4: Verify Volume Configuration

Your "Volumes" section should show:

```
✓ /app/users (1 GB)
✓ /app/saved_qr (1 GB)
✓ /app/data (1 GB)
```

### Step 5: Trigger Redeployment

After adding volumes, Railway needs to redeploy:

1. Click "Deployments" tab
2. Click "Redeploy" on the latest deployment
3. Wait for build + deploy to complete (~3-5 minutes)

---

## Phase 4: Generate Public URL

By default, Railway services are private. To make your API accessible:

### Step 1: Generate Domain

1. **In Railway Dashboard**:
   - Click "Settings" → "Networking" section

2. **Click "Generate Domain"**:
   - Railway will create a public URL like:
     ```
     https://round-table-conferrance-app-production.up.railway.app
     ```

3. **Copy this URL** - you'll use it for:
   - Frontend API calls
   - Testing with curl/Postman
   - Sharing with your team

### Step 2: (Optional) Custom Domain

If you have a custom domain:

1. Click "Add Custom Domain"
2. Enter your domain (e.g., `api.yourcompany.com`)
3. Follow Railway's DNS configuration instructions
4. Add CNAME record to your DNS provider

---

## Phase 5: Verify Deployment

### Step 1: Check Health Endpoint

Test your deployment with curl:

```bash
curl https://<your-railway-url>.railway.app/health/
```

**Expected Response**:

```json
{
  "status": "healthy",
  "multi_user": true,
  "total_users": 0,
  "gemini_configured": true
}
```

### Step 2: Check Deployment Logs

1. **In Railway Dashboard**:
   - Click "Deployments" → Latest deployment
   - Click "View Logs"

2. **Look for these startup messages**:

```
[INIT] Directory ensured: /app/users
[INIT] Directory ensured: /app/saved_qr
[INIT] Directory ensured: /app/data
[INIT] Testing volume persistence...
[INIT] ✓ Volume persistence verified: /app/data
[INIT] ✓ Volume persistence verified: /app/users
[INIT] ✓ Volume persistence verified: /app/saved_qr
[INIT] Gemini API configured successfully
[STARTUP] Contact Assistant API Starting...
```

**⚠️ If you see volume test failures**:
```
[INIT] ✗ WARNING: Volume persistence test failed for /app/users
```

→ **Go back to Phase 3** and verify volume mount paths are **exact matches**.

### Step 3: Test User Registration

```bash
curl -X POST https://<your-railway-url>.railway.app/register/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test User",
    "email": "test@example.com",
    "password": "secure_password_123"
  }'
```

**Expected Response**:

```json
{
  "status": "success",
  "message": "User registered successfully",
  "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "Test User",
  "email": "test@example.com"
}
```

### Step 4: Test User Login

```bash
curl -X POST https://<your-railway-url>.railway.app/login/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "secure_password_123"
  }'
```

**Expected Response**:

```json
{
  "status": "success",
  "message": "Login successful",
  "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "Test User",
  "email": "test@example.com"
}
```

### Step 5: Test Data Persistence

**This is the critical test to verify volumes work:**

1. **Create a test user** (using step 3 above)
2. **In Railway Dashboard**:
   - Click "Deployments"
   - Click "Restart" on the current deployment
3. **Wait for restart** (~30 seconds)
4. **Try logging in again** (using step 4)
5. **If login succeeds** → Volumes are working! ✅
6. **If login fails with "Invalid credentials"** → Volumes not configured properly ❌

---

## Phase 6: Save Important Information

Create a secure note with these critical values:

```
=== Event Scout Railway Deployment ===

Railway URL: https://<your-app>.railway.app
APP_API_KEY: <your-generated-key>
GEMINI_API_KEY: <your-gemini-key>

First Test User:
- Email: test@example.com
- Password: secure_password_123
- User ID: <user-id-from-registration>

Deployment Date: <date>
```

**⚠️ Store this securely** (password manager, encrypted note, etc.)

---

## Phase 7: Monitor & Maintain

### Viewing Logs

Real-time logs:
1. Railway Dashboard → Deployments → Latest → View Logs
2. Watch for errors, API calls, user registrations

### Checking Resource Usage

1. Railway Dashboard → Metrics tab
2. Monitor:
   - CPU usage
   - Memory usage
   - Network bandwidth
   - Volume storage

### Handling Restarts

Railway automatically restarts containers:
- On code push (GitHub auto-deploy)
- On environment variable changes
- On manual restart
- On platform maintenance

**Your data persists** because of volumes ✅

### Troubleshooting Common Issues

#### Issue: "Invalid credentials" after restart

**Cause**: Volumes not mounted properly

**Fix**:
1. Verify volume mount paths in Railway Settings → Volumes
2. Ensure paths are: `/app/users`, `/app/saved_qr`, `/app/data`
3. Redeploy after fixing

#### Issue: Health check returns 503

**Cause**: Container failed to start

**Fix**:
1. Check deployment logs for errors
2. Verify environment variables are set
3. Check resource limits (may need to upgrade Railway plan)

#### Issue: Gemini API errors in logs

**Cause**: Invalid or missing `GEMINI_API_KEY`

**Fix**:
1. Verify key in Railway Variables
2. Test key at https://aistudio.google.com/
3. Check API quota limits

#### Issue: High volume usage

**Cause**: Many users/contacts + QR codes

**Fix**:
1. Railway Dashboard → Settings → Volumes
2. Click volume → "Resize"
3. Increase to 2GB, 5GB, or 10GB

---

## Cost Estimation

**Railway Pricing** (as of 2026):

| Resource | Free Tier | Paid Plan |
|----------|-----------|-----------|
| Service | $0/month (5 services) | $5/month + usage |
| Volume Storage | $0.25/GB/month | Same |
| Bandwidth | $0.10/GB | Same |
| Compute | $0.000463/hour | Same |

**Estimated Monthly Cost for Event Scout**:

- **Light Usage** (1-10 users): $5-8/month
- **Medium Usage** (10-50 users): $10-15/month
- **Heavy Usage** (50-100 users): $15-25/month

**Free Tier**: If you have <5 services and minimal usage, Railway offers $5/month free credit.

---

## Next Steps

- [ ] Deployment verified and working
- [ ] Data persistence tested
- [ ] Railway URL saved securely
- [ ] API keys stored securely
- [ ] Ready to build frontend

→ **Proceed to Frontend Development** (next phase)

---

## Rollback Procedure

If deployment fails catastrophically:

1. **Rollback in Railway**:
   - Click "Deployments"
   - Find the last working deployment
   - Click "⋯" menu → "Redeploy"

2. **Local Rollback**:
   ```bash
   git log --oneline  # Find last good commit
   git reset --hard <commit-hash>
   git push --force origin master
   ```

3. **Restore Data** (if volumes corrupted):
   - Railway doesn't have built-in volume backups
   - Consider implementing periodic backups to S3/Google Cloud Storage

---

## Security Checklist

Before going live with real users:

- [ ] Production `APP_API_KEY` generated and set
- [ ] Gemini API key has usage quotas configured
- [ ] Environment variables not exposed in logs
- [ ] Railway domain is HTTPS (automatic)
- [ ] `.env` file is gitignored (verified)
- [ ] Test user created and login verified
- [ ] Data persists across restarts
- [ ] Health endpoint returns 200
- [ ] API documentation shared with frontend developer

---

## Support Resources

- **Railway Docs**: https://docs.railway.app
- **Railway Community**: https://discord.gg/railway
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **This Project**: See [README.md](README.md)

---

**You're now ready to deploy your frontend!** 🚀
