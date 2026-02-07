# Railway Quick Deployment Guide

## Option 1: Deploy via Web Dashboard (Recommended - Faster)

### Step 1: Go to Railway Dashboard
1. Visit https://railway.app/dashboard
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Authorize Railway to access your GitHub
5. Select this repository: `Round_Table_Conferrance_app_Authenticated`

### Step 2: Wait for Build
- Railway will automatically detect the Dockerfile
- Build takes ~3-5 minutes
- You can watch the logs in the deployment tab

### Step 3: Set Environment Variables
1. Click on your service
2. Go to "Variables" tab
3. Add these variables:
   ```
   APP_API_KEY=OGibuBdW6KP52UMTpv8g46Zs37g47d9SGv4w21W-o6s
   GEMINI_API_KEY=AIzaSyAvIxbwsd1HeCEokTAQrwJH_-a_Ypf9sJQ
   ```
4. Click "Add" for each variable
5. Railway will automatically redeploy

### Step 4: Add Persistent Volumes
1. Go to "Settings" tab
2. Scroll to "Volumes" section
3. Click "Add Volume" and create these 3 volumes:
   - **Volume 1**: Mount Path = `/app/users`, Size = `1 GB`
   - **Volume 2**: Mount Path = `/app/saved_qr`, Size = `1 GB`
   - **Volume 3**: Mount Path = `/app/data`, Size = `1 GB`
4. After adding volumes, manually trigger a redeploy

### Step 5: Generate Public Domain
1. Go to "Settings" tab
2. Scroll to "Networking" section
3. Click "Generate Domain"
4. Copy the generated URL (e.g., `https://your-app.up.railway.app`)

### Step 6: Verify Deployment
Visit: `https://your-app.up.railway.app/health/`

Expected response:
```json
{
  "status": "healthy",
  "multi_user": true,
  "total_users": 0,
  "gemini_configured": true
}
```

---

## Option 2: Deploy via CLI

### Step 1: Login
```bash
railway login
```
This opens your browser for authentication.

### Step 2: Initialize Project
```bash
railway init
```
- Choose "Create new project"
- Name: `event-scout-backend`

### Step 3: Link and Deploy
```bash
# Set environment variables
railway variables set APP_API_KEY="OGibuBdW6KP52UMTpv8g46Zs37g47d9SGv4w21W-o6s"
railway variables set GEMINI_API_KEY="AIzaSyAvIxbwsd1HeCEokTAQrwJH_-a_Ypf9sJQ"

# Deploy
railway up
```

### Step 4: Add Volumes (via Dashboard)
You still need to add volumes via the web dashboard (see Option 1, Step 4)

### Step 5: Get Domain
```bash
railway domain
```
This generates a public URL.

---

## After Deployment

Once you have your Railway URL, share it here so I can:
1. Update `mobile-frontend.html` CONFIG with your Railway URL
2. Create `vercel.json` for frontend deployment
3. Deploy the frontend to Vercel
4. Provide mobile testing instructions

---

## Troubleshooting

**Build fails**: Check Railway logs for errors
**Health check fails**: Verify environment variables are set
**Contacts disappear**: Ensure volumes are configured correctly
**Gemini errors**: Check GEMINI_API_KEY is valid

---

**Next Steps**: After Railway deployment completes, we'll deploy the frontend to Vercel and test on mobile!
