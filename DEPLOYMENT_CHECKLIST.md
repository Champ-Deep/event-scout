# 🚀 Event Scout Deployment Checklist

**Complete deployment workflow from code to production in ~2 hours.**

---

## 📋 Pre-Deployment Preparation (15 min)

### ✅ Phase 0: Verify Prerequisites

- [ ] Git repository up to date (`git status`)
- [ ] Python 3.11+ installed locally
- [ ] Docker installed (optional, for local testing)
- [ ] Railway account created ([railway.app](https://railway.app))
- [ ] Vercel account created ([vercel.com](https://vercel.com)) *or* Netlify account

---

## 🔐 Phase 1: Security Setup (15 min)

### Step 1.1: Generate Production API Key

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Save this key securely** (password manager, encrypted note):

```
Production API Key: ________________________________
```

### Step 1.2: Get Gemini API Key

1. Visit: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Select a Google Cloud project (or create new)
4. Copy the API key

**Save this key securely**:

```
Gemini API Key: ________________________________
```

### Step 1.3: Verify Security Hardening

- [ ] `.env.example` contains API key generation instructions
- [ ] Volume persistence verification added to `app.py`
- [ ] README.md updated with deployment guide

✅ **Security setup complete!**

---

## 🚂 Phase 2: Railway Backend Deployment (30 min)

**Follow detailed guide**: [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md)

### Step 2.1: Create Railway Project

1. [ ] Login to Railway dashboard
2. [ ] Click "New Project" → "Deploy from GitHub repo"
3. [ ] Select `Round_Table_Conferrance_app_Authenticated`
4. [ ] Wait for initial build (~3-5 minutes)

### Step 2.2: Configure Environment Variables

In Railway dashboard → Variables:

```
APP_API_KEY = <your-production-key-from-step-1.1>
GEMINI_API_KEY = <your-gemini-key-from-step-1.2>
```

- [ ] `APP_API_KEY` added
- [ ] `GEMINI_API_KEY` added
- [ ] Variables saved and deployment triggered

### Step 2.3: Configure Persistent Volumes

In Railway dashboard → Settings → Volumes:

**Add 3 volumes:**

1. [ ] **Volume 1**: Mount Path `/app/users`, Size `1 GB`
2. [ ] **Volume 2**: Mount Path `/app/saved_qr`, Size `1 GB`
3. [ ] **Volume 3**: Mount Path `/app/data`, Size `1 GB`

- [ ] All 3 volumes added
- [ ] Manual redeploy triggered after adding volumes

### Step 2.4: Generate Public Domain

1. [ ] Railway dashboard → Settings → Networking
2. [ ] Click "Generate Domain"
3. [ ] Save Railway URL:

```
Railway Backend URL: https://________________________________.railway.app
```

### Step 2.5: Verify Backend Health

```bash
curl https://your-railway-url.railway.app/health/
```

**Expected response:**

```json
{
  "status": "healthy",
  "multi_user": true,
  "total_users": 0,
  "gemini_configured": true
}
```

- [ ] Health endpoint returns 200 OK
- [ ] `status: "healthy"` in response
- [ ] `gemini_configured: true` in response

### Step 2.6: Test Backend with Script

```bash
chmod +x test_railway_deployment.sh
./test_railway_deployment.sh <RAILWAY_URL> <APP_API_KEY>
```

- [ ] All 7 tests passed
- [ ] User registration successful
- [ ] Contact CRUD operations working
- [ ] Test user credentials saved

✅ **Railway backend deployment complete!**

---

## 🎨 Phase 3: Frontend Configuration (10 min)

### Step 3.1: Update Frontend Configuration

Open `frontend.html` in your editor.

Find lines 367-370:

```javascript
const CONFIG = {
    API_BASE_URL: 'https://your-railway-app.railway.app', // UPDATE THIS
    API_KEY: 'your-app-api-key-here' // UPDATE THIS
};
```

**Replace with your actual values:**

```javascript
const CONFIG = {
    API_BASE_URL: 'https://<your-railway-url>.railway.app',
    API_KEY: '<your-production-api-key>'
};
```

- [ ] `API_BASE_URL` updated (no trailing slash!)
- [ ] `API_KEY` updated with production key
- [ ] Changes saved

### Step 3.2: Test Frontend Locally (Optional)

```bash
# Open in browser
open frontend.html
# or
python3 -m http.server 8080
# then visit http://localhost:8080/frontend.html
```

- [ ] Frontend loads without errors
- [ ] Can register a new user
- [ ] Can add a contact
- [ ] QR code displays

✅ **Frontend configuration complete!**

---

## 🌐 Phase 4: Frontend Deployment (15 min)

**Follow detailed guide**: [FRONTEND_DEPLOYMENT.md](FRONTEND_DEPLOYMENT.md)

### Option A: Deploy to Vercel (Recommended)

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel --prod frontend.html
```

**Vercel will output a URL:**

```
Frontend URL (Vercel): https://________________________________.vercel.app
```

- [ ] Vercel CLI installed
- [ ] Deployed successfully
- [ ] Live URL saved

### Option B: Deploy to Netlify

1. [ ] Rename `frontend.html` to `index.html`
2. [ ] Go to [Netlify Drop](https://app.netlify.com/drop)
3. [ ] Drag `index.html` into drop zone
4. [ ] Save live URL:

```
Frontend URL (Netlify): https://________________________________.netlify.app
```

✅ **Frontend deployment complete!**

---

## 🧪 Phase 5: Integration Testing (20 min)

### Test 1: User Registration

1. [ ] Open frontend URL in browser
2. [ ] Click "Register" tab
3. [ ] Fill in test user details:
   - Name: Test User
   - Email: test@yourcompany.com
   - Password: TestPassword123!
4. [ ] Click "Create Account"
5. [ ] **Expected**: Redirected to dashboard

### Test 2: Add Contact

1. [ ] Click "+ Add Contact" button
2. [ ] Fill in contact details:
   - Name: Jane Smith
   - Email: jane@example.com
   - Phone: +1234567890
   - LinkedIn: https://linkedin.com/in/janesmith
   - Company: Acme Corp
3. [ ] Click "Save Contact"
4. [ ] **Expected**: Contact appears with QR code

### Test 3: Search Contacts

1. [ ] Type "Acme" in search bar
2. [ ] Press Enter
3. [ ] **Expected**: Only Acme Corp contacts show

### Test 4: Edit Contact

1. [ ] Click "Edit" on Jane Smith
2. [ ] Change phone to: +9876543210
3. [ ] Click "Save Contact"
4. [ ] **Expected**: Contact updates, new QR code generated

### Test 5: Delete Contact

1. [ ] Click "Delete" on Jane Smith
2. [ ] Confirm deletion
3. [ ] **Expected**: Contact removed from list

### Test 6: Logout and Re-login

1. [ ] Click "Logout" button
2. [ ] Confirm logout
3. [ ] Login again with same credentials
4. [ ] **Expected**: All contacts still present

✅ **Integration testing complete!**

---

## 🔄 Phase 6: Data Persistence Verification (10 min)

**This is critical to verify volumes work correctly.**

### Step 6.1: Restart Railway Container

1. [ ] Railway dashboard → Deployments
2. [ ] Click "Restart" on current deployment
3. [ ] Wait 30 seconds for restart

### Step 6.2: Test Login After Restart

1. [ ] Open frontend URL
2. [ ] Login with test user credentials
3. [ ] **Expected**: Login succeeds, contacts still present

### Step 6.3: Check Railway Logs

1. [ ] Railway dashboard → Deployments → View Logs
2. [ ] Look for volume persistence messages:

```
[INIT] Testing volume persistence...
[INIT] ✓ Volume persistence verified: /app/data
[INIT] ✓ Volume persistence verified: /app/users
[INIT] ✓ Volume persistence verified: /app/saved_qr
```

- [ ] All volume tests passed
- [ ] No warning messages about volume failures
- [ ] User data persists after restart

✅ **Data persistence verified!**

---

## 📊 Phase 7: Production Validation (10 min)

### Step 7.1: Create Real Users

1. [ ] Register your actual user account
2. [ ] Add 3-5 real contacts
3. [ ] Test search with real queries
4. [ ] Verify QR codes scan correctly (use phone camera)

### Step 7.2: Share with Team

**Save these credentials securely:**

```
=== Event Scout Production ===

Backend:
  URL: https://________________________________.railway.app
  Health: https://________________________________.railway.app/health/

Frontend:
  URL: https://________________________________.vercel.app
  (or Netlify URL)

Admin/Test User:
  Email: ________________________________
  Password: ________________________________

API Keys (NEVER SHARE PUBLICLY):
  APP_API_KEY: ________________________________
  GEMINI_API_KEY: ________________________________

Deployment Date: ________________________________
```

- [ ] Credentials saved securely
- [ ] Team members invited to test
- [ ] Feedback collected

### Step 7.3: Monitor First 24 Hours

**Check Railway logs for:**
- [ ] No error spikes
- [ ] Successful user registrations
- [ ] Contact CRUD operations working
- [ ] QR generation successful
- [ ] Gemini API calls working

**Check Frontend (browser console):**
- [ ] No JavaScript errors
- [ ] API calls returning 200
- [ ] No CORS errors

✅ **Production validation complete!**

---

## 🎉 Deployment Complete!

### What You've Built

✅ **Backend (Railway)**
- Multi-user authentication system
- Contact management with CRUD operations
- FAISS semantic search
- Gemini AI conversational queries
- QR code generation/scanning
- Persistent data storage with volumes

✅ **Frontend (Vercel/Netlify)**
- Responsive single-page application
- User registration and login
- Contact management interface
- Real-time search
- QR code display
- Mobile-friendly design

✅ **Infrastructure**
- Production-grade security (bcrypt, API keys)
- Data persistence across restarts
- Health monitoring
- Automatic HTTPS
- Global CDN

---

## 📝 Post-Deployment Tasks

### Immediate (Next 7 Days)

- [ ] **Monitor usage**: Check Railway metrics daily
- [ ] **User feedback**: Collect and document issues
- [ ] **Bug fixes**: Address any critical issues
- [ ] **Documentation**: Update with any deployment quirks

### Short-term (Next 30 Days)

- [ ] **Custom domain**: Configure custom domain for frontend
- [ ] **Analytics**: Add Google Analytics or similar
- [ ] **Backups**: Implement data backup strategy
- [ ] **Rate limiting**: Add rate limiting to protect API quota

### Long-term (Next 90 Days)

- [ ] **Email verification**: Add email verification flow
- [ ] **Password reset**: Implement forgot password
- [ ] **CSV export**: Allow users to export contacts
- [ ] **Admin dashboard**: Build admin interface
- [ ] **Mobile app**: Consider native mobile app (React Native)

---

## 💰 Cost Estimation

### Current Setup (Free Tiers)

| Service | Plan | Monthly Cost |
|---------|------|--------------|
| Railway | Starter (with $5 credit) | $0-5 |
| Vercel/Netlify | Free | $0 |
| Google Gemini API | Free quota | $0 |
| **Total** | | **$0-5/month** |

### Scaling (Medium Usage: 50-100 users)

| Service | Plan | Monthly Cost |
|---------|------|--------------|
| Railway | Pro | $15-25 |
| Vercel/Netlify | Free | $0 |
| Google Gemini API | Pay-as-you-go | $5-10 |
| **Total** | | **$20-35/month** |

---

## 🆘 Troubleshooting

### Backend Issues

**Problem**: Health check fails
**Solution**: Check Railway logs, verify environment variables set

**Problem**: Contacts disappear after restart
**Solution**: Verify volumes configured correctly (Phase 2.3)

**Problem**: Gemini API errors
**Solution**: Check API key, verify quota at aistudio.google.com

### Frontend Issues

**Problem**: "Network error" on login
**Solution**: Verify `API_BASE_URL` matches Railway URL exactly

**Problem**: CORS errors
**Solution**: Check Railway backend logs, CORS should be enabled

**Problem**: QR codes not displaying
**Solution**: Check backend logs, verify `/app/saved_qr` volume mounted

### Detailed Troubleshooting

See:
- [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md) - Backend issues
- [FRONTEND_DEPLOYMENT.md](FRONTEND_DEPLOYMENT.md) - Frontend issues
- [README.md](README.md) - General troubleshooting

---

## 📚 Documentation Reference

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview, API docs, quick start |
| [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md) | Detailed Railway deployment guide |
| [FRONTEND_DEPLOYMENT.md](FRONTEND_DEPLOYMENT.md) | Frontend deployment options |
| [TESTING_GUIDE.md](TESTING_GUIDE.md) | Backend testing guide |
| [test_railway_deployment.sh](test_railway_deployment.sh) | Automated backend testing |

---

## 🎯 Success Criteria

You've successfully deployed if:

- [x] Health endpoint returns 200 OK
- [x] Users can register and login
- [x] Contacts persist after restart
- [x] QR codes generate and display
- [x] Search returns relevant results
- [x] Frontend connects to backend
- [x] No errors in logs
- [x] Team can access and use system

---

## 🎊 Next Steps

**Share your success:**
1. Document any deployment gotchas
2. Share feedback with the team
3. Plan feature roadmap
4. Consider contributing improvements

**Consider upgrades:**
- Custom domain for branding
- Email notifications
- Mobile app
- Advanced analytics
- Team collaboration features

---

**Congratulations! Your Event Scout system is live! 🚀**

**Built with ❤️ for Champions Ranch**

---

## Support

Need help?
- Review troubleshooting sections in guides
- Check Railway/Vercel documentation
- Open an issue on GitHub
- Contact your development team

**Deployment Date**: ________________
**Deployed By**: ________________
**Status**: ✅ Production Ready
