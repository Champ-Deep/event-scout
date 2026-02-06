# Frontend Deployment Guide

**Deploy the Event Scout frontend to Vercel or Netlify in under 5 minutes.**

---

## Before You Start

You need:
- [ ] Railway backend deployed and working
- [ ] Railway URL (e.g., `https://myapp.railway.app`)
- [ ] Your `APP_API_KEY` from Railway environment variables

---

## Step 1: Configure Frontend

Open `frontend.html` and update lines 367-370:

```javascript
const CONFIG = {
    API_BASE_URL: 'https://your-railway-app.railway.app', // UPDATE THIS
    API_KEY: 'your-app-api-key-here' // UPDATE THIS
};
```

**Replace with your actual values:**

```javascript
const CONFIG = {
    API_BASE_URL: 'https://round-table-app-production.up.railway.app',
    API_KEY: 'ZPtQtufMnSQh-fKmyOs_Z5qrgyKdQ1jAwqSI06skGM4'
};
```

**⚠️ Important**: Remove the trailing slash from `API_BASE_URL` if present.

---

## Option A: Deploy to Vercel (Recommended)

### Why Vercel?
- ✅ Free tier (generous)
- ✅ Automatic HTTPS
- ✅ Global CDN
- ✅ One-click deployment
- ✅ Custom domains

### Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

### Step 2: Login to Vercel

```bash
vercel login
```

### Step 3: Deploy

```bash
cd Round_Table_Conferrance_app_Authenticated
vercel --prod frontend.html
```

**Vercel will prompt you:**

```
? Set up and deploy "Round_Table_Conferrance_app_Authenticated"? [Y/n] y
? Which scope do you want to deploy to? <your-account>
? Link to existing project? [y/N] n
? What's your project's name? event-scout-frontend
? In which directory is your code located? ./
```

**That's it!** Vercel will give you a live URL like:

```
https://event-scout-frontend.vercel.app
```

### Step 4: Verify Deployment

1. Open the Vercel URL in your browser
2. Try registering a new user
3. Try adding a contact
4. Verify QR codes display correctly

---

## Option B: Deploy to Netlify

### Why Netlify?
- ✅ Free tier
- ✅ Drag-and-drop deployment
- ✅ Automatic HTTPS
- ✅ Custom domains

### Step 1: Prepare for Deployment

Rename `frontend.html` to `index.html`:

```bash
cp frontend.html index.html
```

### Step 2: Deploy via Netlify Drop

1. Go to [Netlify Drop](https://app.netlify.com/drop)
2. Drag `index.html` into the drop zone
3. Wait 10 seconds for deployment
4. Netlify will give you a URL like:
   ```
   https://silly-beaver-123456.netlify.app
   ```

### Step 3: (Optional) Custom Domain

1. In Netlify dashboard, click "Domain Settings"
2. Click "Add custom domain"
3. Follow DNS configuration instructions

---

## Option C: Deploy to Railway (with Frontend)

You can host the frontend on the same Railway service:

### Step 1: Create `static` Directory

```bash
mkdir static
cp frontend.html static/index.html
```

### Step 2: Update `Dockerfile`

Add these lines before `CMD`:

```dockerfile
# Copy static frontend files
COPY static /app/static

# Serve static files
RUN pip install --no-cache-dir aiofiles
```

### Step 3: Update `app.py`

Add static file serving:

```python
from fastapi.staticfiles import StaticFiles

# Add after app initialization
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

### Step 4: Deploy

```bash
git add .
git commit -m "Add frontend static serving"
git push origin master
```

Railway will auto-deploy. Your frontend will be at:

```
https://your-railway-app.railway.app/
```

---

## Testing Your Deployment

### Test 1: Open Frontend

Open your deployed URL in a browser.

**Expected**: Login/Register screen loads

### Test 2: Register New User

1. Click "Register" tab
2. Fill in:
   - Name: Test User
   - Email: test@yourcompany.com
   - Password: TestPassword123!
3. Click "Create Account"

**Expected**: Successful registration, redirected to dashboard

### Test 3: Add Contact

1. Click "+ Add Contact"
2. Fill in contact details:
   - Name: Jane Smith
   - Email: jane@example.com
   - Phone: +1234567890
   - Company: Acme Corp
3. Click "Save Contact"

**Expected**:
- Contact appears in list
- QR code displays
- Search works

### Test 4: Search Contacts

1. Type "Acme" in search bar
2. Press Enter or click Search

**Expected**: Only contacts from Acme Corp show

### Test 5: Edit Contact

1. Click "Edit" on a contact
2. Change phone number
3. Click "Save Contact"

**Expected**: Contact updates with new QR code

### Test 6: Delete Contact

1. Click "Delete" on a contact
2. Confirm deletion

**Expected**: Contact removed from list

---

## Troubleshooting

### Issue: "Network error" on login/register

**Cause**: Incorrect `API_BASE_URL` or Railway backend not running

**Fix**:
1. Verify Railway backend is deployed:
   ```bash
   curl https://your-railway-app.railway.app/health/
   ```
2. Check `API_BASE_URL` in `frontend.html` matches Railway URL exactly
3. Ensure no trailing slash in URL
4. Check browser console for CORS errors

### Issue: CORS errors in browser console

**Cause**: Backend CORS not configured properly

**Fix**: Your backend already has CORS enabled for all origins:

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

If still seeing CORS errors, check Railway logs for errors.

### Issue: "Invalid API Key" errors

**Cause**: `API_KEY` in frontend doesn't match Railway `APP_API_KEY`

**Fix**:
1. Check Railway environment variables
2. Copy exact `APP_API_KEY` value
3. Update `CONFIG.API_KEY` in `frontend.html`
4. Redeploy frontend

### Issue: QR codes not displaying

**Cause**: QR codes not generated or corrupted

**Fix**:
1. Check backend logs for QR generation errors
2. Verify `/app/saved_qr` volume is mounted in Railway
3. Try adding a new contact (triggers QR generation)

### Issue: Contacts disappear after refresh

**Cause**: Using wrong `user_id` or localStorage cleared

**Fix**:
1. Check browser console for errors
2. Verify `user_id` is stored in localStorage:
   ```javascript
   console.log(localStorage.getItem('eventScoutUser'))
   ```
3. Re-login if user data is missing

---

## Production Hardening (Optional)

### 1. Environment-Based Configuration

Instead of hardcoding API credentials in `frontend.html`, use environment variables:

**Create `config.js`:**

```javascript
window.ENV = {
    API_BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
    API_KEY: import.meta.env.VITE_API_KEY || 'default-key'
};
```

**In Vercel:**
- Settings → Environment Variables
- Add `VITE_API_URL` and `VITE_API_KEY`

### 2. Add Analytics

**Google Analytics:**

```html
<!-- Add before </head> -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

### 3. Add Custom Domain

**Vercel:**
1. Dashboard → Project → Settings → Domains
2. Add domain: `contacts.yourcompany.com`
3. Configure DNS CNAME:
   ```
   contacts.yourcompany.com → cname.vercel-dns.com
   ```

**Netlify:**
1. Dashboard → Domain Settings → Add custom domain
2. Follow DNS instructions

### 4. Enable PWA (Progressive Web App)

Add `manifest.json` for mobile app feel:

```json
{
  "name": "Event Scout",
  "short_name": "Contacts",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#667eea",
  "theme_color": "#667eea",
  "icons": [
    {
      "src": "/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    }
  ]
}
```

---

## Cost Estimation

### Vercel Free Tier
- ✅ 100 GB bandwidth/month
- ✅ Unlimited sites
- ✅ Automatic HTTPS
- ✅ Global CDN
- **Cost**: $0/month

### Netlify Free Tier
- ✅ 100 GB bandwidth/month
- ✅ 300 build minutes/month
- ✅ Automatic HTTPS
- **Cost**: $0/month

### Custom Domain (Optional)
- Domain registration: ~$10-15/year
- DNS management: Free (Cloudflare)

**Total Monthly Cost**: $0 (using free tiers)

---

## Maintenance

### Updating Frontend

1. **Make changes to `frontend.html`**
2. **Redeploy**:

**Vercel:**
```bash
vercel --prod frontend.html
```

**Netlify:**
- Drag updated `index.html` to Netlify Drop

**Railway (if hosting on same service):**
```bash
git add static/index.html
git commit -m "Update frontend"
git push origin master
```

### Monitoring

**Vercel Analytics:**
- Dashboard → Analytics
- See page views, load times, visitors

**Netlify Analytics:**
- Dashboard → Analytics
- Requires paid plan ($9/month)

**Browser Console:**
- Open DevTools (F12)
- Check Console for errors
- Check Network tab for failed requests

---

## Security Best Practices

### ✅ Already Implemented

- HTTPS enforced (Vercel/Netlify automatic)
- User credentials never stored (only in localStorage encrypted by browser)
- API key required for backend calls
- CORS enabled on backend

### ⚠️ Additional Hardening

1. **API Key Rotation**:
   - Rotate `APP_API_KEY` every 90 days
   - Update Railway environment variable
   - Update frontend config
   - Redeploy

2. **Rate Limiting** (backend):
   - Implement rate limiting with slowapi
   - Protect against brute force attacks

3. **Content Security Policy** (frontend):
   ```html
   <meta http-equiv="Content-Security-Policy"
         content="default-src 'self'; connect-src https://your-railway-app.railway.app;">
   ```

---

## Next Steps

- [ ] Frontend deployed and accessible
- [ ] Configuration verified
- [ ] All tests passed
- [ ] Custom domain configured (optional)
- [ ] Analytics enabled (optional)
- [ ] Share URL with team

---

**Your Event Scout system is now fully deployed!** 🎉

Users can:
- Register accounts
- Add/edit/delete contacts
- Generate QR codes
- Search contacts semantically
- Access from any device

---

## Support

If you encounter issues:

1. Check browser console (F12 → Console)
2. Check Railway backend logs
3. Verify API configuration
4. Test backend health endpoint
5. Review this guide's troubleshooting section

---

**Built with ❤️ for Champions Ranch**
