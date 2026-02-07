# Event Scout - Quick Reference Guide
## World Health Expo Dubai 2026 | Feb 9-12, 2026

---

## 📱 **ACCESS THE APP**

**URL:** https://event-scout-delta.vercel.app/

**Add to Home Screen:**
- iOS: Tap Share → "Add to Home Screen"
- Android: Tap menu → "Install App" or "Add to Home Screen"

---

## 🚀 **QUICK START (First Time)**

1. **Register** → Enter name, email, password
2. **Profile Setup** → Tap Profile tab → Set your:
   - Job title & company
   - Target industries (Healthcare, Medical Devices, etc.)
   - Preferred AI model (Claude Opus recommended)
3. **Test Scanner** → Tap Scan tab → Allow camera access

---

## 📸 **SCAN BUSINESS CARDS**

**Steps:**
1. Tap **Scan** tab (camera icon)
2. Position card in frame (landscape works best)
3. Tap **Capture** button
4. AI extracts: Name, Email, Phone, LinkedIn, Company
5. Review & edit if needed
6. Tap **Save Contact**

**Tips:**
- Good lighting = better OCR accuracy
- If scan fails, manually add contact instead
- LinkedIn URLs auto-detected and added

---

## 💡 **AI ASSISTANT (Smart Chat)**

**What It Can Do:**
- Find contacts: *"Who did I meet from Siemens?"*
- Get insights: *"What companies are in medical imaging?"*
- Lead intel: *"Which contacts are hot leads?"*
- Event intel: *"Tell me about exhibitors in Hall 4"*
- Pitch prep: *"What should I say to the Philips team?"*

**How to Use:**
1. Tap **Chat** tab (message icon)
2. Type question or tap suggestion
3. AI searches your contacts + exhibitor database
4. Get instant, context-aware answers

**Smart Features:**
- **Model Selection:** Tap settings → Choose AI (Claude Opus, GPT-5, Gemini)
- **Conversation History:** Scroll up to see past Q&A
- **Exhibitor Context:** AI knows all 82 WHX exhibitors

**Example Questions:**
- *"Summarize my top 5 leads from today"*
- *"Who should I visit in the Health IT category?"*
- *"Find contacts interested in telehealth"*
- *"Brief me on exhibitors from Germany"*

---

## 🔥 **LEAD SCORING**

**Automatic Scoring:**
- Tap **Score This Lead** on any contact card
- AI analyzes: role, company, fit with your profile
- Get score (0-100) + temperature (Hot/Warm/Cold)
- See reasoning + recommended actions

**Batch Scoring:**
- Tap **Contacts** → **Hot Leads** banner → **Score All**
- Scores all unscored contacts at once
- Hot leads (70+) appear at top of list

**What the Score Means:**
- 🔥 **Hot (70-100):** High-priority, strong fit
- 🟡 **Warm (40-69):** Good potential, follow up
- ❄️ **Cold (<40):** Low priority, future opportunity

---

## 🏢 **EXHIBITOR INTEL**

**Browse Exhibitors:**
1. Tap **Expo** tab (building icon)
2. See all 82 WHX Dubai exhibitors
3. Search by name or filter by category

**Categories:**
- Medical Devices
- Health IT
- Pharmaceuticals
- Laboratory & Diagnostics
- Hospital Equipment
- Telemedicine
- Wellness & Nutrition
- ... and 8 more

**Exhibitor Info:**
- Booth number & hall location
- Company website & country
- Product list & tags
- Description

**Use Cases:**
- Pre-plan booth visits
- Research before approaching
- Find competitors at the event
- Identify partnership opportunities

---

## 📊 **DASHBOARD (Intel View)**

**Quick Stats:**
- Total contacts
- Lead distribution (Hot/Warm/Cold)
- Average lead score
- Top companies & industries

**Actions:**
- Export contacts (CSV/JSON)
- View/edit your QR code (digital business card)
- Manage profile settings

---

## 🎯 **EVENT DAY WORKFLOW**

### **Morning (Pre-Event)**
1. Open app → Review **Expo** tab
2. Identify 5-10 target booths
3. Ask AI: *"Brief me on [Company Name]"*
4. Set daily goal (e.g., "Scan 20 cards today")

### **During Event (On the Floor)**
1. Visit booth → Have conversation
2. Collect business card → **Scan immediately**
3. Add quick note: *"Interested in X, follow up on Y"*
4. Move to next booth
5. **Voice Note** (if enabled): Tap mic → Record summary

### **Mid-Day Break**
1. Check **Hot Leads** banner
2. Tap **Score All** to prioritize
3. Review hot leads → Plan afternoon visits
4. Ask AI: *"Summarize my best leads so far"*

### **End of Day**
1. Export contacts: **Profile** → **Export Data** → CSV
2. Ask AI: *"Give me a summary of today's networking"*
3. Plan tomorrow: *"Which exhibitors should I visit tomorrow?"*

---

## 🛠️ **TROUBLESHOOTING**

### **Scanner Not Working**
- Check camera permissions: iOS Settings → Safari → Camera
- Try portrait mode instead of landscape
- Ensure good lighting
- Fallback: Manually add contact

### **AI Chat Slow/Error**
- Check internet connection
- Try different AI model (Profile → AI Settings)
- Refresh page

### **Contact Not Saved**
- Ensure all required fields: Name, Email, Phone
- Check internet connection
- Try again

### **QR Code Not Generating**
- Go to Profile → My QR Code → Generate
- If failed, edit profile fields and regenerate

---

## 📋 **BEST PRACTICES**

✅ **DO:**
- Scan cards immediately (don't stack them up)
- Add notes right after conversations
- Score leads daily (not at the end)
- Use AI to prep before approaching booths
- Export contacts nightly (backup)

❌ **DON'T:**
- Wait until end of day to scan (you'll forget context)
- Skip adding notes (you'll forget who's who)
- Ignore lead scores (defeats the purpose)
- Forget to charge your phone!

---

## 🔐 **SECURITY & DATA**

- **Your data:** Stored securely in PostgreSQL (Railway cloud)
- **Privacy:** User-isolated data (you only see your contacts)
- **Backup:** Auto-synced to cloud on every save
- **Export:** You own your data (export anytime)

---

## 📞 **SUPPORT**

**Issues During Event:**
- **Backend:** https://event-scout-production.up.railway.app/health/
  - Should show: `"status": "healthy"`
- **Frontend:** https://event-scout-delta.vercel.app/
  - Reload page (pull down to refresh)

**Known Limitations:**
- No offline mode (requires internet)
- Camera requires HTTPS (works on Vercel)
- Max 200 exhibitors displayed (we have 82)

---

## 🎉 **POWER USER TIPS**

1. **Hot Leads Banner:** Tap the red banner on Contacts screen → Quick access to top leads
2. **Conversation History:** Chat remembers context → Ask follow-up questions
3. **Exhibitor Filters:** Expo tab → Click category chips to filter
4. **Quick Actions:** Long-press contact card → Score/Edit/Delete
5. **AI Pitch Prep:** Before approaching booth, ask: *"What should I know about [Company]?"*
6. **Model Selection:** Claude Opus = best reasoning, Gemini = fastest, GPT-5 = balanced
7. **Batch Export:** End of each day → Export CSV → Upload to CRM

---

## ✨ **COMING SOON (Post-Event)**

These features are planned for the weeks following WHX:

- **Deep Research:** Auto-research contacts (company, role, pain points)
- **Pitch Decks:** AI-generated Google Slides presentations per contact
- **Email Sequences:** Automated follow-up email drafts
- **LinkedIn Prep:** Pre-written connection messages
- **Voice Scripts:** Call preparation outlines

---

## 📍 **EVENT DETAILS**

**World Health Expo (WHX) Dubai 2026**
- **Formerly:** Arab Health
- **Dates:** February 9-12, 2026
- **Venue:** Dubai Exhibition Centre (DEC), Expo City Dubai
- **Exhibitors:** 4,800+ from 180+ countries
- **Visitors:** 270,000+ expected

**In Event Scout:** 82 verified exhibitors with booth numbers + 60+ confirmed exhibitors

---

## 🏆 **SUCCESS METRICS**

Track your event ROI:
- **Cards Scanned:** Aim for 50+ over 4 days
- **Hot Leads:** Target 10-15 high-quality leads
- **Conversations:** Quality > Quantity
- **Follow-Up:** Within 48 hours of event end

**Dashboard tracks:**
- Total contacts
- Lead temperature distribution
- Top companies & industries
- Average lead score

---

**Happy Networking! 🚀**

*Event Scout v3.1.0 | Built for WHX Dubai 2026*
