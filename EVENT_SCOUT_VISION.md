# Event Scout: Product Vision & Roadmap

> PRD Addendum — This document extends the original Event Scout PRD (`Event_Scout_PRD_Mobile_Web_App.docx`) with the full product vision for transforming Event Scout into a comprehensive event intelligence partner.

---

## Product Vision

**Event Scout is your AI-powered event partner.** It helps professionals get the most out of trade shows, conferences, and networking events by:

- Identifying the right targets before the event
- Providing real-time intelligence during the event
- Automating research, follow-up, and outreach after the event
- Being an always-available assistant that remembers every contact, conversation, and insight

The end goal is a platform where a user can walk into any event and have Event Scout guide them through who to meet, what to say, and how to follow up — turning event attendance into measurable business outcomes.

---

## User Journey

### 1. Onboarding & Personalization

**Current:** Users register with name, email, password.

**Target State:**
- Users log in via **LinkedIn OAuth** → profile auto-populated from LinkedIn data
- App understands: job title, company, industry, products/services they sell
- User sets up **personalization**: target industries, ideal customer roles, pitch style, value propositions
- User configures **event context**: which event they're attending, their goals for the event

**Why:** Every AI output (lead scores, pitch angles, chat responses, email drafts) becomes dramatically more useful when personalized to the user's specific context.

### 2. Pre-Event Intelligence

**Current:** Not implemented.

**Target State:**
- User imports **exhibitor list** (CSV, URL scrape, or manual entry)
- Event Scout auto-researches each exhibitor: company overview, key people, recent news, pain points
- AI generates **target recommendations**: "Based on your data services and these exhibitors, here are the top 10 booths to visit and why"
- Each recommendation includes: pitch angles, talking points, relevant case studies from user's profile
- **Briefing document** generated: a concise pre-event guide the user can review on the way to the event

**Data Sources:** Perplexity API (deep research), Scrapper (website content), event websites, LinkedIn

### 3. During the Event

**Current:** Business card scanning (camera OCR via Gemini), manual contact entry, QR code generation, AI chat.

**Target State (additions):**
- **Instant lead scoring** on scan: as soon as a card is scanned, the contact is scored against the user's profile
  - Score: 0-100 with temperature (hot/warm/cold)
  - Reasoning: "This VP of Marketing at a healthcare company matches your target market and is the right decision-maker for your data enrichment services"
  - Recommended actions: "Ask about their current data vendor" or "Mention the Cigna case study"
- **Pitch angle suggestions**: based on the person's role, company, and user's products
- **Smart assistant** that is context-aware:
  - "Who should I prioritize visiting next?" → answers based on scores + event layout
  - "What should I say to the Acme Corp people?" → uses research + user's pitch style
  - "Did I meet someone who does health tech?" → semantic search across all contacts + notes
  - "Summarize my best leads from today" → aggregates scores and notes
- **Quick notes**: voice-to-text notes attached to contacts immediately after conversations

### 4. Post-Event Intelligence

**Current:** Contact list with export (CSV/JSON), n8n enrichment webhook.

**Target State:**

#### 4a. Deep Research (Automated)
- Triggered automatically or manually per contact (or batch "Research All")
- **Research pipeline:**
  1. Perplexity: company overview, recent news, competitors, financial signals
  2. Scrapper: company website content (about page, team, blog, case studies)
  3. Role analysis: what does this person's role typically care about? What are their KPIs?
  4. Pain point identification: specific to their industry + company size + role
- Results stored on the contact and visible in the app

#### 4b. Pitch Deck Generation
- **Notebook LM-style presentations** — in-depth, research-backed, compelling
- Generated per contact via Google Slides API
- **8-slide structure:**
  1. **Hero:** Personalized opening — "{Contact Name}, here's how we can help {Company}"
  2. **The Challenge:** Role-specific pain points backed by research data
  3. **By The Numbers:** Industry data, competitor moves, market trends
  4. **Cost of Inaction:** What happens if they don't address these challenges
  5. **The Approach:** User's methodology/philosophy (from profile)
  6. **The Solution:** Specific product/service mapping to their pain points
  7. **Case Study:** Most relevant case study from user's profile
  8. **Next Steps:** Clear CTA with proposed meeting/demo
- Personalized across 3 dimensions:
  - Company + role research data
  - User's products, value props, and pitch style
  - Event conversation notes and context
- Shareable Google Slides link attached to the contact

#### 4c. Automated Follow-Up

**Email Sequences (via ChampMail):**
- AI generates personalized 3-email sequence per contact
- Email 1 (Day 2-3): "Great meeting you at {event}" — personal, reference conversation
- Email 2 (Day 5-7): Value-add — share relevant insight or case study
- Email 3 (Day 10-14): Soft CTA — propose a quick call or demo
- Track: sent, opened, clicked, replied
- Auto-pause sequence if recipient replies

**LinkedIn Outreach:**
- AI generates personalized connection message (< 300 chars)
- Copy-paste ready (manual send to protect account)
- Track connection status

**Voice Qualification (via Voicebox):**
- AI generates call script: opener, key questions, objection handlers
- Voice preview audio generated for rehearsal
- Track call outcomes

### 5. Ongoing Intelligence

**Current:** AI chat can search contacts semantically.

**Target State:**
- **Lead engagement dashboard:** hot/warm/cold distribution, outreach status, reply rates
- **Smart reminders:** "You haven't followed up with 3 hot leads from last week"
- **Conversational memory:** "Who did I meet that works in advisory?" → instant answers
- **Cross-event intelligence:** contacts persist across events, enrichment compounds over time
- **Daily digest emails:** summary of lead activity, upcoming follow-ups, new research results

---

## Personalization System

The personalization system is the backbone of all AI-powered features. It lives at `users/{user_id}/profile.json`.

### Profile Fields

**Identity:**
- Full name, job title, company, website, LinkedIn URL

**Products & Services:**
- List of products/services with descriptions and ideal customer profiles
- This is what the AI recommends when suggesting pitch angles

**Target Market:**
- Industries (Healthcare, FinTech, Manufacturing, etc.)
- Company sizes (startup, mid-market, enterprise)
- Roles (CTO, VP Marketing, Head of Data, etc.)
- Geographies (Middle East, India, US, etc.)

**Sales Approach:**
- Pitch style: consultative, direct, challenger, relationship-based
- Value propositions in user's own words
- Common objections and responses
- Case studies with company, result, and industry

**Preferences:**
- Follow-up delay (default 3 days)
- Auto-research on scan (on/off)
- Auto-score on add (on/off)
- Email signature

**Event Context (updated per event):**
- Event name and description
- Goals for this event
- Updated before each event to tune all AI outputs

---

## Lead Scoring Model

Scores are computed by Gemini AI using structured prompting, not a hardcoded formula. This allows nuanced reasoning.

**Input:** Contact data + user profile + event context + notes

**Scoring Dimensions (total: 0-100):**

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Profile Fit | 0-25 | Does this contact match the user's target market? |
| Role Relevance | 0-25 | Is this the right decision-maker for the user's product? |
| Company Fit | 0-20 | Industry, company size, geography alignment |
| Engagement Signals | 0-15 | Quality of notes, conversation depth, expressed interest |
| Timing Score | 0-15 | Event context, urgency signals, recent activity |

**Output:**
- Score (0-100)
- Temperature: Hot (70+), Warm (40-69), Cold (<40)
- Reasoning: 2-3 sentence explanation
- Recommended actions: specific next steps

---

## Technical Architecture for Intelligence Features

### Integration Bus: n8n

All integrations flow through n8n (self-hosted, cloud). Event Scout exposes:
- **Trigger endpoints:** Event Scout calls n8n to start workflows
- **Callback endpoints:** n8n calls Event Scout to report results

This decouples Event Scout from external tool APIs and provides visual debugging, retry logic, and easy workflow modification.

### Data Flow: Contact Research

```
User taps "Research" → Event Scout → n8n webhook
                                        ↓
                                   Perplexity API (deep research)
                                        ↓
                                   Scrapper API (website content)
                                        ↓
                                   Parse + structure results
                                        ↓
                              n8n callback → Event Scout
                                        ↓
                              Contact metadata updated
                                        ↓
                              UI shows research results
```

### Data Flow: Pitch Deck Generation

```
User taps "Generate Pitch" → Event Scout → n8n webhook
                                              ↓
                                         Claude/Gemini generates narrative
                                              ↓
                                         Google Slides API creates deck
                                              ↓
                                    n8n callback → Event Scout
                                              ↓
                                    Contact gets pitch_deck_url
                                              ↓
                                    UI shows "View Pitch Deck" link
```

### Data Flow: Email Outreach

```
User taps "Email" → Event Scout → n8n webhook
                                      ↓
                                 Create prospect in ChampMail
                                      ↓
                                 Claude generates 3-email sequence
                                      ↓
                                 ChampMail sends sequence
                                      ↓
                            ChampMail events → n8n → Event Scout callback
                                      ↓
                            Contact tracks: sent, opened, clicked, replied
```

---

## Implementation Phases

### Phase 1: Event-Ready (This Week)
- User personalization settings (backend + frontend)
- Enhanced AI chat (Gemini system prompt includes user profile)
- Lead scoring endpoint (Gemini-powered, no n8n dependency)
- Score badges on contact cards

### Phase 2: Post-Event Intelligence (1-2 Weeks)
- n8n webhook infrastructure
- Contact research workflow (Perplexity via n8n)
- Pitch deck generation (Google Slides via n8n)
- Research + pitch UI in contact detail
- Intel dashboard tab

### Phase 3: Automated Outreach (3-4 Weeks)
- Deploy Scrapper and ChampMail
- Email sequence automation
- LinkedIn message generation
- Voice call script generation
- Outreach tracking UI

### Phase 4: Full Event Partner (Future)
- LinkedIn OAuth login + profile import
- Pre-event exhibitor import + auto-research
- Target recommendations with AI reasoning
- Pre-event briefing document generation
- Push notifications for hot lead activity
- Daily digest emails
- Cross-event intelligence
- QR-CTWA analytics integration

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Cards scanned per event | 50+ |
| Contacts with research completed | 90%+ within 48 hours |
| Pitch decks generated | 1 per hot/warm lead |
| Email sequences sent | All warm+ leads within 1 week |
| Reply rate on personalized emails | 15%+ (vs 2-5% generic) |
| Meetings booked from event contacts | 10%+ conversion |
| Time from card scan to first outreach | < 48 hours (automated) |
