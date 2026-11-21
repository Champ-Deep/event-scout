# Event Scout field-sales design contract

## 1. Visual theme and atmosphere
A mobile-first event networking utility for people moving between conversations. Operate mode: capture a card, find a contact, prepare a follow-up. The approved component grammar and `docs/ui-brief-v1.md` are authoritative. This is not a new brand direction.

## 2. Color palette and roles
Primary #2563eb. Primary dark and focus #1e40af. Ink #0f172a. Secondary ink #475569. Page #f9fafb. Surface #ffffff. Line #e2e8f0. Field border #64748b. Error #b91c1c. Success text #065f46. Hot #dc2626 with white text. Warm #f59e0b with ink. Cold #e2e8f0 with ink. Every colored control declares its foreground explicitly.

## 3. Typography rules
System sans only, without downloaded fonts. Titles 24px bold, state headings 20px bold, contact names 18px bold, body and controls 16px, metadata and lead badges 14px, labels 12px. Sentence case. Contact identifiers wrap without widening the page.

## 4. Component stylings
The four-tab pill navigation adapts 21st.dev 8343. Sign in adapts 28353 with the existing Circle Scout mascot and no social providers. The scan queue adapts 44136, with failures first and a collapsed completed ledger. Camera and gallery capture adapt 19201. Contact rows adapt 28596 with an avatar, title/company, server lead badge and chevron. Date filters reuse 35042, inputs 160 and badges 11351. Contact editing adapts 26142 over Base UI Dialog. Ask reuses Suggestions 12342 and Chat Input 557, without streaming. Me adapts 28358 over Base UI Tabs. Notifications use Sonner.

Inputs and actions have a 44px minimum target and a 3px blue focus ring. Modal focus, inert background and Escape use the existing Base UI foundation. Errors remain inline with an available recovery action. Deletion requires confirmation. Copy confirms success and retains a manual fallback on failure.

## 5. Layout principles
4px spacing grid. 16px page gutters. Main content capped at 42rem. Floating navigation reserves bottom safe-area clearance. Ask composer sits above it. Contact detail is a bottom sheet on phones and a 448px side panel at 768px and above. Settings use three card tabs: Profile, Event, My card.

## 6. Depth and elevation
Contacts use hairlines and no row shadows. Subtle elevation is reserved for the sign-in card, navigation and transient sheet. Solid surfaces only. No gradients, glass or decorative entrance motion.

## 7. Do and do not
Show actual server data, including score band words and numbers. Do not create sample records in the shipped runtime, calculate substitute scores, port Feed, send admin_mode or change admin/public-card surfaces. Render all user, contact and assistant fields as React text, never HTML. Only validated http(s) links become hrefs. Keep queued photos owner-scoped on the device. Never describe provider failure as successful extraction.

## 8. Responsive behavior
Device class: web pointer and touch, mobile first. Verify 360, 375, 390, 699, 768, 1280 and 1440px. Forms scroll vertically; wide content must not widen the viewport. Safe areas are applied to navigation, composer and sheet. No dark theme ships. Motion budget: inherited chip selection feedback only, without decorative choreography; reduced motion removes transitions and animations. Toast feedback uses transform and opacity, not animated height.

## 9. Agent prompt guide
Use `docs/ui-brief-v1.md` over donor integration notes. Preserve the product blue palette, system font and content cap rather than importing group marketing tokens. Mount the app at `/`, preserve the legacy page at `/admin.html`, and keep the public card viewer unchanged. Test against a local Worker with local D1 and KV only. Test fixtures belong in loopback-only QA scripts, not runtime code. Production deployment is a separate approval step.
