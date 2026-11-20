// AI routes: semantic search, conversational assistant, lead scoring, n8n enrichment,
// pipeline triggers (stubs). Port of app.py /search/, /converse/, score, enrich and pipeline endpoints.
import type { Route } from "../index";
import {
  Ctx, Env, HttpError, all, chat, contactOut, extractJson, first, now, pj, run, sj,
} from "../lib";
import { contactText, reindexLater, searchContacts } from "../embed";

const arr = (v: unknown): any[] => (Array.isArray(v) ? v : []);

async function requireQuery(ctx: Ctx): Promise<string> {
  const b = await ctx.body();
  if (typeof b.query !== "string") throw new HttpError(422, "query is required");
  return b.query;
}

async function loadProfile(env: Env, userId: string): Promise<any> {
  const p = await first<any>(env, "SELECT profile_data FROM user_profiles WHERE user_id = ?", userId);
  return pj<any>(p?.profile_data, {});
}

// ---- semantic search ----
async function search(ctx: Ctx) {
  const query = await requireQuery(ctx);
  const rows = await searchContacts(ctx.env, query, ctx.user.id, 4);
  return {
    status: "success",
    results: rows.map((c) => ({ text: contactText(c), meta: contactOut(c, true) })),
  };
}

// ---- prompt construction (port of IntelligenceEngine) ----
const BASE_PROMPT = `You are Event Scout AI: a strategic event intelligence partner. You don't just retrieve contacts, you help the user WIN at this event. You think like a sales strategist, relationship builder, and event tactician.

YOUR CAPABILITIES:
1. CONTACT INTELLIGENCE: Analyze who the user has met and provide strategic context
2. PITCH GENERATION: Create tailored pitch angles using the user's products/value props
3. LEAD PRIORITIZATION: Rank contacts by strategic value, not just score numbers
4. RELATIONSHIP MAPPING: Identify connections between contacts (same company, industry, role)
5. ACTION PLANNING: Give specific next-step advice (e.g. "Visit booth X to meet Y because Z")
6. BRIEFINGS: Prepare the user before meetings with key talking points
7. EXHIBITOR INTELLIGENCE: Recommend which exhibitors to visit based on the user's goals

RESPONSE STYLE:
- Be strategic and direct, like a trusted advisor at the event
- Lead with the most important insight
- Use the user's own language (their products, value props, pitch style)
- Give SPECIFIC actions, not vague advice
- Reference actual contacts and data, never make things up
- When asked "who should I prioritize?", give a ranked list with clear reasoning tied to their goals
- When asked about pitch angles, tailor to the specific contact's company/role/industry
- Format responses with clear headers and bullet points for easy mobile reading
- Keep responses concise but actionable: the user is at an event and needs quick answers
- MIRROR the user's communication style. If they're direct, be direct. If they're casual, be casual.
- Be a PARTNER, not just an assistant: anticipate needs, connect dots between contacts, think ahead
- Write like you're texting a trusted colleague at the event, not writing a formal report
- When the user has a LinkedIn profile or company website, weave those insights naturally into advice
- Always frame advice in terms of the user's specific products, value props, and goals: make it feel like you truly understand their business

Important: Only use information from the provided contact/exhibitor context. Do not make up contact details or exhibitor info.`;

const PITCH_STYLES: Record<string, string> = {
  consultative: "Consultative (discover needs first, ask questions, position as partner)",
  direct: "Direct (lead with solution and ROI, get to the point fast)",
  challenger: "Challenger (reframe their thinking, share insights they haven't considered)",
  relationship: "Relationship (build trust first, find common ground, long-term play)",
};

function buildSystemPrompt(profile: any, exhibitors: any[]): string {
  let prompt = BASE_PROMPT;
  if (profile && Object.values(profile).some(Boolean)) {
    let c = "\n\nUSER CONTEXT (use this to deeply personalize ALL advice):";
    if (profile.full_name) c += `\n- User: ${profile.full_name}`;
    if (profile.job_title) c += `\n- Role: ${profile.job_title}`;
    if (profile.company_name) c += `\n- Company: ${profile.company_name}`;
    arr(profile.products).forEach((p: any, i: number) => {
      c += `\n- Product ${i + 1}: ${p?.name || ""}, ${p?.description || ""}`;
      if (p?.ideal_customer) c += ` (Ideal customer: ${p.ideal_customer})`;
    });
    if (arr(profile.target_industries).length) c += `\n- Target Industries: ${arr(profile.target_industries).join(", ")}`;
    if (arr(profile.target_roles).length) c += `\n- Target Roles: ${arr(profile.target_roles).join(", ")}`;
    if (arr(profile.target_company_sizes).length) c += `\n- Target Company Sizes: ${arr(profile.target_company_sizes).join(", ")}`;
    if (arr(profile.value_propositions).length) {
      c += "\n- Value Propositions:";
      for (const vp of arr(profile.value_propositions)) c += `\n  * ${vp}`;
    }
    if (profile.pitch_style) c += `\n- Pitch Style: ${PITCH_STYLES[profile.pitch_style] || profile.pitch_style}`;
    if (profile.current_event_name) c += `\n- Current Event: ${profile.current_event_name}`;
    if (profile.current_event_description) c += `\n- Event Details: ${profile.current_event_description}`;
    if (arr(profile.event_goals).length) {
      c += "\n- Event Goals:";
      for (const g of arr(profile.event_goals)) c += `\n  * ${g}`;
    }
    if (profile.linkedin_url) c += `\n- LinkedIn: ${profile.linkedin_url}`;
    if (profile.company_website) c += `\n- Company Website: ${profile.company_website}`;
    if (arr(profile.target_geographies).length) c += `\n- Target Geographies: ${arr(profile.target_geographies).join(", ")}`;
    c += "\n\nCRITICAL: Use ALL of the above context when giving advice. Every pitch suggestion must reference the user's actual products and value props. Every prioritization must consider their target market. Every briefing must connect to their event goals. Speak as if you truly know and understand this person: their business, their style, their goals.";
    prompt += c;
  }
  if (exhibitors.length) {
    let ec = `\n\nEXHIBITOR DATA (${exhibitors.length} exhibitors at the event):`;
    for (const ex of exhibitors.slice(0, 30)) {
      let line = `\n- ${ex.name}`;
      if (ex.booth) {
        line += ` (Booth: ${ex.booth}`;
        if (ex.hall) line += `, Hall: ${ex.hall}`;
        line += ")";
      }
      if (ex.category) line += `, ${ex.category}`;
      if (ex.country) line += ` [${ex.country}]`;
      if (ex.description) line += ` | ${String(ex.description).slice(0, 200)}`;
      ec += line;
    }
    prompt += ec;
  }
  return prompt;
}

function contactContext(retrieved: any[], allContacts: any[]): string {
  if (!retrieved.length && !allContacts.length) return "No contacts found in the database matching your query.";
  const parts: string[] = [];
  if (retrieved.length) {
    parts.push(`RELEVANT CONTACTS (semantic match for your query, ${retrieved.length} results):\n`);
    retrieved.forEach((meta, i) => {
      let info = `Contact ${i + 1}: ${meta.name || "N/A"}`;
      info += ` | Company: ${meta.company_name || "N/A"}`;
      info += ` | Email: ${meta.email || "N/A"}`;
      info += ` | Phone: ${meta.phone || "N/A"}`;
      if (meta.linkedin && meta.linkedin !== "N/A") info += ` | LinkedIn: ${meta.linkedin}`;
      if (meta.notes) info += ` | Notes: ${String(meta.notes).slice(0, 300)}`;
      if (meta.lead_score !== null && meta.lead_score !== undefined) {
        info += ` | Score: ${meta.lead_score}/100 (${meta.lead_temperature || "unscored"})`;
      }
      if (meta.lead_score_reasoning) info += ` | Reasoning: ${meta.lead_score_reasoning}`;
      const actions = meta.lead_recommended_actions;
      if (Array.isArray(actions) && actions.length) info += ` | Actions: ${actions.slice(0, 3).join("; ")}`;
      parts.push(info);
    });
  }
  if (allContacts.length > retrieved.length) {
    parts.push(`\nALL CONTACTS SUMMARY (${allContacts.length} total):`);
    for (const c of allContacts) {
      let s = `- ${c.name || "N/A"} at ${c.company_name || "N/A"}`;
      if (c.lead_score !== null && c.lead_score !== undefined) s += ` (Score: ${c.lead_score}, ${c.lead_temperature || "?"})`;
      parts.push(s);
    }
  }
  return parts.join("\n");
}

async function generateResponse(
  env: Env, augmentedQuery: string, retrieved: any[], history: any[], profile: any,
  allContacts: any[], exhibitors: any[],
): Promise<{ text: string; model: string }> {
  const messages: { role: string; content: any }[] = [{ role: "system", content: buildSystemPrompt(profile, exhibitors) }];
  for (const m of arr(history)) {
    messages.push({ role: m?.role === "assistant" ? "assistant" : "user", content: m?.content ?? "" });
  }
  messages.push({ role: "user", content: `${contactContext(retrieved, allContacts)}\n\n---\nUser Query: ${augmentedQuery}` });
  try {
    return await chat(env, messages, { maxTokens: 2000 });
  } catch (e) {
    // app.py fell back to direct Gemini and returned this text as a normal reply.
    return { text: `AI temporarily unavailable. Error: ${String((e as Error)?.message || e)}`, model: "" };
  }
}

// ---- intent detection + supplements ----
const RESEARCH_KEYWORDS = ["research", "look up", "find out about", "investigate", "tell me about", "what do you know about", "company info"];
const PITCH_KEYWORDS = ["pitch", "pitch deck", "presentation", "slides", "proposal", "generate pitch", "create pitch"];

function detectIntent(query: string): "research" | "pitch" | "chat" {
  const q = query.toLowerCase().trim();
  if (PITCH_KEYWORDS.some((k) => q.includes(k))) return "pitch";
  if (RESEARCH_KEYWORDS.some((k) => q.includes(k))) return "research";
  return "chat";
}

const RESEARCH_PROMPT_SUPPLEMENT = `
IMPORTANT: The user wants you to RESEARCH this company/person in depth. Provide a comprehensive analysis with these sections:

## Company Overview
What they do, their size, market position, key products/services

## Key Insights
Recent developments, competitive positioning, growth signals, challenges

## Role-Specific Analysis
What challenges someone in this contact's role typically faces, their likely priorities and KPIs

## Strategic Fit
How this contact/company aligns with the user's products and services. Identify specific pain points the user can address.

## Recommended Approach
Best pitch angle, talking points, potential objections to prepare for, and ideal next steps.

Be specific and actionable. This research will be used to prepare for a real sales conversation.
`;

const PITCH_PROMPT_SUPPLEMENT = `
IMPORTANT: The user wants you to generate a PITCH DECK for this contact. Create a compelling, personalized pitch.
Return your response as regular text with clear slide headers. Structure it as an 8-slide pitch:

## Slide 1: Opening
Personalized opening addressing the contact by name and their company. Hook them with a relevant insight about their industry.

## Slide 2: The Challenge
Their role-specific pain points backed by industry context. Show you understand their world.

## Slide 3: By The Numbers
Relevant industry data, market trends, competitor moves that create urgency.

## Slide 4: Cost of Inaction
What happens if they don't address these challenges. Make it concrete with examples.

## Slide 5: Our Approach
The user's methodology and philosophy. What makes their approach different.

## Slide 6: The Solution
Specific product/service mapping to the contact's pain points. Be concrete about value.

## Slide 7: Proof Points
Most relevant case studies, testimonials, or results from the user's profile.

## Slide 8: Next Steps
Clear CTA with a proposed meeting, demo, or follow-up action.

Make each slide compelling, concise, and personalized to this specific contact and their company.
`;

// ---- generative UI components ----
const EXHIBITOR_KEYWORDS = ["exhibitor", "booth", "visit", "hall", "vendor", "supplier", "who should i see"];
const SCORE_KEYWORDS = ["score", "lead", "hot", "warm", "cold", "rating", "qualify", "pipeline", "best contact"];

function generateUiComponents(
  query: string, responseText: string, retrieved: any[], allContacts: any[], exhibitors: any[],
): any[] {
  const components: any[] = [];
  const queryLower = query.toLowerCase();

  if (retrieved.length) components.push({ type: "contact_cards", data: retrieved.slice(0, 4) });

  if (EXHIBITOR_KEYWORDS.some((kw) => queryLower.includes(kw)) && exhibitors.length) {
    const respLower = responseText.toLowerCase();
    let matched = exhibitors.filter((e) => String(e.name || "").toLowerCase().includes(respLower)).slice(0, 4);
    if (!matched.length) matched = exhibitors.slice(0, 4);
    components.push({ type: "exhibitor_cards", data: matched });
  }

  if (SCORE_KEYWORDS.some((kw) => queryLower.includes(kw))) {
    const scored = allContacts.filter((c) => c.lead_score !== null && c.lead_score !== undefined);
    if (scored.length) {
      components.push({
        type: "score_summary",
        data: {
          total: allContacts.length,
          scored: scored.length,
          hot: scored.filter((c) => c.lead_temperature === "hot").length,
          warm: scored.filter((c) => c.lead_temperature === "warm").length,
          cold: scored.filter((c) => c.lead_temperature === "cold").length,
          top_contacts: [...scored].sort((a, b) => (b.lead_score ?? 0) - (a.lead_score ?? 0)).slice(0, 3),
        },
      });
    }
  }

  const actions: any[] = [];
  if (retrieved.length) {
    const firstc = retrieved[0];
    const cid = firstc.id || "";
    const name = firstc.name || "";
    if (!firstc.lead_score) actions.push({ label: `Score ${name}`, action: "score", contact_id: cid, icon: "fa-chart-bar" });
    actions.push({ label: `Research ${name}`, action: "research", contact_id: cid, icon: "fa-search" });
    actions.push({ label: `Pitch for ${name}`, action: "pitch", contact_id: cid, icon: "fa-file-powerpoint" });
  }
  if (actions.length) components.push({ type: "action_buttons", data: actions });

  let quickReplies: string[];
  if (retrieved.length) {
    const name = retrieved[0].name || "";
    const company = retrieved[0].company_name || "";
    quickReplies = [`Research ${company}`, `Generate pitch for ${name}`];
    if (!retrieved[0].lead_score) quickReplies.push(`Score ${name} as a lead`);
    quickReplies.push("Who should I visit next?");
  } else {
    quickReplies = ["Show my hot leads", "Who should I visit next?", "Summarize my contacts", "Which exhibitors should I visit?"];
  }
  components.push({ type: "quick_replies", data: quickReplies.slice(0, 4) });

  return components;
}

// ---- /converse/ ----
async function converse(ctx: Ctx) {
  const query = await requireQuery(ctx);
  const b = await ctx.body();
  try {
    const isAdminMode = !!b.admin_mode && !!ctx.user.is_admin;
    const topK = b.top_k ? Number(b.top_k) || (isAdminMode ? 8 : 4) : isAdminMode ? 8 : 4;

    const rows = await searchContacts(ctx.env, query, isAdminMode ? null : ctx.user.id, topK);
    const retrieved = rows.map((c: any) => contactOut(c, true));

    const profile = await loadProfile(ctx.env, ctx.user.id);
    const allContacts = isAdminMode
      ? await all(ctx.env, "SELECT name, company_name, lead_score, lead_temperature FROM contacts")
      : await all(ctx.env, "SELECT name, company_name, lead_score, lead_temperature FROM contacts WHERE user_id = ?", ctx.user.id);

    let exhibitors: any[] = [];
    try {
      exhibitors = await all(
        ctx.env,
        "SELECT name, booth, hall, category, country, description FROM exhibitors WHERE event_name = ? LIMIT 50",
        profile.current_event_name ?? "WHX Dubai 2026",
      );
    } catch (e) {
      console.log("[CHAT] Error loading exhibitors", String(e));
    }

    const intent = detectIntent(query);
    let augmented = query;
    if (intent === "research") augmented = query + "\n\n" + RESEARCH_PROMPT_SUPPLEMENT;
    else if (intent === "pitch") augmented = query + "\n\n" + PITCH_PROMPT_SUPPLEMENT;

    const { text, model } = await generateResponse(
      ctx.env, augmented, retrieved, arr(b.conversation_history), profile, allContacts, exhibitors,
    );

    const enriched = rows.map((c: any) => ({
      id: c.id,
      name: c.name || "N/A",
      email: c.email || "N/A",
      phone: c.phone || "N/A",
      linkedin: c.linkedin || "N/A",
      company_name: c.company_name || "N/A",
      lead_score: c.lead_score ?? null,
      lead_temperature: c.lead_temperature ?? null,
      lead_score_reasoning: c.lead_score_reasoning || "",
      lead_recommended_actions: pj(c.lead_recommended_actions, []),
      photo_base64: c.photo_base64 ?? "",
      source: c.source || "manual",
      notes: String(c.notes || "").slice(0, 200),
    }));

    const uiComponents = generateUiComponents(query, text, enriched, allContacts, exhibitors);
    if (intent === "research" && enriched.length) {
      uiComponents.unshift({
        type: "research_card",
        data: {
          contact_id: enriched[0].id || "", contact_name: enriched[0].name || "",
          company: enriched[0].company_name || "", saved: false,
        },
      });
    } else if (intent === "pitch" && enriched.length) {
      uiComponents.unshift({
        type: "pitch_preview",
        data: {
          contact_id: enriched[0].id || "", contact_name: enriched[0].name || "",
          company: enriched[0].company_name || "",
        },
      });
    }

    return {
      status: "success",
      response: text,
      model,
      retrieved_contacts: enriched,
      ui_components: uiComponents,
      intent,
      query,
    };
  } catch (e) {
    throw new HttpError(500, `Conversation error: ${String((e as Error)?.message || e)}`);
  }
}

// ---- lead scoring (port of score_contact_with_gemini) ----
const SCORE_FALLBACK = {
  score: 50,
  temperature: "warm",
  reasoning: "Scoring temporarily unavailable.",
  recommended_actions: ["Retry scoring later"],
  breakdown: { profile_fit: 10, role_relevance: 10, company_fit: 10, engagement: 10, timing: 10 },
};

class ScoreTimeout extends Error {}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new ScoreTimeout("timeout")), ms);
    p.then((v) => { clearTimeout(t); resolve(v); }, (e) => { clearTimeout(t); reject(e); });
  });
}

function scorePrompt(meta: any, profile: any): string {
  return `You are a lead scoring expert. Score this contact against the user's profile and event context.

USER PROFILE:
- Company: ${profile.company_name ?? "N/A"}
- Job Title: ${profile.job_title ?? "N/A"}
- Products/Services: ${JSON.stringify(profile.products ?? [])}
- Target Industries: ${JSON.stringify(profile.target_industries ?? [])}
- Target Roles: ${JSON.stringify(profile.target_roles ?? [])}
- Target Company Sizes: ${JSON.stringify(profile.target_company_sizes ?? [])}
- Target Geographies: ${JSON.stringify(profile.target_geographies ?? [])}
- Value Propositions: ${JSON.stringify(profile.value_propositions ?? [])}
- Event: ${profile.current_event_name ?? "N/A"}
- Event Goals: ${JSON.stringify(profile.event_goals ?? [])}

CONTACT:
- Name: ${meta.name ?? "N/A"}
- Company: ${meta.company_name ?? "N/A"}
- Email: ${meta.email ?? "N/A"}
- Phone: ${meta.phone ?? "N/A"}
- LinkedIn: ${meta.linkedin ?? "N/A"}
- Notes: ${meta.notes ?? ""}

Score this contact on 5 dimensions (each 0-20, total 0-100):
1. profile_fit 2. role_relevance 3. company_fit 4. engagement 5. timing

Return ONLY a valid JSON object:
{"score": <0-100>, "temperature": "<hot|warm|cold>", "reasoning": "<2-3 sentences>", "recommended_actions": ["action1", "action2", "action3"], "breakdown": {"profile_fit": <0-20>, "role_relevance": <0-20>, "company_fit": <0-20>, "engagement": <0-20>, "timing": <0-20>}}

Rules: hot >= 70, warm = 40-69, cold < 40. Return ONLY raw JSON.`;
}

/** Returns the fallback dict on model failure; throws ScoreTimeout after 30s like app.py's wait_for. */
async function scoreContact(env: Env, meta: any, profile: any) {
  let text: string;
  try {
    ({ text } = await withTimeout(chat(env, [{ role: "user", content: scorePrompt(meta, profile) }], { maxTokens: 700 }), 30000));
  } catch (e) {
    if (e instanceof ScoreTimeout) throw e;
    console.log("[SCORING] model error", String(e));
    return { ...SCORE_FALLBACK };
  }
  try {
    const r = extractJson(text);
    const score = Number.isFinite(Number(r.score)) ? Math.min(100, Math.max(0, Number(r.score))) : 50;
    return {
      score,
      temperature: score >= 70 ? "hot" : score >= 40 ? "warm" : "cold",
      reasoning: String(r.reasoning ?? ""),
      recommended_actions: arr(r.recommended_actions),
      breakdown: r.breakdown && typeof r.breakdown === "object" ? r.breakdown : {},
    };
  } catch (e) {
    console.log("[SCORING] bad model reply", String(e));
    return { ...SCORE_FALLBACK };
  }
}

const scoreMeta = (c: any) => ({
  name: c.name, company_name: c.company_name || "N/A", email: c.email || "N/A",
  phone: c.phone || "N/A", linkedin: c.linkedin || "N/A", notes: c.notes || "",
});

async function saveScore(env: Env, contactId: string, r: any) {
  await run(
    env,
    "UPDATE contacts SET lead_score = ?, lead_temperature = ?, lead_score_reasoning = ?, lead_score_breakdown = ?, lead_recommended_actions = ?, updated_at = ? WHERE id = ?",
    r.score, r.temperature, r.reasoning, sj(r.breakdown), sj(r.recommended_actions), now(), contactId,
  );
}

async function scoreOne(ctx: Ctx) {
  const c = await first<any>(ctx.env, "SELECT * FROM contacts WHERE id = ? AND user_id = ?", ctx.params.contact_id, ctx.user.id);
  if (!c) throw new HttpError(404, "Contact not found");
  const profile = await loadProfile(ctx.env, ctx.user.id);
  let r: any;
  try {
    r = await scoreContact(ctx.env, scoreMeta(c), profile);
  } catch (e) {
    if (e instanceof ScoreTimeout) throw new HttpError(504, "Scoring timed out: Gemini API unresponsive");
    throw e;
  }
  await saveScore(ctx.env, c.id, r);
  return {
    status: "success", contact_id: c.id, score: r.score, temperature: r.temperature,
    reasoning: r.reasoning, recommended_actions: r.recommended_actions, breakdown: r.breakdown,
  };
}

// shortcut: app.py scores every unscored contact in one call; Workers requests are time-boxed,
// so 50 per call and the response says so. Raise only with a queue/trigger instead.
const SCORE_ALL_CAP = 50;

async function scoreAll(ctx: Ctx) {
  const profile = await loadProfile(ctx.env, ctx.user.id);
  const pending = (await first<any>(ctx.env, "SELECT count(*) n FROM contacts WHERE user_id = ? AND lead_score IS NULL", ctx.user.id))?.n ?? 0;
  const unscored = await all<any>(
    ctx.env, "SELECT * FROM contacts WHERE user_id = ? AND lead_score IS NULL LIMIT ?", ctx.user.id, SCORE_ALL_CAP,
  );
  let scored = 0, errors = 0;
  for (const c of unscored) {
    try {
      await saveScore(ctx.env, c.id, await scoreContact(ctx.env, scoreMeta(c), profile));
      scored++;
    } catch (e) {
      console.log("[SCORING] Error scoring", c.id, String(e));
      errors++;
    }
  }
  const total = (await first<any>(ctx.env, "SELECT count(*) n FROM contacts WHERE user_id = ?", ctx.user.id))?.n ?? 0;
  const out: any = { status: "success", scored, skipped: total - scored - errors, errors, total };
  if (pending > unscored.length) out.message = `Capped at ${SCORE_ALL_CAP} contacts per call; ${pending - unscored.length} left, run again.`;
  return out;
}

// ---- enrichment (n8n callback, no AI) ----
async function enrich(ctx: Ctx) {
  const c = await first<any>(ctx.env, "SELECT * FROM contacts WHERE id = ?", ctx.params.contact_id);
  if (!c || (c.user_id !== ctx.user.id && !ctx.user.is_admin)) throw new HttpError(404, "Contact not found");
  const b = await ctx.body();
  if (b.links !== undefined && b.links !== null && !Array.isArray(b.links)) throw new HttpError(422, "links must be a list");

  let notes = c.notes || "";
  if (b.notes_append) notes = notes ? `${notes}\n\n---\n${b.notes_append}` : b.notes_append;
  const links = pj<any[]>(c.links, []);
  for (const l of arr(b.links)) {
    l.added_by = l.added_by ?? "n8n";
    links.push(l);
  }

  await run(ctx.env, "UPDATE contacts SET notes = ?, links = ?, updated_at = ? WHERE id = ?", notes, sj(links), now(), c.id);
  await run(
    ctx.env,
    "UPDATE shared_contacts SET notes = ?, links = ?, enriched = 1, enriched_at = ? WHERE original_contact_id = ?",
    notes, sj(links), now(), c.id,
  );
  reindexLater(ctx.env, ctx.exec, [{ ...c, notes, links }]);
  return { status: "success", message: "Contact enriched successfully", contact_id: c.id, notes: notes || "", links };
}

// ---- pipeline triggers: no background runner on Cloudflare ----
const pipelineGone = async (): Promise<Response | unknown> => {
  throw new HttpError(501, "The AI research pipeline is not available on the Cloudflare build yet.");
};

export const routes: Route[] = [
  ["POST", "/search/", search],
  ["POST", "/converse/", converse],
  ["POST", "/contact/:contact_id/enrich", enrich],
  ["POST", "/contact/:contact_id/score", scoreOne],
  ["POST", "/contacts/score_all", scoreAll],
  ["POST", "/contact/:contact_id/pipeline/run", pipelineGone],
  ["POST", "/contact/:contact_id/pipeline/retry", pipelineGone],
];
