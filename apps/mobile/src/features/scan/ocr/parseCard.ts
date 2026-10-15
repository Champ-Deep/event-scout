import type { OcrLine } from './pipeline';

/**
 * Business-card lines → contact fields, on the phone (no AI). Emails, phones, websites and
 * LinkedIn come from patterns; name / title / company are ranked by keywords, the email
 * domain and text size (the name is usually the biggest text on a card).
 */

export interface CardField { value: string; confidence: number }
export interface ParsedCard {
  firstName: CardField | null;
  lastName: CardField | null;
  title: CardField | null;
  company: CardField | null;
  email: CardField | null;
  phone: CardField | null;
  website: CardField | null;
  linkedin: CardField | null;
  /** 0–1 overall: how much of a usable contact was read */
  confidence: number;
  /** the lines that weren't used — kept in notes so nothing is lost */
  otherLines: string[];
}

const EMAIL = /[A-Z0-9._%+-]+\s?@\s?[A-Z0-9.-]+\.[A-Z]{2,}/i;
const LINKEDIN = /(?:https?:\/\/)?(?:[a-z]{2,3}\.)?linkedin\.com\/(?:in|pub)\/[A-Z0-9_%-]+/i;
const URL = /(?:https?:\/\/)?(?:www\.)?[A-Z0-9-]+(?:\.[A-Z0-9-]+)*\.(?:com|net|org|io|ai|co|in|ae|uk|de|fr|biz|info|tech|app|dev|me|us|ca|au|sg|sa|qa)(?:\.[a-z]{2})?(?:\/\S*)?/i;
const PHONE_LABEL = /^(?:t|tel|telephone|phone|ph|p|m|mob|mobile|cell|c|o|off|office|d|direct|w|work|f|fax|whatsapp)\b\.?\s*[:.]?\s*/i;
const FAX = /\b(?:f|fax)\b\s*[:.]?/i;
const MOBILE = /\b(?:m|mob|mobile|cell|c|whatsapp)\b\s*[:.]?/i;

const TITLE_WORDS = [
  'ceo', 'cto', 'cfo', 'coo', 'cmo', 'cio', 'cro', 'founder', 'co-founder', 'cofounder', 'owner', 'partner', 'president',
  'chairman', 'director', 'manager', 'head', 'lead', 'leader', 'vp', 'vice', 'officer', 'chief', 'executive', 'engineer',
  'developer', 'architect', 'consultant', 'specialist', 'analyst', 'advisor', 'adviser', 'associate', 'coordinator',
  'representative', 'rep', 'sales', 'marketing', 'business development', 'account', 'designer', 'scientist', 'researcher',
  'doctor', 'dr.', 'physician', 'surgeon', 'nurse', 'professor', 'lecturer', 'principal', 'administrator', 'supervisor',
  'strategist', 'recruiter', 'attorney', 'lawyer', 'accountant', 'agent', 'broker', 'intern', 'trainee', 'senior', 'sr.',
  'junior', 'jr.', 'assistant', 'secretary', 'editor', 'producer', 'operations', 'procurement', 'buyer', 'purchasing',
  'general manager', 'md', 'gm', 'evp', 'svp', 'avp',
];
const COMPANY_SUFFIX = /\b(?:inc|inc\.|ltd|ltd\.|llc|l\.l\.c|llp|plc|gmbh|ag|sa|s\.a\.|bv|b\.v\.|nv|pvt|pte|pty|co\.|corp|corporation|company|limited|group|holdings?|technologies|technology|tech|solutions|systems|services|consulting|labs?|studios?|partners|industries|international|global|enterprises?|networks?|software|healthcare|pharma|medical|bank|capital|ventures|fz-?llc|fze|fzco|dmcc|wll|est\.?)\b/i;
const ADDRESS = /\b(?:street|st\.|road|rd\.?|avenue|ave\.?|blvd|boulevard|lane|ln\.|drive|dr\b|floor|flr|suite|ste\.?|building|bldg|tower|p\.?\s?o\.?\s?box|po box|zip|city|district|sector|plot|block|near|opp\.?|square|park|business bay|free zone|dubai|abu dhabi|sharjah|riyadh|doha|london|new york|bangalore|bengaluru|mumbai|delhi|chennai|hyderabad|pune|singapore|uae|u\.a\.e|india|usa|united states|united kingdom|germany|france|saudi arabia|qatar|oman|kuwait|bahrain)\b/i;

function field(value: string | null | undefined, confidence: number): CardField | null {
  const v = (value || '').replace(/\s+/g, ' ').trim();
  return v ? { value: v, confidence: Math.max(0, Math.min(1, confidence)) } : null;
}

function digitsOf(s: string) { return (s.match(/\d/g) || []).length; }

function isPhoneLike(s: string) {
  const t = s.replace(PHONE_LABEL, '');
  const d = digitsOf(t);
  return d >= 7 && d <= 16 && /^[+\d()\s./-]+$/.test(t.replace(/\b(?:ext|x)\.?\s*\d+$/i, '').trim());
}

function cleanPhone(s: string) {
  return s.replace(PHONE_LABEL, '').replace(/[^\d+()\s-]/g, ' ').replace(/\s+/g, ' ').trim();
}

function titleCase(s: string) {
  return s.toLowerCase().replace(/(^|[\s'-])(\p{L})/gu, (_, a, b) => a + b.toUpperCase());
}

function looksLikeName(s: string) {
  if (/\d|@|\/|www\.|\.com/i.test(s)) return false;
  const words = s.split(/\s+/).filter(Boolean);
  if (words.length < 1 || words.length > 4) return false;
  if (words.some((w) => w.length > 20)) return false;
  // Letters, dots (initials), hyphens and apostrophes only
  return words.every((w) => /^[\p{L}][\p{L}.'-]*$/u.test(w));
}

function titleScore(s: string) {
  const l = ` ${s.toLowerCase()} `;
  let hits = 0;
  for (const w of TITLE_WORDS) if (l.includes(` ${w} `) || l.includes(` ${w},`) || l.includes(` ${w}/`) || l.startsWith(` ${w}`)) hits++;
  return hits;
}

function domainOf(email: string | null, website: string | null): string | null {
  const host = email ? email.split('@')[1] : website ? website.replace(/^https?:\/\//i, '').replace(/^www\./i, '').split('/')[0] : null;
  if (!host) return null;
  const parts = host.toLowerCase().split('.');
  const generic = ['gmail', 'yahoo', 'hotmail', 'outlook', 'live', 'icloud', 'aol', 'proton', 'protonmail', 'zoho', 'yandex', 'mail', 'rediffmail'];
  const core = parts.length > 2 && parts[parts.length - 2]!.length <= 3 ? parts[parts.length - 3] : parts[parts.length - 2];
  return core && !generic.includes(core) ? core : null;
}

export function parseCard(lines: OcrLine[]): ParsedCard {
  let email: CardField | null = null;
  let linkedin: CardField | null = null;
  let website: CardField | null = null;
  const phones: { value: string; conf: number; mobile: boolean }[] = [];
  const rest: OcrLine[] = [];

  for (const line of lines) {
    let text = line.text.replace(/\s+/g, ' ').trim();
    if (!text) continue;
    let used = false;

    const li = text.match(LINKEDIN);
    if (li && !linkedin) { linkedin = field(li[0].replace(/\s/g, ''), line.score); text = text.replace(li[0], ' '); used = true; }

    const em = text.match(EMAIL);
    if (em && !email) { email = field(em[0].replace(/\s/g, '').toLowerCase(), line.score); text = text.replace(em[0], ' '); used = true; }

    // A line can hold several numbers ("T +971 4 123 4567 | M +971 50 123 4567")
    for (const part of text.split(/\s*[|•·,;]\s*|\s{3,}/)) {
      if (!part.trim()) continue;
      if (isPhoneLike(part) && !FAX.test(part)) {
        phones.push({ value: cleanPhone(part), conf: line.score, mobile: MOBILE.test(part) });
        text = text.replace(part, ' ');
        used = true;
      }
    }

    const url = text.match(URL);
    if (url && !website && !/@/.test(url[0])) { website = field(url[0], line.score * 0.95); text = text.replace(url[0], ' '); used = true; }

    text = text.replace(/\s+/g, ' ').replace(/^[\s|•·,:;-]+|[\s|•·,:;-]+$/g, '').trim();
    if (text.length >= 2 && (!used || /\p{L}{3,}/u.test(text))) rest.push({ ...line, text });
  }

  // Drop address lines and labels before ranking the rest.
  const candidates = rest.filter((l) => !(ADDRESS.test(l.text) && digitsOf(l.text) > 0) && !/^(?:email|e-mail|web|website|tel|phone|mobile|fax|address)\s*:?$/i.test(l.text));
  const domain = domainOf(email?.value ?? null, website?.value ?? null);
  const heights = candidates.map((l) => l.lineHeight).sort((a, b) => b - a);
  const tallest = heights[0] || 1;

  // ---- name ----
  let nameLine: OcrLine | null = null;
  let nameConf = 0;
  for (const l of candidates) {
    if (!looksLikeName(l.text) || titleScore(l.text) > 0 || COMPANY_SUFFIX.test(l.text)) continue;
    const words = l.text.split(/\s+/).length;
    let s = (l.lineHeight / tallest) * 0.55 + (words >= 2 && words <= 3 ? 0.3 : 0.1);
    if (domain && l.text.toLowerCase().replace(/[^a-z]/g, '').includes(domain)) s -= 0.4; // that's the brand, not a person
    if (email && l.text.split(/\s+/).some((w) => w.length > 2 && email!.value.split('@')[0]!.includes(w.toLowerCase().replace(/[^a-z]/g, '')))) s += 0.35;
    if (s > nameConf) { nameConf = s; nameLine = l; }
  }

  // ---- title ----
  let titleLine: OcrLine | null = null;
  let best = 0;
  for (const l of candidates) {
    if (l === nameLine) continue;
    const hits = titleScore(l.text);
    if (hits > best && l.text.length <= 70) { best = hits; titleLine = l; }
  }

  // ---- company ----
  let companyLine: OcrLine | null = null;
  let companyConf = 0;
  for (const l of candidates) {
    if (l === nameLine || l === titleLine) continue;
    let s = 0;
    if (COMPANY_SUFFIX.test(l.text)) s += 0.55;
    if (domain && l.text.toLowerCase().replace(/[^a-z0-9]/g, '').includes(domain)) s += 0.5;
    if (l.lineHeight >= tallest * 0.8) s += 0.15; // logos / brand text are large too
    if (digitsOf(l.text) > 3) s -= 0.3;
    if (s > companyConf) { companyConf = s; companyLine = l; }
  }

  const nameParts = nameLine ? nameLine.text.split(/\s+/) : [];
  // Names start with a capital: OCR often reads a capital I as a lower-case l ("lyer" → "Iyer").
  const fixCase = (s: string) => {
    const t = s === s.toUpperCase() && s.length > 3 ? titleCase(s) : s;
    return t.split(/\s+/).map((w) => (/^l[a-z]/.test(w) ? 'I' + w.slice(1) : w.charAt(0).toUpperCase() + w.slice(1))).join(' ');
  };
  const firstName = nameParts.length ? field(fixCase(nameParts[0]!), nameLine!.score * Math.min(1, nameConf + 0.2)) : null;
  const lastName = nameParts.length > 1 ? field(fixCase(nameParts.slice(1).join(' ')), nameLine!.score * Math.min(1, nameConf + 0.2)) : null;
  const title = titleLine ? field(titleLine.text, titleLine.score * Math.min(1, 0.6 + best * 0.2)) : null;
  let company = companyLine && companyConf >= 0.3 ? field(companyLine.text, companyLine.score * Math.min(1, companyConf + 0.3)) : null;
  if (!company && domain) company = field(domain.charAt(0).toUpperCase() + domain.slice(1), 0.4); // from the email/website

  phones.sort((a, b) => Number(b.mobile) - Number(a.mobile));
  const phone = phones[0] ? field(phones[0].value, phones[0].conf) : null;

  const used = new Set([nameLine, titleLine, companyLine].filter(Boolean));
  const otherLines = candidates.filter((l) => !used.has(l)).map((l) => l.text).slice(0, 8);

  // Overall: a usable contact needs a name plus a way to reach them.
  const weights: [CardField | null, number][] = [[firstName, 0.3], [email, 0.25], [phone, 0.15], [company, 0.15], [title, 0.1], [lastName, 0.05]];
  const confidence = weights.reduce((s, [f, w]) => s + (f ? f.confidence * w : 0), 0);

  return { firstName, lastName, title, company, email, phone, website, linkedin, confidence: Math.round(confidence * 100) / 100, otherLines };
}
