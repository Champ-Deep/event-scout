// COPY of apps/mobile/src/features/ask/openui/spec.ts — the OpenUI component set the backend's prompt
// (apps/backend/openui_prompt.txt) is generated from. Keep the two identical: if you change the mobile one,
// copy it here too, so the admin console renders every component the AI may send.
// The Ask AI component set for OpenUI (https://github.com/thesysdev/openui).
//
// One definition, used twice:
//   - the app (library.tsx) plugs in React Native renderers;
//   - scripts/gen-openui-prompt.mjs plugs in null renderers and writes the system prompt the
//     backend sends to the model (apps/backend/openui_prompt.txt).
// After changing anything here, run `node scripts/gen-openui-prompt.mjs` so the model and the
// app agree on the components. No React Native / JSX imports in this file (it runs in Node).
import { z } from 'zod/v4'; // OpenUI needs Zod 4 schemas (zod@3.25 ships them under zod/v4)

type Define = (spec: { name: string; description: string; props: any; component: any }) => any;
type Renderers = Record<string, any>;

export const OPENUI_ROOT = 'Card';

export function defineAskComponents(define: Define, render: Renderers) {
  const Text = define({
    name: 'Text',
    description: 'A paragraph of text. variant: "heading" for a short title, "body" (default) for normal text, "caption" for small secondary text. Supports **bold**.',
    props: z.object({
      content: z.string().describe('The text'),
      variant: z.enum(['heading', 'body', 'caption']).optional().describe('heading | body | caption'),
    }),
    component: render.Text,
  });

  const ContactCard = define({
    name: 'ContactCard',
    description: "One of the user's contacts. Use the exact name from the contact data so the app can open it.",
    props: z.object({
      name: z.string().describe('Full name exactly as in the contact data'),
      company: z.string().optional().describe('Company'),
      title: z.string().optional().describe('Job title'),
      score: z.number().optional().describe('Lead score 0-100, only if known'),
      note: z.string().optional().describe('One short line: why this person matters right now'),
    }),
    component: render.ContactCard,
  });

  const ContactList = define({
    name: 'ContactList',
    description: 'A titled group of ContactCards, e.g. "Top 3 to follow up". Order them by priority.',
    props: z.object({
      contacts: z.array(ContactCard.ref).describe('ContactCard items'),
      title: z.string().optional().describe('Heading for the group'),
    }),
    component: render.ContactList,
  });

  const Stats = define({
    name: 'Stats',
    description: 'A row of 2-4 headline numbers, e.g. contacts met, hot leads, average score.',
    props: z.object({
      items: z.array(z.object({
        label: z.string().describe('Short label'),
        value: z.string().describe('The number or short value, e.g. "12" or "78%"'),
      })).describe('2-4 stats'),
    }),
    component: render.Stats,
  });

  const BarChart = define({
    name: 'BarChart',
    description: 'Horizontal bars comparing a few categories (e.g. contacts per company, leads per level). Max 8 bars.',
    props: z.object({
      data: z.array(z.object({
        label: z.string().describe('Category'),
        value: z.number().describe('Numeric value'),
      })).describe('Bars'),
      title: z.string().optional().describe('Chart title'),
    }),
    component: render.BarChart,
  });

  const Steps = define({
    name: 'Steps',
    description: 'A numbered action plan: concrete next steps in order.',
    props: z.object({
      steps: z.array(z.string()).describe('Each step as one sentence'),
      title: z.string().optional().describe('Heading, e.g. "Your next 3 moves"'),
    }),
    component: render.Steps,
  });

  const Callout = define({
    name: 'Callout',
    description: 'A highlighted single insight. tone: "insight" (default), "tip", or "warning".',
    props: z.object({
      text: z.string().describe('The message'),
      tone: z.enum(['insight', 'tip', 'warning']).optional().describe('insight | tip | warning'),
    }),
    component: render.Callout,
  });

  const Draft = define({
    name: 'Draft',
    description: 'A ready-to-send message the user can copy: follow-up email, LinkedIn note or text. Put the full message in body.',
    props: z.object({
      channel: z.enum(['email', 'linkedin', 'text']).describe('email | linkedin | text'),
      body: z.string().describe('The full message'),
      to: z.string().optional().describe('Recipient name'),
      subject: z.string().optional().describe('Email subject (email only)'),
    }),
    component: render.Draft,
  });

  const Exhibitor = define({
    name: 'Exhibitor',
    description: 'An exhibitor at the active event worth visiting, with where to find them.',
    props: z.object({
      name: z.string().describe('Exhibitor name'),
      booth: z.string().optional().describe('Booth / stand'),
      hall: z.string().optional().describe('Hall'),
      reason: z.string().optional().describe('Why visit, one line'),
    }),
    component: render.Exhibitor,
  });

  const Card = define({
    name: 'Card',
    description: 'The root container for an answer. Holds the other components in reading order.',
    props: z.object({
      children: z.array(z.union([
        Text.ref, ContactCard.ref, ContactList.ref, Stats.ref, BarChart.ref,
        Steps.ref, Callout.ref, Draft.ref, Exhibitor.ref,
      ])).describe('The blocks of the answer, in order'),
    }),
    component: render.Card,
  });

  return [Text, ContactCard, ContactList, Stats, BarChart, Steps, Callout, Draft, Exhibitor, Card];
}

export const OPENUI_PROMPT_OPTIONS = {
  inlineMode: true,
  preamble:
    'Your answers are shown in a mobile app that can render rich UI. Write normal conversational text, and when structure helps ' +
    '(several contacts, numbers, a plan, a message to send, exhibitors to visit) add ONE openui-lang block for it.',
  additionalRules: [
    'Short factual answers and small talk: plain text only, no UI block.',
    'Wrap every UI block in a single root Card.',
    'Use ContactCard / ContactList whenever you name specific contacts; copy their names exactly from the contact data.',
    'Never invent contacts, scores, companies or exhibitors. Only use what is in the provided data.',
    'Use Draft for any message the user should send (email, LinkedIn note, text).',
    'Use Stats or BarChart only for real numbers from the data.',
    'Keep text inside components short; put longer explanation in the plain text around the block.',
    'Arguments are positional ONLY. Never write name: value. Wrong: Draft("email", "Hi…", to: "Tom", subject: "Demo"). Right: Draft("email", "Hi…", "Tom", "Demo"). To skip an optional argument in the middle, pass null.',
    'This is a chat, not a dashboard: every UI block is a complete new answer. Always define root again; never output only the changed statements of an earlier block.',
  ],
  examples: [
    'root = Card([intro, people, plan])\nintro = Text("Your three hottest leads from today", "heading")\npeople = ContactList([a, b], "Follow up first")\na = ContactCard("Priya Shah", "Northwind Health", "VP Operations", 88, "Asked for pricing at the booth")\nb = ContactCard("Tom Berg", "Acme Labs", "CTO", 81, "Evaluating vendors this quarter")\nplan = Steps(["Email Priya the pricing deck today", "Book a 20-minute call with Tom"], "Next moves")',
    'root = Card([d])\nd = Draft("email", "Hi Priya,\\n\\nThanks for stopping by...", "Priya Shah", "Great meeting you at the show")',
  ],
};
