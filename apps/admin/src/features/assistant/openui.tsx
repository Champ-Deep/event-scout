// The AI assistant's rich answers, rendered with OpenUI (https://github.com/thesysdev/openui).
// The model writes plain text plus ```openui-lang blocks; the blocks use the component set in
// openuiSpec.ts (shared with the mobile app's Ask AI). Here every component gets a web renderer.
import React, { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { Renderer, createLibrary, defineComponent } from '@openuidev/react-lang';
import { Icon, type IconName } from '../../ui/icons';
import { MOTION } from '../../ui/tokens';
import { cx, levelOf } from '../../ui/kit';
import type { ConverseContact } from '../../lib/types';
import { defineAskComponents, OPENUI_ROOT } from './openuiSpec';

/* ------------------------------------------------------------------ context: what the answer can link to */

export interface AnswerLinks {
  /** the contacts the backend retrieved for this answer — ContactCards with a matching name open them */
  contacts: ConverseContact[];
  onContact?: ((id: string) => void) | undefined;
}
const LinksContext = createContext<AnswerLinks>({ contacts: [] });
const norm = (s?: string | null) => (s || '').trim().toLowerCase().replace(/\s+/g, ' ');
function useContactId(name?: string): string | undefined {
  const { contacts } = useContext(LinksContext);
  const n = norm(name);
  if (!n) return undefined;
  return (contacts.find((c) => norm(c.name) === n) || contacts.find((c) => norm(c.name).startsWith(n) || n.startsWith(norm(c.name))))?.id;
}
const initials = (name: string) => name.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]!.toUpperCase()).join('') || '?';

/* ------------------------------------------------------------------ markdown (plain-text parts) */

/** Inline: **bold**, *italic*, `code`, [text](https://link) — links only to http(s). */
function inline(s: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]+\]\(https?:\/\/[^)\s]+\))/g;
  let last = 0;
  for (const m of s.matchAll(re)) {
    if (m.index! > last) out.push(s.slice(last, m.index));
    const t = m[0];
    const k = out.length;
    if (t.startsWith('**')) out.push(<strong key={k} className="font-semibold text-foreground">{t.slice(2, -2)}</strong>);
    else if (t.startsWith('`')) out.push(<code key={k} className="rounded bg-secondary px-1 py-px font-mono text-[12px]">{t.slice(1, -1)}</code>);
    else if (t.startsWith('[')) {
      const lm = /^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/.exec(t)!;
      out.push(<a key={k} href={lm[2]} target="_blank" rel="noreferrer" className="font-medium text-brand-700 underline-offset-2 hover:underline">{lm[1]}</a>);
    } else out.push(<em key={k}>{t.slice(1, -1)}</em>);
    last = m.index! + t.length;
  }
  if (last < s.length) out.push(s.slice(last));
  return out;
}

/** Paragraphs, headings, bullet / numbered lists and the inline marks above. */
export function Markdown({ text }: { text: string }) {
  const blocks: React.ReactNode[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  const flush = () => {
    if (!list) return;
    const L = list.ordered ? 'ol' : 'ul';
    blocks.push(<L key={blocks.length} className={cx('ml-5 space-y-1', list.ordered ? 'list-decimal' : 'list-disc marker:text-brand-400')}>{list.items.map((it, i) => <li key={i}>{inline(it)}</li>)}</L>);
    list = null;
  };
  text.split('\n').forEach((raw) => {
    const line = raw.trimEnd();
    const b = /^\s*[-*•]\s+(.*)/.exec(line);
    const n = /^\s*\d+[.)]\s+(.*)/.exec(line);
    if (b || n) {
      const ordered = !!n;
      if (!list || list.ordered !== ordered) { flush(); list = { ordered, items: [] }; }
      list.items.push((b || n)![1]!);
    } else if (!line.trim()) {
      flush();
    } else {
      flush();
      const h = /^(#{1,4})\s+(.*)/.exec(line);
      blocks.push(h
        ? <p key={blocks.length} className={cx('font-semibold text-foreground', h[1]!.length <= 2 ? 'text-[15px]' : 'text-sm')}>{inline(h[2]!)}</p>
        : <p key={blocks.length}>{inline(line)}</p>);
    }
  });
  flush();
  return <div className="space-y-2 text-sm leading-relaxed text-body">{blocks}</div>;
}

/* ------------------------------------------------------------------ the components */

const appear = (i = 0) => ({ initial: { opacity: 0, y: 6 }, animate: { opacity: 1, y: 0 }, transition: { duration: MOTION.base, ease: MOTION.ease, delay: Math.min(i, 10) * 0.04 } });

function TextR({ props }: any) {
  const { content = '', variant = 'body' } = props;
  if (variant === 'heading') return <h4 className="text-[15px] font-bold leading-6 text-foreground">{content}</h4>;
  if (variant === 'caption') return <p className="text-xs text-muted-foreground">{content}</p>;
  return <Markdown text={String(content)} />;
}

function ContactCardR({ props }: any) {
  const { name = '', company, title, score, note } = props;
  const { onContact } = useContext(LinksContext);
  const id = useContactId(name);
  const lvl = typeof score === 'number' ? levelOf(score) : null;
  const sub = [title, company].filter(Boolean).join(' · ');
  const body = (
    <>
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-brand-100 text-xs font-bold text-brand-700 dark:bg-brand-100 dark:text-brand-800">{initials(name)}</span>
      <span className="min-w-0 flex-1">
        <span className="block break-words text-sm font-semibold text-foreground">{name}</span>
        {sub ? <span className="block break-words text-xs text-muted-foreground">{sub}</span> : null}
        {note ? <span className="mt-0.5 block text-xs leading-snug text-body">{note}</span> : null}
      </span>
      {typeof score === 'number' ? (
        <span className="shrink-0 rounded-lg px-2 py-0.5 text-xs font-bold tabular-nums"
          style={{ background: `color-mix(in oklab, var(--heat-${lvl?.key ?? 'none'}) 16%, transparent)`, color: `color-mix(in oklab, var(--heat-${lvl?.key ?? 'none'}) 75%, var(--foreground))` }}
          title={lvl ? `${lvl.label} lead` : undefined}>{Math.round(score)}</span>
      ) : null}
      {id && onContact ? <Icon name="chevronRight" size={15} className="shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-brand-600" /> : null}
    </>
  );
  const cls = 'group flex w-full items-center gap-3 rounded-xl border bg-card px-3 py-2.5 text-left shadow-card';
  return id && onContact
    ? <button type="button" onClick={() => onContact(id)} title={`Open ${name}`} className={cx(cls, 'cursor-pointer transition-colors hover:border-brand-300')}>{body}</button>
    : <div className={cls} title={id ? undefined : "Not one of this answer's contacts, so it can't be opened"}>{body}</div>;
}

function Block({ title, icon, children }: { title?: string; icon: IconName; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      {title ? <div className="flex items-center gap-1.5 text-[11px] font-extrabold uppercase tracking-[0.07em] text-brand-700"><Icon name={icon} size={12} />{title}</div> : null}
      {children}
    </div>
  );
}

function ContactListR({ props, renderNode }: any) {
  const items = Array.isArray(props.contacts) ? props.contacts : [];
  if (!items.length) return null;
  return (
    <Block title={props.title} icon="users">
      <div className="grid gap-2 sm:grid-cols-2">{items.map((c: any, i: number) => <motion.div key={i} {...appear(i)}>{renderNode(c)}</motion.div>)}</div>
    </Block>
  );
}

function StatsR({ props }: any) {
  const items = (Array.isArray(props.items) ? props.items : []).filter((s: any) => s && s.label != null).slice(0, 4);
  if (!items.length) return null;
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      {items.map((s: any, i: number) => (
        <motion.div key={i} {...appear(i)} className="rounded-xl border bg-card px-3 py-2.5 shadow-card">
          <div className="es-num text-xl font-bold leading-6 text-foreground">{String(s.value ?? '–')}</div>
          <div className="text-[11px] font-medium leading-tight text-muted-foreground">{String(s.label)}</div>
        </motion.div>
      ))}
    </div>
  );
}

function BarChartR({ props }: any) {
  const data = (Array.isArray(props.data) ? props.data : []).filter((d: any) => d && d.label != null && Number.isFinite(Number(d.value))).slice(0, 8);
  if (!data.length) return null;
  const max = Math.max(...data.map((d: any) => Number(d.value)), 0) || 1;
  return (
    <Block title={props.title} icon="columns">
      <div className="space-y-1.5 rounded-xl border bg-card p-3 shadow-card">
        {data.map((d: any, i: number) => (
          <div key={i} className="grid grid-cols-[minmax(0,9rem)_1fr_auto] items-center gap-2.5 text-xs">
            <span className="truncate text-body" title={String(d.label)}>{String(d.label)}</span>
            <span className="h-2 overflow-hidden rounded-full bg-secondary">
              <motion.span className="block h-full rounded-full bg-brand-500" initial={{ width: 0 }} animate={{ width: `${Math.max(4, (Number(d.value) / max) * 100)}%` }}
                transition={{ duration: 0.6, ease: MOTION.ease, delay: 0.1 + i * 0.05 }} />
            </span>
            <span className="es-num text-right font-semibold tabular-nums text-foreground">{Number(d.value).toLocaleString()}</span>
          </div>
        ))}
      </div>
    </Block>
  );
}

function StepsR({ props }: any) {
  const steps = (Array.isArray(props.steps) ? props.steps : []).filter((s: any) => typeof s === 'string' && s.trim());
  if (!steps.length) return null;
  return (
    <Block title={props.title} icon="list">
      <ol className="space-y-2">
        {steps.map((s: string, i: number) => (
          <motion.li key={i} {...appear(i)} className="flex items-start gap-2.5 text-sm text-body">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-brand-600 text-[11px] font-bold text-white">{i + 1}</span>
            <span className="pt-0.5 leading-snug">{inline(s)}</span>
          </motion.li>
        ))}
      </ol>
    </Block>
  );
}

function CalloutR({ props }: any) {
  if (!props.text) return null;
  const tone = props.tone === 'warning' ? 'warning' : props.tone === 'tip' ? 'tip' : 'insight';
  const style = tone === 'warning' ? 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-100'
    : tone === 'tip' ? 'border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-100'
      : 'border-brand-200 bg-brand-50 text-brand-900 dark:border-brand-300 dark:bg-brand-50 dark:text-brand-900';
  return (
    <div className={cx('flex items-start gap-2.5 rounded-xl border px-3 py-2.5 text-sm leading-snug', style)}>
      <Icon name={tone === 'warning' ? 'alert' : tone === 'tip' ? 'check' : 'sparkles'} size={15} className="mt-0.5 shrink-0" />
      <span>{inline(String(props.text))}</span>
    </div>
  );
}

function DraftR({ props }: any) {
  const { channel = 'email', to, subject, body = '' } = props;
  const [copied, setCopied] = useState(false);
  if (!body) return null;
  const label = channel === 'linkedin' ? 'LinkedIn message' : channel === 'text' ? 'Text message' : 'Email';
  const copy = () => navigator.clipboard.writeText(subject ? `Subject: ${subject}\n\n${body}` : body).then(() => {
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }, () => {});
  return (
    <div className="overflow-hidden rounded-xl border bg-card shadow-card">
      <div className="flex items-center gap-2 border-b bg-secondary/50 px-3 py-2">
        <Icon name={channel === 'linkedin' ? 'link' : channel === 'text' ? 'smartphone' : 'mail'} size={14} className="text-brand-700" />
        <span className="flex-1 truncate text-xs font-semibold text-foreground">{label}{to ? ` · to ${to}` : ''}</span>
        <button type="button" onClick={copy} className="inline-flex cursor-pointer items-center gap-1 rounded-md px-2 py-1 text-xs font-semibold text-brand-700 hover:bg-brand-50">
          <Icon name={copied ? 'check' : 'copy'} size={12} />{copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      {subject ? <div className="border-b px-3 py-2 text-sm font-semibold text-foreground">{subject}</div> : null}
      <p className="whitespace-pre-wrap px-3 py-2.5 text-sm leading-relaxed text-body">{body}</p>
    </div>
  );
}

function ExhibitorR({ props }: any) {
  const { name, booth, hall, reason } = props;
  if (!name) return null;
  const where = [hall && (/^hall\b/i.test(hall) ? hall : `Hall ${hall}`), booth && `Booth ${booth}`].filter(Boolean).join(' · ');
  return (
    <div className="flex items-start gap-3 rounded-xl border bg-card px-3 py-2.5 shadow-card">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-sky-50 text-sky-700 dark:bg-sky-500/15 dark:text-sky-300"><Icon name="mapPin" size={15} /></span>
      <span className="min-w-0">
        <span className="block text-sm font-semibold text-foreground">{name}</span>
        {where ? <span className="block text-xs text-muted-foreground">{where}</span> : null}
        {reason ? <span className="mt-0.5 block text-xs text-body">{reason}</span> : null}
      </span>
    </div>
  );
}

function CardR({ props, renderNode }: any) {
  const children = Array.isArray(props.children) ? props.children : [];
  return <div className="space-y-3">{children.map((c: any, i: number) => <motion.div key={i} {...appear(i)}>{renderNode(c)}</motion.div>)}</div>;
}

export const assistantLibrary = createLibrary({
  components: defineAskComponents(defineComponent as any, {
    Text: TextR, ContactCard: ContactCardR, ContactList: ContactListR, Stats: StatsR, BarChart: BarChartR,
    Steps: StepsR, Callout: CalloutR, Draft: DraftR, Exhibitor: ExhibitorR, Card: CardR,
  }),
  root: OPENUI_ROOT,
});

/* ------------------------------------------------------------------ an answer: text + OpenUI blocks */

type Segment = { kind: 'text'; text: string } | { kind: 'ui'; code: string };
// ```openui-lang … ``` (also ```openui, and a missing closing fence at the end while streaming)
const FENCE = /```(?:openui-lang|openui)[^\n]*\n([\s\S]*?)(?:```|$)/g;

/** Splits an answer into plain-text and OpenUI parts, in order (same rules as the mobile app). */
export function splitAnswer(answer: string, streaming = false): Segment[] {
  let text = (answer || '').replace(/\r\n/g, '\n');
  if (streaming) text = text.replace(/(^|\n)`{1,3}[\w-]*$/, '$1'); // a half-written opening fence
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(FENCE)) {
    const before = text.slice(last, m.index).trim();
    if (before) out.push({ kind: 'text', text: before });
    if (m[1]!.trim()) out.push({ kind: 'ui', code: m[1]!.trim() });
    last = m.index! + m[0].length;
  }
  const rest = text.slice(last).trim();
  if (rest) out.push({ kind: 'text', text: rest });
  if (out.length === 1 && out[0]!.kind === 'text' && /^root\s*=\s*\w+\(/m.test(out[0]!.text)) return [{ kind: 'ui', code: out[0]!.text }];
  return out;
}

class Boundary extends React.Component<{ children: React.ReactNode; fallback: React.ReactNode; resetKey: string }, { failed: boolean }> {
  override state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  // a half-written block can trip a component: try again when more of it arrives
  override componentDidUpdate(prev: { resetKey: string }) { if (this.state.failed && prev.resetKey !== this.props.resetKey) this.setState({ failed: false }); }
  override render() { return this.state.failed ? this.props.fallback : this.props.children; }
}

function UiBlock({ code, streaming }: { code: string; streaming: boolean }) {
  const [failed, setFailed] = useState(false);
  const hadRoot = useRef(false);
  // while streaming the program is incomplete; judge it once it's all here
  useEffect(() => { if (!streaming && !hadRoot.current) setFailed(true); }, [streaming]);
  const fallback = <p className="text-xs italic text-muted-foreground">Part of this answer couldn't be displayed.</p>;
  if (failed) return fallback;
  return (
    <Boundary resetKey={streaming ? code : 'final'} fallback={streaming ? null : fallback}>
      <Renderer response={code} library={assistantLibrary} isStreaming={streaming} publishObservability={false}
        onParseResult={(r: any) => { hadRoot.current = !!r?.root; if (!streaming && !r?.root) setFailed(true); }} />
    </Boundary>
  );
}

/** An answer: Markdown text plus OpenUI components. Contact cards open the contacts it drew on. */
export function RichAnswer({ text, streaming = false, contacts, onContact }: { text: string; streaming?: boolean } & AnswerLinks) {
  const segments = useMemo(() => splitAnswer(text, streaming), [text, streaming]);
  const links = useMemo(() => ({ contacts, onContact }), [contacts, onContact]);
  return (
    <LinksContext.Provider value={links}>
      <div className="space-y-3">
        {segments.map((s, i) => (s.kind === 'text' ? <Markdown key={i} text={s.text} /> : <UiBlock key={i} code={s.code} streaming={streaming} />))}
      </div>
    </LinksContext.Provider>
  );
}

/** The answer without its UI blocks (for Copy). */
export function answerText(text: string): string {
  return splitAnswer(text).map((s) => (s.kind === 'text' ? s.text : '')).filter(Boolean).join('\n\n');
}
