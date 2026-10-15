// /assistant — Ask AI for the whole workspace, laid out like the app's Ask AI screen. All of it is the real API:
//   answer   POST /api/converse/stream        (SSE; uiFormat "openui": rich answers rendered with OpenUI; the
//            contacts it drew on come with it, so contact cards open the real contact; follow-ups = its quick_replies)
//   files    POST /api/converse/stream/files  (images / PDF / text-CSV, 3 × 10 MB — same rules as the app)
//   voice    POST /api/transcribe             (the browser's recording → text, then it's asked)
//   models   GET  /api/models · context GET /api/admin/dashboard · GET /api/admin/events (real counts)
// Conversations are kept in this browser (localStorage, per admin); files aren't stored, only their name and a preview.
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from '@tanstack/react-router';
import { AnimatePresence, motion } from 'motion/react';
import { api } from '../lib/api';
import { q as queries } from '../lib/queries';
import { getSession } from '../lib/auth';
import type { ConverseContact } from '../lib/types';
import { Icon, type IconName } from '../ui/icons';
import { MOTION } from '../ui/tokens';
import { AnimatedNumber } from '../ui/micro';
import { Badge, Card, Skeleton, cx, errorMessage, relTime, useUi } from '../ui/kit';
import { RichAnswer, answerText } from '../features/assistant/openui';
import { Composer } from '../features/assistant/Composer';
import { useVoiceRecorder, type Recording } from '../features/assistant/useVoiceRecorder';
import { addFiles, formatBytes, isImage, metaOf, type AttachmentMeta } from '../features/assistant/attachments';

interface Msg {
  /** stable id: the answer being written and the saved answer are the same message (no re-render flash) */
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  at: string;
  model?: string | undefined;
  error?: boolean | undefined;
  /** the contacts this answer drew on */
  contacts?: ConverseContact[] | undefined;
  /** follow-up questions the server suggested */
  quick?: string[] | undefined;
  /** stopped by the admin before it finished */
  stopped?: boolean | undefined;
  /** still being written (only while streaming; never saved) */
  pending?: boolean | undefined;
  /** asked by voice */
  via?: 'voice' | 'text' | undefined;
  /** files attached to the question (name, size, small preview) */
  files?: AttachmentMeta[] | undefined;
}
interface Chat { id: string; title: string; updatedAt: string; messages: Msg[] }
interface Live { chatId: string; id: string; text: string; contacts: ConverseContact[]; model: string; withFiles: boolean }
type History = { role: string; content: string }[];

const uid = () => getSession()?.userId || 'me';
const STORE = () => `es_admin_chats_${uid()}`;
const LEGACY = () => `es_admin_chat_${uid()}`; // the single chat the old page kept
const PANEL_KEY = 'es_admin_chats_panel';
const spring = { type: 'spring', stiffness: 420, damping: 28 } as const;
const newId = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`);
const titleOf = (q: string) => (q.length > 60 ? `${q.slice(0, 57)}…` : q);
const historyOf = (messages: Msg[]): History => messages.filter((m) => !m.error && !m.stopped && !m.pending).slice(-12).map((m) => ({ role: m.role, content: m.content }));
const questions = (c: Chat) => c.messages.filter((m) => m.role === 'user').length;
const TEMP: Record<string, string> = {
  hot: 'bg-rose-100 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300',
  warm: 'bg-amber-100 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300',
  // temperatures have four levels (score_levels.py); cool keeps the colour 50–69 always had here
  cool: 'bg-sky-100 text-sky-700 dark:bg-sky-500/15 dark:text-sky-300',
  cold: 'bg-sky-100 text-sky-700 dark:bg-sky-500/15 dark:text-sky-300',
};

/** Saved conversations (and the old single chat, carried over once with a fixed id). */
function loadChats(): Chat[] {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE()) || '[]');
    if (Array.isArray(saved) && saved.length) return saved;
    const legacy = JSON.parse(localStorage.getItem(LEGACY()) || '[]');
    if (Array.isArray(legacy) && legacy.length) {
      const first = legacy.find((m: Msg) => m.role === 'user');
      return [{ id: `legacy-${uid()}`, title: titleOf(first?.content || 'Earlier chat'), updatedAt: legacy[legacy.length - 1]?.at || new Date().toISOString(), messages: legacy }];
    }
  } catch { /* blocked / corrupt storage: start fresh */ }
  return [];
}

/** Prompts to start from — one built around the workspace's real most recently active event. */
function useSuggestions(): { icon: IconName; tint: string; text: string }[] {
  const events = useQuery(queries.adminEvents());
  const latest = events.data?.find((e) => e.contacts > 0)?.name;
  return [
    { icon: 'flame', tint: 'bg-rose-50 text-rose-600 dark:bg-rose-500/15 dark:text-rose-300', text: 'Who are our hottest leads right now, and why?' },
    { icon: 'users', tint: 'bg-amber-50 text-amber-600 dark:bg-amber-500/15 dark:text-amber-300', text: 'Which reps added the most contacts this week, and how good were their leads?' },
    { icon: 'calendar', tint: 'bg-sky-50 text-sky-600 dark:bg-sky-500/15 dark:text-sky-300',
      text: latest ? `Summarise the leads from ${latest} and who should follow up first.` : 'Summarise the leads from our latest event and who should follow up first.' },
    { icon: 'building', tint: 'bg-emerald-50 text-emerald-600 dark:bg-emerald-500/15 dark:text-emerald-300', text: 'Which companies did we meet more than once, and who met them?' },
    { icon: 'mail', tint: 'bg-brand-50 text-brand-700 dark:text-brand-800', text: 'Draft a follow-up email for our highest-scoring lead.' },
    { icon: 'note', tint: 'bg-slate-100 text-slate-600 dark:bg-white/5 dark:text-slate-300', text: 'Which hot leads have no follow-up notes yet?' },
  ];
}

export function AssistantComponent() {
  const navigate = useNavigate();
  const { toast, confirm } = useUi();
  // one load: the list and the selected chat come from the same data
  const [chats, setChats] = useState<Chat[]>(loadChats);
  const [activeId, setActiveId] = useState<string | null>(() => chats[0]?.id ?? null);
  const active = chats.find((c) => c.id === activeId) ?? null;
  const [draft, setDraft] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const [transcribing, setTranscribing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  const [convOpen, setConvOpenState] = useState<boolean>(() => { try { return localStorage.getItem(PANEL_KEY) !== 'closed'; } catch { return true; } });
  const setConvOpen = (open: boolean) => {
    setConvOpenState(open);
    try { localStorage.setItem(PANEL_KEY, open ? 'open' : 'closed'); } catch { /* not remembered */ }
  };
  const [live, setLive] = useState<Live | null>(null);
  const sending = useRef(false);
  const abort = useRef<AbortController | null>(null);
  // the files of questions asked in this visit, so "Ask again" can send them again (files aren't stored)
  const sentFiles = useRef(new Map<string, File[]>());
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true); // follow the answer only while the admin is at the bottom
  const [awayFromBottom, setAwayFromBottom] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);
  const voice = useVoiceRecorder();
  const models = useQuery(queries.models());
  const dash = useQuery(queries.dashboard(7));
  const events = useQuery(queries.adminEvents());
  const suggestions = useSuggestions();
  const answering = !!live;
  const busy = answering || transcribing || voice.isRecording;
  const aiReady = !models.data || models.data.some((m) => m.configured);
  const chatModel = models.data?.find((m) => m.default);
  const audioModel = models.data?.find((m) => m.role === 'audio');
  const mode: PillMode = !aiReady ? 'offline' : voice.isRecording ? 'listening' : transcribing ? 'transcribing' : answering ? 'thinking' : 'ready';

  // the messages on screen: the saved ones, plus the answer being written (same id it's saved under)
  const items: Msg[] = useMemo(() => {
    const saved = active?.messages ?? [];
    return live && live.chatId === active?.id
      ? [...saved, { id: live.id, role: 'assistant', content: live.text, contacts: live.contacts, model: live.model, at: new Date().toISOString(), pending: true }]
      : saved;
  }, [active, live]);
  const last = items[items.length - 1];

  useEffect(() => {
    try { localStorage.setItem(STORE(), JSON.stringify(chats.slice(0, 30).map((c) => ({ ...c, messages: c.messages.slice(-60) })))); } catch { /* storage full / blocked */ }
  }, [chats]);
  useEffect(() => () => abort.current?.abort(), []);
  useEffect(() => { if (voice.error) setNotice(voice.error); }, [voice.error]);

  const toBottom = useCallback((smooth: boolean) => {
    const el = scroller.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
  }, []);
  // switching chats → its latest message; a new message → smooth; while the answer streams → instantly,
  // and only while the admin hasn't scrolled up to read
  useLayoutEffect(() => { stick.current = true; setAwayFromBottom(false); toBottom(false); }, [activeId, toBottom]);
  useLayoutEffect(() => { if (stick.current) toBottom(false); }, [live?.text, toBottom]);
  useEffect(() => { if (stick.current) toBottom(true); }, [items.length, toBottom]);
  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    const away = el.scrollHeight - el.scrollTop - el.clientHeight > 120;
    stick.current = !away;
    setAwayFromBottom(away);
  };

  const update = (id: string, fn: (c: Chat) => Chat) => setChats((all) => all.map((c) => (c.id === id ? fn(c) : c)).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)));
  const attach = (incoming: File[]) => {
    if (!incoming.length) return;
    const { files: next, problem } = addFiles(files, incoming);
    setFiles(next);
    setNotice(problem || null);
  };

  /** Ask `text` (with `withFiles`) in chat `chatId` (a new chat when null). `history` = what came before (retry passes its own). */
  const send = async (text: string, opts: { chatId?: string | null; history?: History; withFiles?: File[]; via?: 'voice' | 'text' | undefined } = {}) => {
    const q = text.trim();
    const sendFiles = opts.withFiles ?? [];
    if ((!q && !sendFiles.length) || sending.current) return;
    sending.current = true;
    const metas = sendFiles.length ? await metaOf(sendFiles) : undefined;
    let id = opts.chatId === undefined ? activeId : opts.chatId;
    const existing = id ? chats.find((c) => c.id === id) : undefined;
    const prior = opts.history ?? historyOf(existing?.messages ?? []);
    const now = new Date().toISOString();
    const question = q || `About ${sendFiles.length === 1 ? sendFiles[0]!.name : `${sendFiles.length} files`}`;
    if (!existing) {
      id = newId();
      setChats((all) => [{ id: id!, title: titleOf(question), updatedAt: now, messages: [] }, ...all]);
      setActiveId(id);
    }
    const cid = id!;
    const qid = newId();
    if (sendFiles.length) sentFiles.current.set(qid, sendFiles);
    update(cid, (c) => ({ ...c, updatedAt: now, messages: [...c.messages, { id: qid, role: 'user', content: question, at: now, via: opts.via, files: metas }] }));
    setDraft('');
    setFiles([]);
    setNotice(null);
    stick.current = true;
    setAwayFromBottom(false);
    const ctrl = new AbortController();
    abort.current = ctrl;
    const aid = newId();
    let cur: Live = { chatId: cid, id: aid, text: '', contacts: [], model: '', withFiles: sendFiles.length > 0 };
    setLive(cur);
    const save = (m: Omit<Msg, 'id' | 'at'>) => update(cid, (c) => ({ ...c, updatedAt: new Date().toISOString(), messages: [...c.messages, { ...m, id: aid, at: new Date().toISOString() }] }));
    try {
      const r = await api.converseStream({ query: q, conversationHistory: prior, uiFormat: 'openui', files: sendFiles }, {
        onMeta: (m) => { cur = { ...cur, contacts: m.contacts }; setLive(cur); },
        onModel: (model) => { cur = { ...cur, model }; setLive(cur); },
        onDelta: (t) => { cur = { ...cur, text: t }; setLive(cur); },
      }, ctrl.signal);
      save({ role: 'assistant', content: r.reply, model: r.modelUsed, contacts: r.contacts, quick: r.quickReplies });
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') {
        save({ role: 'assistant', content: cur.text || 'Stopped before the answer started.', model: cur.model, contacts: cur.contacts, stopped: true });
      } else {
        save({ role: 'assistant', content: errorMessage(e, "The assistant couldn't answer."), error: true });
      }
    } finally {
      abort.current = null;
      sending.current = false;
      setLive(null);
      input.current?.focus();
    }
  };

  /** Ask the last question again: drop it and its answer, then send it with the history (and files) from before it. */
  const retryLast = () => {
    if (!active || busy) return;
    const cut = active.messages.map((m) => m.role).lastIndexOf('user');
    if (cut < 0) return;
    const asked = active.messages[cut]!;
    const kept = asked.id ? sentFiles.current.get(asked.id) : undefined;
    if (asked.files?.length && !kept) {
      // the files aren't stored: put the question back and ask for them again
      setDraft(asked.content);
      setNotice(`Attach ${asked.files.map((f) => f.name).join(', ')} again to ask this — files aren't kept after the page reloads.`);
      input.current?.focus();
      return;
    }
    const before = active.messages.slice(0, cut);
    update(active.id, (c) => ({ ...c, messages: before }));
    void send(asked.content, { chatId: active.id, history: historyOf(before), withFiles: kept ?? [], via: asked.via });
  };

  // voice: record → transcribe → ask (like the app)
  const askRecording = async (rec: Recording | null) => {
    if (!rec) return;
    setTranscribing(true);
    setNotice(null);
    let transcript = '';
    try {
      transcript = await api.transcribe(rec.blob, rec.name);
    } catch (e) {
      setNotice(errorMessage(e, "Couldn't transcribe that. Please try again, or type your question."));
    } finally {
      setTranscribing(false);
    }
    if (transcript) await send(transcript, { withFiles: files, via: 'voice' });
  };
  useEffect(() => {
    if (!voice.autoRecording) return;
    const rec = voice.autoRecording;
    voice.clearAutoRecording();
    void askRecording(rec);
  }, [voice.autoRecording]); // eslint-disable-line react-hooks/exhaustive-deps

  const newChat = () => { if (busy) return; setActiveId(null); setDraft(''); setFiles([]); setNotice(null); input.current?.focus(); };
  const removeChat = async (c: Chat) => {
    if (!(await confirm({ title: 'Delete this conversation?', message: `“${c.title}” is removed from this browser.`, confirmLabel: 'Delete', danger: true }))) return;
    setChats((all) => all.filter((x) => x.id !== c.id));
    if (activeId === c.id) setActiveId(null);
  };
  const copy = (text: string) => navigator.clipboard.writeText(text).then(() => true, () => { toast('error', "Couldn't copy"); return false; });
  const openContact = (id: string) => navigate({ to: '/contacts/$id', params: { id }, search: {} as never });

  // drop files anywhere on the chat
  const onDragEnter = (e: React.DragEvent) => { if (!e.dataTransfer.types.includes('Files')) return; e.preventDefault(); dragDepth.current += 1; setDragging(true); };
  const onDragLeave = () => { dragDepth.current = Math.max(0, dragDepth.current - 1); if (!dragDepth.current) setDragging(false); };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    if (!busy) attach(Array.from(e.dataTransfer.files));
  };

  // "Knows N contacts from R reps · E events" — real counts, like the app's context line
  const knowsLine = dash.data
    ? `Knows ${dash.data.totalContacts.toLocaleString()} contact${dash.data.totalContacts === 1 ? '' : 's'} from ${dash.data.reps} rep${dash.data.reps === 1 ? '' : 's'}${events.data ? ` · ${events.data.length} event${events.data.length === 1 ? '' : 's'}` : ''}`
    : 'Reads every rep’s contacts, scores and events';
  const context: { icon: IconName; label: string; value: number | undefined; color: string }[] = [
    { icon: 'contacts', label: 'Contacts', value: dash.data?.totalContacts, color: 'var(--brand-600)' },
    { icon: 'users', label: 'Reps', value: dash.data?.reps, color: '#0ea5e9' },
    { icon: 'flame', label: 'Hot leads', value: dash.data?.levels.hot, color: '#f43f5e' },
    { icon: 'calendar', label: 'Events', value: events.data?.length, color: '#10b981' },
  ];

  return (
    <>
      <h1 className="sr-only">AI assistant</h1>

      <div className={cx('grid gap-3 transition-[grid-template-columns] duration-300 ease-out',
        convOpen ? 'lg:grid-cols-[256px_minmax(0,1fr)] xl:grid-cols-[256px_minmax(0,1fr)_280px]' : 'lg:grid-cols-[60px_minmax(0,1fr)] xl:grid-cols-[60px_minmax(0,1fr)_280px]')}>

        {/* ---------------- conversations: a collapsible section ---------------- */}
        <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease }} className="min-w-0">
          <Card className={cx('flex flex-col overflow-hidden rounded-3xl', convOpen ? 'max-h-72 lg:h-[calc(100dvh-84px)] lg:max-h-none lg:min-h-[520px]' : 'lg:h-[calc(100dvh-84px)] lg:min-h-[520px]')}>
            {convOpen ? (
              <>
                <div className="flex items-center gap-2.5 border-b px-3 py-3">
                  <span className="flex size-8 items-center justify-center rounded-xl bg-brand-50 text-brand-700 dark:text-brand-800"><Icon name="chats" size={15} /></span>
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-extrabold text-foreground">Conversations</div>
                    <div className="text-[11px] text-muted-foreground">Kept in this browser</div>
                  </div>
                  <span className="es-num rounded-full bg-secondary px-2 text-xs font-black leading-6 text-body"><AnimatedNumber value={chats.length} duration={400} /></span>
                  <motion.button type="button" onClick={() => setConvOpen(false)} aria-expanded="true" aria-label="Collapse conversations" title="Collapse"
                    whileHover={{ x: -2 }} whileTap={{ scale: 0.88 }}
                    className="flex size-8 cursor-pointer items-center justify-center rounded-lg text-muted-foreground hover:bg-secondary hover:text-foreground">
                    <Icon name="panelClose" size={16} />
                  </motion.button>
                </div>
                {/* layoutScroll: the animated chat rows don't jump by the scroll offset as an answer streams in */}
                <motion.div layoutScroll className="es-scroll flex-1 space-y-0.5 overflow-y-auto p-2">
                  <motion.button type="button" onClick={newChat} disabled={busy} whileHover={{ y: -1 }} whileTap={{ scale: 0.97 }}
                    className="group mb-1.5 flex w-full cursor-pointer items-center gap-2 rounded-2xl bg-fill px-3 py-2.5 text-left text-[13px] font-bold text-fill-foreground shadow-active transition-colors hover:bg-fill-hover disabled:cursor-not-allowed disabled:opacity-50">
                    <Icon name="plus" size={15} className="transition-transform duration-300 group-hover:rotate-90" />
                    New chat
                  </motion.button>
                  {!chats.length ? (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center gap-2 px-2 py-8 text-center">
                      <motion.span animate={{ y: [0, -4, 0] }} transition={{ duration: 2.4, repeat: Infinity }} className="flex size-10 items-center justify-center rounded-2xl bg-secondary text-muted-foreground"><Icon name="chats" size={18} /></motion.span>
                      <p className="text-xs text-muted-foreground">Your conversations show up here.</p>
                    </motion.div>
                  ) : null}
                  <AnimatePresence initial={false}>
                    {chats.map((c, i) => {
                      const on = c.id === activeId;
                      return (
                        <motion.div key={c.id} layout initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -16, height: 0 }}
                          transition={{ duration: MOTION.fast, delay: Math.min(i, 8) * 0.02 }} className="group relative flex items-center gap-1 rounded-xl pr-1">
                          {on ? <motion.span layoutId="es-chat-active" className="absolute inset-0 rounded-xl bg-accent" transition={spring} /> : null}
                          {on ? <motion.span layoutId="es-chat-bar" className="absolute bottom-2 left-0 top-2 w-[3px] rounded-r bg-brand-600" transition={spring} /> : null}
                          <button type="button" onClick={() => { if (!busy) setActiveId(c.id); }} disabled={busy && !on}
                            className={cx('relative flex min-w-0 flex-1 cursor-pointer items-center gap-2.5 rounded-xl px-2.5 py-2 text-left transition-colors disabled:cursor-not-allowed', !on && 'hover:bg-secondary')}>
                            <span className={cx('flex size-7 shrink-0 items-center justify-center rounded-lg text-[11px] font-black uppercase transition-transform duration-200 group-hover:scale-110',
                              on ? 'bg-brand-600 text-white' : 'bg-secondary text-body')}>{c.title.trim()[0] || '?'}</span>
                            <span className="min-w-0 flex-1">
                              <span className={cx('block truncate text-[13px]', on ? 'font-bold text-accent-foreground' : 'font-semibold text-body')}>{c.title}</span>
                              <span className="flex items-center gap-1 text-[11px] text-muted-foreground">
                                {live?.chatId === c.id ? <><motion.span className="size-1.5 rounded-full bg-brand-500" animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 1, repeat: Infinity }} />Answering…</>
                                  : <>{relTime(c.updatedAt)} · {questions(c)} question{questions(c) === 1 ? '' : 's'}</>}
                              </span>
                            </span>
                          </button>
                          <motion.button type="button" onClick={() => removeChat(c)} aria-label={`Delete ${c.title}`} title="Delete conversation" disabled={busy && on}
                            whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.85 }}
                            className="relative flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-lg text-muted-foreground opacity-0 transition-opacity hover:bg-card hover:text-rose-600 group-hover:opacity-100 focus-visible:opacity-100 disabled:hidden">
                            <Icon name="trash" size={13} />
                          </motion.button>
                        </motion.div>
                      );
                    })}
                  </AnimatePresence>
                </motion.div>
              </>
            ) : (
              <>
                <button type="button" onClick={() => setConvOpen(true)} aria-expanded="false"
                  className="flex cursor-pointer items-center gap-2 px-3 py-2.5 text-left hover:bg-secondary lg:hidden">
                  <Icon name="chats" size={15} className="text-brand-600" />
                  <span className="flex-1 text-sm font-bold text-foreground">Conversations</span>
                  <span className="rounded-full bg-secondary px-2 text-xs font-black leading-6 text-body">{chats.length}</span>
                  <Icon name="chevronDown" size={15} className="text-muted-foreground" />
                </button>
                <div className="hidden flex-1 flex-col items-center gap-1.5 py-3 lg:flex">
                  <motion.button type="button" onClick={() => setConvOpen(true)} aria-label="Show conversations" title="Show conversations" whileHover={{ x: 2 }} whileTap={{ scale: 0.88 }}
                    className="flex size-9 cursor-pointer items-center justify-center rounded-xl text-muted-foreground hover:bg-secondary hover:text-foreground"><Icon name="panelOpen" size={17} /></motion.button>
                  <motion.button type="button" onClick={newChat} disabled={busy} aria-label="New chat" title="New chat" whileHover={{ rotate: 90 }} whileTap={{ scale: 0.88 }}
                    className="flex size-9 cursor-pointer items-center justify-center rounded-xl bg-fill text-fill-foreground shadow-active disabled:opacity-50"><Icon name="plus" size={17} /></motion.button>
                  <div className="my-1 h-px w-7 bg-border" />
                  <div className="no-scrollbar flex flex-col items-center gap-1.5 overflow-y-auto">
                    {chats.slice(0, 14).map((c, i) => (
                      <motion.button key={c.id} type="button" onClick={() => { if (!busy) setActiveId(c.id); }} disabled={busy && c.id !== activeId} title={c.title}
                        initial={{ opacity: 0, scale: 0.7 }} animate={{ opacity: 1, scale: 1 }} transition={{ ...spring, delay: i * 0.03 }} whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}
                        className={cx('flex size-9 shrink-0 cursor-pointer items-center justify-center rounded-xl text-xs font-black uppercase transition-colors disabled:cursor-not-allowed',
                          c.id === activeId ? 'bg-brand-600 text-white shadow-active' : 'bg-secondary text-body hover:bg-brand-50 hover:text-brand-700')}>
                        {c.title.trim()[0] || '?'}
                      </motion.button>
                    ))}
                  </div>
                </div>
              </>
            )}
          </Card>
        </motion.div>

        {/* ---------------- the chat ---------------- */}
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base * 1.3, ease: MOTION.ease, delay: 0.05 }} className="min-w-0">
          <Card className="relative flex h-[calc(100dvh-84px)] min-h-[520px] flex-col overflow-hidden rounded-3xl"
            onDragEnter={onDragEnter} onDragOver={(e) => e.preventDefault()} onDragLeave={onDragLeave} onDrop={onDrop}>
            {/* header, like the app's Ask AI header: icon · title + what it knows · status · new chat */}
            <div className="relative overflow-hidden border-b px-4 py-3"
              style={{ background: 'linear-gradient(120deg, color-mix(in oklab, var(--brand-500) 12%, var(--card)), var(--card) 65%)' }}>
              <motion.span aria-hidden className="pointer-events-none absolute -right-10 -top-16 size-44 rounded-full" style={{ background: 'color-mix(in oklab, var(--brand-500) 12%, transparent)' }}
                animate={{ scale: [1, 1.12, 1], rotate: [0, 10, 0] }} transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }} />
              <div className="relative flex items-center gap-3">
                <motion.span className="relative flex size-11 shrink-0 items-center justify-center rounded-2xl bg-brand-600 text-white shadow-active"
                  initial={{ scale: 0.5, rotate: -20 }}
                  animate={answering || transcribing ? { scale: [1, 1.08, 1], rotate: [0, -10, 10, 0] } : { scale: 1, rotate: [0, -10, 8, -4, 0] }}
                  transition={answering || transcribing ? { duration: 1.4, repeat: Infinity, ease: 'easeInOut' } : { scale: spring, rotate: { duration: 1.2, delay: 0.3, repeat: Infinity, repeatDelay: 6 } }}>
                  {answering ? <motion.span aria-hidden className="absolute inset-0 rounded-2xl bg-brand-500" animate={{ scale: [1, 1.5], opacity: [0.4, 0] }} transition={{ duration: 1.2, repeat: Infinity }} /> : null}
                  <Icon name="sparkles" size={20} className="relative" />
                </motion.span>
                <div className="min-w-0 flex-1">
                  <AnimatePresence mode="wait" initial={false}>
                    <motion.div key={active?.id ?? 'new'} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: MOTION.fast }}
                      className="truncate font-display text-[17px] font-black leading-6 tracking-[-0.02em] text-foreground">{active?.title ?? 'Ask AI'}</motion.div>
                  </AnimatePresence>
                  <div className="flex items-center gap-1.5 truncate text-xs text-body">
                    <Icon name="database" size={11} className="shrink-0 text-brand-600" /><span className="truncate">{knowsLine}</span>
                  </div>
                </div>
                <StatusPill mode={mode} />
                <AnimatePresence>
                  {items.length ? (
                    <motion.button key="new" type="button" onClick={newChat} disabled={busy} aria-label="New chat" title="New chat"
                      initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} transition={spring}
                      whileHover={{ rotate: -8, scale: 1.06 }} whileTap={{ scale: 0.9 }}
                      className="flex size-10 shrink-0 cursor-pointer items-center justify-center rounded-full border bg-card text-foreground shadow-card hover:border-brand-300 disabled:cursor-not-allowed disabled:opacity-50">
                      <Icon name="pencil" size={16} />
                    </motion.button>
                  ) : null}
                </AnimatePresence>
              </div>
            </div>

            <div ref={scroller} onScroll={onScroll} className="es-scroll flex-1 overflow-y-auto px-4 py-5 sm:px-6">
              {!items.length ? (
                <EmptyState aiReady={aiReady} suggestions={suggestions} busy={busy} onAsk={(t) => send(t, { chatId: null })} />
              ) : (
                <div className="mx-auto max-w-3xl space-y-5">
                  {items.map((m, i) => (
                    <Message key={m.id ?? `${m.at}-${i}`} m={m} last={i === items.length - 1} busy={busy} withFiles={live?.withFiles ?? false}
                      onRetry={retryLast} onCopy={() => copy(answerText(m.content) || m.content)} onContact={openContact} />
                  ))}
                  {/* follow-ups the server suggested, under the latest answer */}
                  {!answering && last?.role === 'assistant' && last.quick?.length ? (
                    <div className="flex flex-wrap gap-2 pl-11">
                      {last.quick.map((s, i) => (
                        <motion.button key={s} type="button" onClick={() => send(s)} disabled={busy}
                          initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.15 + i * 0.06 }} whileHover={{ y: -2 }} whileTap={{ scale: 0.96 }}
                          className="group inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-brand-200 bg-brand-50 px-3 py-1.5 text-xs font-semibold text-brand-700 transition-colors hover:border-brand-400 disabled:opacity-50 dark:text-brand-800">
                          <Icon name="sparkles" size={11} className="transition-transform duration-300 group-hover:rotate-12 group-hover:scale-125" />{s}
                        </motion.button>
                      ))}
                    </div>
                  ) : null}
                </div>
              )}
            </div>

            {/* jump back to the latest when the admin scrolled up */}
            <AnimatePresence>
              {awayFromBottom && items.length ? (
                <motion.button key="down" type="button" onClick={() => { stick.current = true; setAwayFromBottom(false); toBottom(true); }}
                  initial={{ opacity: 0, y: 10, scale: 0.9 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 10, scale: 0.9 }} whileHover={{ y: -2 }} whileTap={{ scale: 0.92 }}
                  className="absolute bottom-32 left-1/2 z-10 inline-flex -translate-x-1/2 cursor-pointer items-center gap-1.5 rounded-full border bg-card px-3 py-1.5 text-xs font-semibold text-body shadow-xl">
                  <motion.span className="flex" animate={{ y: [0, 2, 0] }} transition={{ duration: 1.2, repeat: Infinity }}><Icon name="arrowDown" size={13} /></motion.span>
                  {answering ? 'Follow the answer' : 'Latest'}
                </motion.button>
              ) : null}
            </AnimatePresence>

            {/* problems with voice / files, like the app's notice */}
            <AnimatePresence>
              {notice ? (
                <motion.div key="notice" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 6 }} transition={spring}
                  className="mx-3 mb-2 flex items-start gap-2 rounded-2xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200">
                  <motion.span className="mt-px flex" animate={{ rotate: [0, -12, 10, 0] }} transition={{ duration: 0.5 }}><Icon name="alert" size={14} /></motion.span>
                  <span className="flex-1 leading-snug">{notice}</span>
                  <button type="button" onClick={() => { setNotice(null); voice.clearError(); }} aria-label="Dismiss" className="cursor-pointer rounded p-0.5 hover:bg-amber-100 dark:hover:bg-amber-500/20"><Icon name="x" size={13} /></button>
                </motion.div>
              ) : null}
            </AnimatePresence>

            <div className="border-t bg-card px-3 pb-2.5 pt-2.5">
              <Composer value={draft} onChange={setDraft} inputRef={input}
                onSend={() => send(draft, { withFiles: files, via: 'text' })}
                files={files} onAddFiles={attach} onRemoveFile={(i) => setFiles((f) => f.filter((_, j) => j !== i))}
                answering={answering} onStopAnswer={() => abort.current?.abort()} transcribing={transcribing}
                voice={{
                  isRecording: voice.isRecording, seconds: voice.seconds, level: voice.level, maxSeconds: voice.maxSeconds,
                  onStart: () => { setNotice(null); voice.clearError(); void voice.start(); },
                  onStop: () => { void voice.stop().then(askRecording); },
                  onCancel: () => { void voice.cancel(); },
                }} />
            </div>

            {/* drop files on the chat */}
            <AnimatePresence>
              {dragging ? (
                <motion.div key="drop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="pointer-events-none absolute inset-2 z-20 flex flex-col items-center justify-center gap-3 rounded-[20px] border-2 border-dashed border-brand-400 bg-card/90 backdrop-blur-sm">
                  <motion.span animate={{ y: [0, -8, 0], rotate: [0, -8, 8, 0] }} transition={{ duration: 1.2, repeat: Infinity }}
                    className="flex size-14 items-center justify-center rounded-2xl bg-brand-600 text-white shadow-active"><Icon name="paperclip" size={24} /></motion.span>
                  <div className="text-base font-extrabold text-foreground">Drop to attach</div>
                  <div className="text-xs text-muted-foreground">Images, PDFs, CSV or text · up to 3 files, 10 MB each</div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </Card>
        </motion.div>

        {/* ---------------- what it's connected to ---------------- */}
        <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.1 }} className="hidden space-y-3 xl:block">
          <Card className="overflow-hidden rounded-3xl">
            <div className="flex items-center gap-2.5 border-b px-4 py-3">
              <span className="relative flex size-8 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600 dark:bg-emerald-500/15">
                <Icon name="bolt" size={15} />
                {aiReady && !models.isLoading ? <motion.span aria-hidden className="absolute -right-0.5 -top-0.5 size-2.5 rounded-full bg-emerald-500 ring-2 ring-card" animate={{ scale: [1, 1.3, 1] }} transition={{ duration: 1.6, repeat: Infinity }} /> : null}
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-extrabold text-foreground">{models.isLoading ? 'Checking the AI…' : aiReady ? 'AI connected' : 'AI not set up'}</div>
                <div className="truncate text-[11px] text-muted-foreground">{chatModel ? `Answers with ${chatModel.name}` : 'The models this server uses'}</div>
              </div>
            </div>
            <div className="p-3">
              {models.isLoading ? <Skeleton className="h-24 w-full" /> : models.error ? <p className="text-xs text-muted-foreground">{errorMessage(models.error)}</p> : (
                <ul className="space-y-1.5">
                  {(models.data || []).filter((m) => m.role !== 'fallback').map((m, i) => (
                    <motion.li key={`${m.role}-${m.id}`} initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} transition={{ ...spring, delay: 0.1 + i * 0.05 }} whileHover={{ x: 2 }}
                      className="group flex items-center gap-2.5 rounded-2xl border px-2.5 py-2 transition-colors hover:border-brand-200">
                      <span className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-secondary text-muted-foreground transition-colors group-hover:bg-brand-50 group-hover:text-brand-700">
                        <Icon name={m.role === 'audio' ? 'mic' : m.role === 'vision' ? 'camera' : m.default ? 'sparkles' : 'bolt'} size={13} className="transition-transform duration-300 group-hover:-rotate-12" />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-xs font-bold text-foreground" title={m.id}>{m.name}</span>
                        <span className="block truncate text-[10.5px] text-muted-foreground">{m.uses.join(' · ') || m.role}</span>
                      </span>
                      {m.default ? <Badge tone="violet">Chat</Badge> : null}
                    </motion.li>
                  ))}
                </ul>
              )}
              {audioModel ? <p className="mt-2 flex items-center gap-1 text-[11px] text-muted-foreground"><Icon name="mic" size={11} />Voice questions use {audioModel.name}.</p> : null}
            </div>
          </Card>
          <Card className="overflow-hidden rounded-3xl">
            <div className="flex items-center gap-2.5 border-b px-4 py-3">
              <span className="flex size-8 items-center justify-center rounded-xl bg-sky-50 text-sky-600 dark:bg-sky-500/15 dark:text-sky-300"><Icon name="database" size={15} /></span>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-extrabold text-foreground">What it can read</div>
                <div className="text-[11px] text-muted-foreground">Live from your workspace</div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 p-3">
              {context.map((c, i) => (
                <motion.div key={c.label} initial={{ opacity: 0, y: 8, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ ...spring, delay: 0.15 + i * 0.06 }} whileHover={{ y: -2 }}
                  className="group rounded-2xl border px-2.5 py-2 transition-colors hover:border-brand-200">
                  <span className="mb-1 flex size-7 items-center justify-center rounded-lg transition-transform duration-300 group-hover:-rotate-12 group-hover:scale-110"
                    style={{ background: `color-mix(in oklab, ${c.color} 14%, transparent)`, color: c.color }}><Icon name={c.icon} size={13} /></span>
                  <div className="es-num text-lg font-black leading-5 text-foreground">{c.value == null ? '—' : <AnimatedNumber value={c.value} duration={700} />}</div>
                  <div className="text-[11px] font-semibold text-muted-foreground">{c.label}</div>
                </motion.div>
              ))}
            </div>
            <p className="px-4 pb-3 text-[11px] leading-snug text-muted-foreground">Every rep's contacts with their scores and notes, the reps, events and booth directories — plus any file you attach.</p>
          </Card>
        </motion.div>
      </div>
    </>
  );
}

/** First screen: a glowing assistant mark, questions to start from (one uses your latest event) and how to ask. */
function EmptyState({ aiReady, suggestions, busy, onAsk }: { aiReady: boolean; suggestions: { icon: IconName; tint: string; text: string }[]; busy: boolean; onAsk: (t: string) => void }) {
  return (
    <div className="mx-auto flex min-h-full max-w-2xl flex-col items-center justify-center py-2 text-center">
      <div className="relative mb-4">
        <motion.span aria-hidden className="absolute -inset-5 rounded-full bg-brand-500/25 blur-xl"
          animate={{ scale: [1, 1.18, 1], opacity: [0.5, 1, 0.5] }} transition={{ duration: 2.6, repeat: Infinity, ease: 'easeInOut' }} />
        <motion.span initial={{ scale: 0.4, rotate: -30, opacity: 0 }} animate={{ scale: 1, rotate: 0, opacity: 1 }} transition={{ type: 'spring', stiffness: 300, damping: 15 }}
          className="relative flex size-[72px] items-center justify-center rounded-[22px] bg-brand-600 text-white shadow-active">
          <motion.span className="flex" animate={{ rotate: [-8, 8, -8], scale: [1, 1.05, 1] }} transition={{ duration: 2.6, repeat: Infinity, ease: 'easeInOut' }}>
            <Icon name="sparkles" size={32} />
          </motion.span>
        </motion.span>
      </div>
      <motion.h3 initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.12 }} className="font-display text-[22px] font-black tracking-[-0.02em] text-foreground">What do you want to know?</motion.h3>
      <motion.p initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.18 }} className="mt-1 max-w-md text-sm text-body">
        Ask about every rep's contacts, leads and events — by typing, by voice, or with a photo or file.
      </motion.p>
      {!aiReady ? (
        <motion.div initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200">
          The AI isn't set up on the server (<code className="font-mono">OPENROUTER_API_KEY</code>), so questions will fail until it is.
        </motion.div>
      ) : null}
      <div className="mt-5 grid w-full gap-2 sm:grid-cols-2">
        {suggestions.map((s, i) => (
          <motion.button key={s.text} type="button" onClick={() => onAsk(s.text)} disabled={busy}
            initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.24 + i * 0.06 }}
            whileHover={{ y: -3 }} whileTap={{ scale: 0.98 }}
            className="group flex cursor-pointer items-center gap-3 rounded-2xl border bg-card px-3 py-2.5 text-left text-[13px] font-medium text-body shadow-card transition-[border-color,box-shadow] hover:border-brand-300 hover:shadow-active disabled:cursor-not-allowed disabled:opacity-60">
            <span className={cx('flex size-9 shrink-0 items-center justify-center rounded-xl transition-transform duration-300 group-hover:-rotate-12 group-hover:scale-110', s.tint)}>
              <Icon name={s.icon} size={16} />
            </span>
            <span className="flex-1 leading-snug">{s.text}</span>
            <Icon name="chevronRight" size={15} className="shrink-0 text-muted-foreground transition-all duration-200 group-hover:translate-x-1 group-hover:text-brand-600" />
          </motion.button>
        ))}
      </div>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 }} className="mt-4 flex flex-wrap justify-center gap-2">
        {([['mic', 'Click the mic and speak'], ['paperclip', 'Attach slides, badges, PDFs, CSVs'], ['enter', 'Enter to send']] as [IconName, string][]).map(([icon, text]) => (
          <span key={text} className="inline-flex items-center gap-1.5 rounded-full bg-secondary px-2.5 py-1 text-[11px] font-semibold text-body">
            <Icon name={icon} size={12} className="text-brand-600" />{text}
          </span>
        ))}
      </motion.div>
    </div>
  );
}

type PillMode = 'ready' | 'thinking' | 'listening' | 'transcribing' | 'offline';
/** Ready · Thinking · Listening · Transcribing · AI offline — the app's Ask AI pill (blinking dot while busy). */
function StatusPill({ mode }: { mode: PillMode }) {
  const s = {
    ready: { label: 'Ready', cls: 'bg-emerald-50 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300', dot: 'bg-emerald-500' },
    thinking: { label: 'Thinking', cls: 'bg-brand-50 text-brand-700 dark:text-brand-800', dot: 'bg-brand-600' },
    listening: { label: 'Listening', cls: 'bg-rose-50 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300', dot: 'bg-rose-500' },
    transcribing: { label: 'Transcribing', cls: 'bg-brand-50 text-brand-700 dark:text-brand-800', dot: 'bg-brand-600' },
    offline: { label: 'AI offline', cls: 'bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300', dot: 'bg-amber-500' },
  }[mode];
  const blink = mode !== 'ready' && mode !== 'offline';
  return (
    <motion.span layout transition={spring} title={mode === 'offline' ? "The AI isn't set up on the server (OPENROUTER_API_KEY)" : undefined}
      className={cx('hidden shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-bold tracking-[0.02em] sm:inline-flex', s.cls)}>
      <span className="relative flex size-1.5">
        {mode === 'ready' ? <span className={cx('absolute inline-flex size-full animate-ping rounded-full opacity-60', s.dot)} /> : null}
        <motion.span className={cx('relative inline-flex size-1.5 rounded-full', s.dot)}
          animate={blink ? { opacity: [1, 0.25, 1] } : { opacity: 1 }} transition={blink ? { duration: 1, repeat: Infinity } : { duration: 0.2 }} />
      </span>
      <AnimatePresence mode="wait" initial={false}>
        <motion.span key={mode} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.15 }}>{s.label}</motion.span>
      </AnimatePresence>
    </motion.span>
  );
}

/** Three dots bouncing in turn. */
function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1" aria-hidden>
      {[0, 1, 2].map((i) => (
        <motion.span key={i} className="size-1.5 rounded-full bg-brand-500" animate={{ y: [0, -5, 0], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 0.56, repeat: Infinity, delay: i * 0.14, ease: 'easeInOut' }} />
      ))}
    </span>
  );
}

/** Before the first words: what it's doing, in phases (like the app). */
function Thinking({ withFiles, found }: { withFiles: boolean; found: number }) {
  const [phase, setPhase] = useState(0);
  useEffect(() => {
    const a = setTimeout(() => setPhase(1), 2500);
    const b = setTimeout(() => setPhase(2), 9000);
    return () => { clearTimeout(a); clearTimeout(b); };
  }, []);
  const label = [
    withFiles ? 'Reading your files…' : found ? `Reading ${found} matching contact${found === 1 ? '' : 's'}…` : 'Searching the workspace…',
    'Writing an answer…',
    'Still working — longer answers take a moment…',
  ][phase];
  return (
    <div className="inline-flex items-center gap-2.5 rounded-2xl rounded-tl-md border bg-card px-3.5 py-2.5 text-sm text-muted-foreground shadow-card">
      <TypingDots />
      <AnimatePresence mode="wait" initial={false}>
        <motion.span key={label} initial={{ opacity: 0, x: 6 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -6 }} transition={{ duration: 0.18 }}>{label}</motion.span>
      </AnimatePresence>
    </div>
  );
}

/** Copy that turns into a tick for a moment. */
function CopyButton({ onCopy }: { onCopy: () => Promise<boolean> }) {
  const [done, setDone] = useState(false);
  useEffect(() => { if (!done) return; const t = setTimeout(() => setDone(false), 1500); return () => clearTimeout(t); }, [done]);
  return (
    <button type="button" onClick={async () => { if (await onCopy()) setDone(true); }}
      className={cx('inline-flex cursor-pointer items-center gap-1 rounded-md px-1.5 py-0.5 transition-colors', done ? 'text-emerald-700 dark:text-emerald-400' : 'hover:bg-secondary hover:text-foreground')}>
      <AnimatePresence mode="wait" initial={false}>
        <motion.span key={done ? 'ok' : 'copy'} className="flex" initial={{ scale: 0.4, rotate: -90, opacity: 0 }} animate={{ scale: 1, rotate: 0, opacity: 1 }} exit={{ scale: 0.4, rotate: 90, opacity: 0 }}
          transition={{ type: 'spring', stiffness: 500, damping: 24 }}>
          <Icon name={done ? 'check' : 'copy'} size={11} />
        </motion.span>
      </AnimatePresence>
      {done ? 'Copied' : 'Copy'}
    </button>
  );
}

/** A contact the answer drew on: initials, name, company, score (by temperature); opens the contact. */
function ContactChip({ c, index, onOpen }: { c: ConverseContact; index: number; onOpen: () => void }) {
  const initials = (c.name || '?').split(/\s+/).map((p) => p[0]).join('').slice(0, 2).toUpperCase();
  return (
    <motion.button type="button" onClick={onOpen} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.12 + index * 0.07 }}
      whileHover={{ x: 3 }} whileTap={{ scale: 0.98 }}
      className="group flex w-full cursor-pointer items-center gap-2.5 rounded-xl border bg-card px-2.5 py-2 text-left transition-colors hover:border-brand-300">
      <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-brand-50 text-[11px] font-black text-brand-700 transition-transform duration-200 group-hover:scale-110 dark:text-brand-800">{initials}</span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[13px] font-bold text-foreground">{c.name || 'Unnamed contact'}</span>
        <span className="block truncate text-[11px] text-muted-foreground">{c.company || 'No company'}</span>
      </span>
      {c.score != null ? <span className={cx('rounded-full px-2 text-[11px] font-black leading-5', TEMP[(c.temperature || '').toLowerCase()] ?? 'bg-secondary text-body')}>{c.score}</span>
        : <span className="text-[11px] text-muted-foreground">Not scored</span>}
      <Icon name="chevronRight" size={14} className="text-muted-foreground transition-transform duration-200 group-hover:translate-x-0.5 group-hover:text-brand-600" />
    </motion.button>
  );
}

function Message({ m, last, busy, withFiles, onRetry, onCopy, onContact }: {
  m: Msg; last: boolean; busy: boolean; withFiles: boolean; onRetry: () => void; onCopy: () => Promise<boolean>; onContact: (id: string) => void;
}) {
  const me = getSession()?.name || 'Me';
  if (m.role === 'user') {
    return (
      <motion.div initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} transition={spring} className="flex flex-col items-end gap-1.5">
        {m.files?.length ? (
          <div className="flex flex-wrap justify-end gap-1.5 pr-9">
            {m.files.map((f, i) => (
              <motion.span key={`${f.name}-${i}`} initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ ...spring, delay: i * 0.05 }}>
                {f.thumb ? <img src={f.thumb} alt={f.name} title={f.name} className="size-16 rounded-xl border object-cover shadow-card" /> : (
                  <span className="inline-flex max-w-[220px] items-center gap-1.5 rounded-xl border bg-card px-2.5 py-1.5 text-xs shadow-card">
                    <Icon name={isImage(f) ? 'image' : 'file'} size={14} className="shrink-0 text-brand-600" />
                    <span className="truncate font-semibold text-foreground">{f.name}</span>
                    <span className="shrink-0 text-[10px] text-muted-foreground">{formatBytes(f.size)}</span>
                  </span>
                )}
              </motion.span>
            ))}
          </div>
        ) : null}
        <div className="flex max-w-[82%] flex-row-reverse items-end gap-2">
          <span className="flex size-7 shrink-0 items-center justify-center rounded-full border bg-secondary text-[11px] font-bold text-foreground">{me.slice(0, 1).toUpperCase()}</span>
          <div className="rounded-[20px] rounded-br-md bg-fill px-4 py-2.5 text-sm leading-relaxed text-fill-foreground shadow-active">
            {m.via === 'voice' ? <span className="mb-0.5 flex items-center gap-1 text-[10.5px] font-bold uppercase tracking-[0.06em] opacity-75"><Icon name="mic" size={11} />Voice</span> : null}
            <p className="whitespace-pre-wrap break-words">{m.content}</p>
          </div>
        </div>
        <span className="pr-9 text-[10.5px] text-muted-foreground">{relTime(m.at)}</span>
      </motion.div>
    );
  }
  // the rich answer shows contacts as cards itself: don't list them twice
  const showContacts = !m.pending && !m.error && !!m.contacts?.length && !m.content.includes('ContactCard(');
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={spring} className="group flex gap-3">
      <motion.span initial={{ scale: 0.4, rotate: -30 }} animate={m.pending ? { scale: 1, rotate: [0, -12, 12, 0] } : { scale: 1, rotate: 0 }}
        transition={m.pending ? { duration: 1.4, repeat: Infinity } : spring}
        className={cx('relative flex size-8 shrink-0 items-center justify-center rounded-full', m.error ? 'bg-rose-100 text-rose-600 dark:bg-rose-500/15 dark:text-rose-300' : 'bg-brand-600 text-white shadow-active')}>
        {m.pending ? <motion.span aria-hidden className="absolute inset-0 rounded-full bg-brand-500" animate={{ scale: [1, 1.6], opacity: [0.35, 0] }} transition={{ duration: 1.2, repeat: Infinity }} /> : null}
        <Icon name={m.error ? 'alert' : 'sparkles'} size={14} className="relative" />
      </motion.span>
      <div className="min-w-0 flex-1">
        {m.error ? (
          <motion.div animate={{ x: [0, -5, 5, -3, 3, 0] }} transition={{ duration: 0.4 }}
            className="flex items-start gap-2.5 rounded-2xl border border-rose-200 bg-rose-50 px-3.5 py-3 text-sm text-rose-800 dark:border-rose-500/30 dark:bg-rose-500/10 dark:text-rose-200">
            <span className="flex-1">{m.content}</span>
            {last && !busy ? (
              <motion.button type="button" onClick={onRetry} whileTap={{ scale: 0.95 }}
                className="group/r inline-flex shrink-0 cursor-pointer items-center gap-1 rounded-full bg-card px-2.5 py-1 text-xs font-bold text-brand-700 shadow-card dark:text-brand-800">
                <Icon name="refresh" size={12} className="transition-transform duration-500 group-hover/r:-rotate-180" />Try again
              </motion.button>
            ) : null}
          </motion.div>
        ) : m.pending && !m.content ? (
          <Thinking withFiles={withFiles} found={m.contacts?.length ?? 0} />
        ) : (
          <div className="rounded-2xl rounded-tl-md border bg-card px-4 py-3 shadow-card">
            <RichAnswer text={m.content} streaming={!!m.pending} contacts={m.contacts ?? []} onContact={onContact} />
            {showContacts ? (
              <div className="mt-3 border-t pt-3">
                <div className="mb-2 flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-[0.06em] text-brand-700 dark:text-brand-800"><Icon name="users" size={12} />From your contacts</div>
                <div className="grid gap-1.5 sm:grid-cols-2">
                  {m.contacts!.slice(0, 4).map((c, i) => <ContactChip key={c.id} c={c} index={i} onOpen={() => onContact(c.id)} />)}
                </div>
              </div>
            ) : null}
          </div>
        )}
        <div className="mt-1.5 flex min-h-5 items-center gap-2 text-[11px] text-muted-foreground">
          {m.pending ? (m.content ? <span className="inline-flex items-center gap-1.5 text-brand-600"><TypingDots />{m.model ? `Writing with ${m.model}` : 'Writing…'}</span> : null) : (
            <>
              {m.stopped ? <span className="inline-flex items-center gap-1 font-semibold text-amber-600 dark:text-amber-400"><span className="size-2 rounded-[2px] bg-current" />Stopped</span> : null}
              {m.model ? <span className="inline-flex items-center gap-1"><Icon name="bolt" size={10} />{m.model}</span> : null}
              <span>{relTime(m.at)}</span>
              {!m.error ? (
                <span className="ml-auto flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                  <CopyButton onCopy={onCopy} />
                  {last && !busy ? (
                    <button type="button" onClick={onRetry} className="group/r inline-flex cursor-pointer items-center gap-1 rounded-md px-1.5 py-0.5 hover:bg-secondary hover:text-foreground">
                      <Icon name="refresh" size={11} className="transition-transform duration-500 group-hover/r:rotate-180" />Ask again
                    </button>
                  ) : null}
                </span>
              ) : null}
            </>
          )}
        </div>
      </div>
    </motion.div>
  );
}
