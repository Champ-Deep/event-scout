// The assistant's input, laid out like the app's Ask AI composer (features/ask/components/ChatComposer):
//   [📎 attach]  [ question box (border + background glow on focus) ]  [ 🎤 mic  ⇄  ↑ send ]
// with the attached files as chips above it, a "Transcribing…" pill, and a recording mode:
//   [✕ cancel]  ● 0:07  Listening… click ■ to ask   ▁▃▆▇▅▂ live waveform   [■ stop, pulsing]
// While an answer is being written the send button becomes Stop.
import React, { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { Icon } from '../../ui/icons';
import { cx } from '../../ui/kit';
import { ACCEPT, MAX_ATTACHMENTS, formatBytes, isImage } from './attachments';

const BARS = 32;
const spring = { type: 'spring', stiffness: 460, damping: 28 } as const;
const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

export interface ComposerVoice {
  isRecording: boolean;
  seconds: number;
  level: number;
  maxSeconds: number;
  onStart: () => void;
  onStop: () => void;
  onCancel: () => void;
}

/** Live waveform from the mic level: newest bar on the right. */
function Waveform({ level }: { level: number }) {
  const [samples, setSamples] = useState<number[]>(() => Array(BARS).fill(0));
  useEffect(() => { setSamples((prev) => [...prev.slice(1), level]); }, [level]);
  return (
    <div className="flex h-8 items-center gap-[3px]" aria-hidden>
      {samples.map((v, i) => (
        <motion.span key={i} className={cx('w-[3px] rounded-full', v > 0.55 ? 'bg-brand-700' : 'bg-brand-400')}
          animate={{ height: 4 + v * 26 }} transition={{ duration: 0.08 }} style={{ opacity: 0.35 + (i / BARS) * 0.65 }} />
      ))}
    </div>
  );
}

/** An attached file: image preview or file icon, name and size; ✕ removes it. */
function FileChip({ f, onRemove, disabled }: { f: File; onRemove: () => void; disabled: boolean }) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!isImage(f) || /hei[cf]/i.test(f.type)) return;
    const u = URL.createObjectURL(f);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [f]);
  return (
    <motion.div layout initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.6 }} transition={spring}
      className="relative flex items-center gap-2 rounded-xl border border-brand-100 bg-brand-50 py-1 pl-1 pr-2.5 dark:border-brand-200/30">
      {url ? <img src={url} alt="" className="size-8 rounded-lg object-cover" />
        : <span className="flex size-8 items-center justify-center rounded-lg bg-brand-100 text-brand-700"><Icon name={isImage(f) ? 'image' : 'file'} size={15} /></span>}
      <span className="min-w-0 max-w-[140px]">
        <span className="block truncate text-xs font-semibold text-foreground">{f.name}</span>
        <span className="block text-[10px] text-muted-foreground">{formatBytes(f.size)}</span>
      </span>
      {!disabled ? (
        <motion.button type="button" onClick={onRemove} aria-label={`Remove ${f.name}`} whileHover={{ rotate: 90, scale: 1.1 }} whileTap={{ scale: 0.85 }}
          className="absolute -right-1.5 -top-1.5 flex size-[18px] cursor-pointer items-center justify-center rounded-full bg-foreground text-background ring-2 ring-card">
          <Icon name="x" size={10} strokeWidth={3} />
        </motion.button>
      ) : null}
    </motion.div>
  );
}

export function Composer({ value, onChange, onSend, files, onAddFiles, onRemoveFile, answering, onStopAnswer, transcribing, voice, inputRef, disabled }: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  files: File[];
  onAddFiles: (files: File[]) => void;
  onRemoveFile: (i: number) => void;
  answering: boolean;
  onStopAnswer: () => void;
  transcribing: boolean;
  voice: ComposerVoice;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
  disabled?: boolean | undefined;
}) {
  const picker = useRef<HTMLInputElement>(null);
  const [focused, setFocused] = useState(false);
  const busy = answering || transcribing;
  const hasContent = value.trim().length > 0 || files.length > 0;
  const canSend = hasContent && !busy && !disabled;

  // the box grows with the question (up to ~8 lines)
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [value, inputRef]);
  // Esc cancels a recording
  useEffect(() => {
    if (!voice.isRecording) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') voice.onCancel(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [voice]);

  if (voice.isRecording) {
    const left = Math.max(0, voice.maxSeconds - voice.seconds);
    return (
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={spring} className="mx-auto flex w-full max-w-3xl items-center gap-3 py-1">
        <motion.button type="button" onClick={voice.onCancel} aria-label="Cancel recording" title="Cancel (Esc)" whileHover={{ rotate: 90 }} whileTap={{ scale: 0.88 }}
          className="flex size-11 shrink-0 cursor-pointer items-center justify-center rounded-full bg-secondary text-body hover:text-foreground">
          <Icon name="x" size={19} />
        </motion.button>
        <div className="min-w-0 flex-1 rounded-[22px] border-[1.5px] border-rose-200 bg-rose-50/50 px-4 py-1.5 dark:border-rose-500/30 dark:bg-rose-500/5">
          <div className="flex items-center gap-2">
            <motion.span className="size-2 rounded-full bg-rose-500" animate={{ opacity: [1, 0.2, 1] }} transition={{ duration: 1.2, repeat: Infinity }} />
            <span className="text-[13px] font-bold tabular-nums text-foreground">{fmt(voice.seconds)}</span>
            <span className="truncate text-xs text-muted-foreground">{left <= 15 ? `${Math.ceil(left)}s left` : 'Listening… click ■ to ask'}</span>
          </div>
          <Waveform level={voice.level} />
        </div>
        <span className="relative flex size-[52px] shrink-0 items-center justify-center">
          <motion.span aria-hidden className="absolute inset-0 rounded-full bg-rose-500" animate={{ scale: [1, 1.6], opacity: [0.5, 0] }} transition={{ duration: 1.3, repeat: Infinity, ease: 'easeOut' }} />
          <motion.button type="button" onClick={voice.onStop} aria-label="Stop and ask" title="Stop and ask" whileHover={{ scale: 1.06 }} whileTap={{ scale: 0.9 }}
            className="relative flex size-[52px] cursor-pointer items-center justify-center rounded-full bg-rose-500 text-white shadow-lg">
            <span className="size-4 rounded-[3px] bg-white" />
          </motion.button>
        </span>
      </motion.div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-2">
      <AnimatePresence initial={false}>
        {files.length ? (
          <motion.div key="files" layout initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
            className="flex flex-wrap items-center gap-2 pt-1.5">
            <AnimatePresence initial={false}>
              {files.map((f, i) => <FileChip key={`${f.name}-${f.size}`} f={f} disabled={busy} onRemove={() => onRemoveFile(i)} />)}
            </AnimatePresence>
            <span className="text-[11px] tabular-nums text-muted-foreground">{files.length}/{MAX_ATTACHMENTS}</span>
          </motion.div>
        ) : null}
        {transcribing ? (
          <motion.div key="tx" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="inline-flex items-center gap-2 rounded-full bg-brand-50 px-3 py-1.5 text-xs font-semibold text-brand-700 dark:text-brand-800">
            <motion.span className="flex" animate={{ rotate: 360 }} transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }}><Icon name="loader" size={14} /></motion.span>
            Transcribing your question…
          </motion.div>
        ) : null}
      </AnimatePresence>

      <div className="flex items-end gap-2">
        <input ref={picker} type="file" multiple accept={ACCEPT} hidden
          onChange={(e) => { onAddFiles(Array.from(e.target.files ?? [])); e.target.value = ''; }} />
        <motion.button type="button" onClick={() => picker.current?.click()} disabled={busy || files.length >= MAX_ATTACHMENTS || disabled}
          aria-label="Attach a file or photo" title={files.length >= MAX_ATTACHMENTS ? `Up to ${MAX_ATTACHMENTS} files` : 'Attach images, PDFs or CSV / text (or drop / paste them)'}
          whileHover={{ rotate: -18, scale: 1.05 }} whileTap={{ scale: 0.88 }}
          className="flex size-11 shrink-0 cursor-pointer items-center justify-center rounded-full border border-brand-100 bg-brand-50 text-brand-700 transition-opacity disabled:cursor-not-allowed disabled:opacity-40 dark:border-brand-200/30 dark:text-brand-800">
          <Icon name="paperclip" size={19} />
        </motion.button>

        <motion.div className="min-h-11 flex-1 rounded-[22px] border-[1.5px] px-4"
          animate={{
            borderColor: focused ? 'var(--brand-500)' : 'var(--border)',
            backgroundColor: focused ? 'var(--card)' : 'var(--secondary)',
            boxShadow: focused ? '0 0 0 4px color-mix(in oklab, var(--brand-400) 16%, transparent)' : '0 0 0 0px transparent',
          }}
          transition={{ duration: 0.18 }}>
          <textarea ref={inputRef} rows={1} value={value} onChange={(e) => onChange(e.target.value)} maxLength={4000}
            disabled={transcribing || disabled} aria-label="Your question"
            placeholder={transcribing ? 'Transcribing…' : files.length ? 'Ask about the attached file…' : 'Ask about your leads, reps or events…'}
            onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); if (canSend) onSend(); } }}
            onPaste={(e) => { const pasted = Array.from(e.clipboardData.files); if (pasted.length) { e.preventDefault(); onAddFiles(pasted); } }}
            className="block max-h-[180px] w-full resize-none bg-transparent py-[11px] text-[15px] leading-5 text-foreground outline-none placeholder:text-muted-foreground disabled:opacity-60" />
        </motion.div>

        <AnimatePresence mode="popLayout" initial={false}>
          {answering ? (
            <motion.span key="stop" className="relative flex size-11 shrink-0" initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.4, opacity: 0 }} transition={spring}>
              <motion.span aria-hidden className="absolute inset-0 rounded-full bg-foreground" animate={{ scale: [1, 1.45], opacity: [0.25, 0] }} transition={{ duration: 1.4, repeat: Infinity }} />
              <motion.button type="button" onClick={onStopAnswer} aria-label="Stop the answer" title="Stop the answer" whileTap={{ scale: 0.88 }}
                className="relative flex size-11 cursor-pointer items-center justify-center rounded-full bg-foreground text-background">
                <span className="size-3.5 rounded-[3px] bg-background" />
              </motion.button>
            </motion.span>
          ) : hasContent ? (
            <motion.button key="send" type="button" onClick={onSend} disabled={!canSend} aria-label="Send question" title="Send (Enter)"
              initial={{ scale: 0.4, opacity: 0, rotate: -45 }} animate={{ scale: 1, opacity: 1, rotate: 0 }} exit={{ scale: 0.4, opacity: 0 }} transition={spring}
              whileHover={{ y: -2 }} whileTap={{ scale: 0.88 }}
              className="group flex size-11 shrink-0 cursor-pointer items-center justify-center rounded-full bg-fill text-fill-foreground shadow-[0_4px_12px_-2px_color-mix(in_oklab,var(--fill)_55%,transparent)] hover:bg-fill-hover disabled:cursor-not-allowed disabled:opacity-55">
              <Icon name="arrowUp" size={20} className="transition-transform duration-200 group-hover:-translate-y-0.5" />
            </motion.button>
          ) : (
            <motion.button key="mic" type="button" onClick={voice.onStart} disabled={busy || disabled} aria-label="Ask by voice" title="Ask by voice"
              initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.4, opacity: 0 }} transition={spring}
              whileHover={{ scale: 1.06 }} whileTap={{ scale: 0.88 }}
              className="group flex size-11 shrink-0 cursor-pointer items-center justify-center rounded-full bg-brand-100 text-brand-700 disabled:cursor-not-allowed disabled:opacity-55 dark:text-brand-800">
              <Icon name="mic" size={20} className="transition-transform duration-300 group-hover:-rotate-12 group-hover:scale-110" />
            </motion.button>
          )}
        </AnimatePresence>
      </div>
      <p className="text-center text-[11px] text-muted-foreground">
        <kbd className="rounded border bg-secondary px-1 font-mono text-[10px]">Enter</kbd> to send · <kbd className="rounded border bg-secondary px-1 font-mono text-[10px]">Shift</kbd>+<kbd className="rounded border bg-secondary px-1 font-mono text-[10px]">Enter</kbd> new line · mic to speak · drop or paste files · answers come from your workspace data
      </p>
    </div>
  );
}
