// Voice questions in the browser, like the app's Ask AI mic (features/ask/useVoiceRecorder in the app):
// MediaRecorder records the question (webm/opus where supported), an AnalyserNode gives a live level for
// the waveform, and stop() hands back the recording for POST /api/transcribe. Limits match the server:
// about 2 minutes; anything under 0.8 s is treated as a mis-click.
import { useCallback, useEffect, useRef, useState } from 'react';

export type VoiceStatus = 'idle' | 'recording' | 'denied' | 'error';
export const MAX_SECONDS = 120;
const MIN_SECONDS = 0.8;

export interface Recording { blob: Blob; name: string }

const pickType = () => {
  if (typeof MediaRecorder === 'undefined') return '';
  return ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'].find((t) => MediaRecorder.isTypeSupported?.(t)) ?? '';
};
const extOf = (type: string) => (type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : 'webm');

export function useVoiceRecorder() {
  const [status, setStatus] = useState<VoiceStatus>('idle');
  const [error, setError] = useState('');
  const [seconds, setSeconds] = useState(0);
  const [level, setLevel] = useState(0);
  const [autoRecording, setAutoRecording] = useState<Recording | null>(null);
  const rec = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const ctx = useRef<AudioContext | null>(null);
  const chunks = useRef<Blob[]>([]);
  const timer = useRef<number | null>(null);
  const startedAt = useRef(0);
  const settle = useRef<((r: Recording | null) => void) | null>(null);
  const keep = useRef(true);
  const starting = useRef(false); // a second click while the permission prompt is open must not open a second mic
  const peak = useRef(0);         // loudest level heard in this recording (to catch a muted / wrong mic)

  const teardown = useCallback(() => {
    if (timer.current) { window.clearInterval(timer.current); timer.current = null; }
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    void ctx.current?.close().catch(() => {});
    ctx.current = null;
    rec.current = null;
    setLevel(0);
  }, []);
  useEffect(() => () => { keep.current = false; rec.current?.state === 'recording' && rec.current.stop(); teardown(); }, [teardown]);

  const start = useCallback(async () => {
    if (rec.current || starting.current) return false;
    setError('');
    if (!window.isSecureContext) {
      // browsers only offer the microphone on https:// or http://localhost
      setStatus('error');
      setError(`The microphone only works on a secure page. Open the console at https://… or http://localhost:${location.port || '3000'} instead of ${location.host}.`);
      return false;
    }
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setStatus('error');
      setError("Voice input isn't supported in this browser. Use Chrome, Edge or Firefox, or type your question.");
      return false;
    }
    starting.current = true;
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      stream.current = s;
      const type = pickType();
      const r = new MediaRecorder(s, type ? { mimeType: type } : undefined);
      chunks.current = [];
      r.ondataavailable = (e) => { if (e.data.size) chunks.current.push(e.data); };
      r.onstop = () => {
        const duration = (performance.now() - startedAt.current) / 1000;
        const mime = r.mimeType || type || 'audio/webm';
        const blob = new Blob(chunks.current, { type: mime });
        // only trust "silent" when the level meter was really running
        const silent = ctx.current?.state === 'running' && peak.current < 0.02;
        teardown();
        setStatus('idle');
        setSeconds(0);
        let out: Recording | null = null;
        if (keep.current) {
          if (duration < MIN_SECONDS || blob.size < 1000) setError('That was too short — click the mic, speak your question, then ■.');
          else if (silent) setError("The microphone didn't pick up any sound. Check it isn't muted and that the right one is selected (lock icon in the address bar → Microphone, and Windows Settings → Sound → Input).");
          else out = { blob, name: `question.${extOf(mime)}` };
        }
        settle.current?.(out);
        settle.current = null;
      };
      // live level (RMS of the waveform) for the bars
      try {
        const ac = new AudioContext();
        ctx.current = ac;
        // created after the permission prompt, so Chrome may start it suspended (the bars would stay flat)
        if (ac.state === 'suspended') void ac.resume().catch(() => {});
        const an = ac.createAnalyser();
        an.fftSize = 512;
        ac.createMediaStreamSource(s).connect(an);
        const data = new Uint8Array(an.fftSize);
        startedAt.current = performance.now();
        timer.current = window.setInterval(() => {
          an.getByteTimeDomainData(data);
          let sum = 0;
          for (const v of data) sum += ((v - 128) / 128) ** 2;
          const lv = Math.min(1, Math.sqrt(sum / data.length) * 4);
          if (lv > peak.current) peak.current = lv;
          setLevel(lv);
          setSeconds((performance.now() - startedAt.current) / 1000);
        }, 80);
      } catch {
        startedAt.current = performance.now();
        timer.current = window.setInterval(() => setSeconds((performance.now() - startedAt.current) / 1000), 200);
      }
      peak.current = 0;
      rec.current = r;
      r.start(250);
      setSeconds(0);
      setStatus('recording');
      return true;
    } catch (e) {
      teardown();
      const name = (e as Error)?.name;
      const msg = (e as Error)?.message || '';
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        setStatus('denied');
        setError(/system/i.test(msg)
          ? 'Windows is blocking the microphone for the browser. Turn on Settings → Privacy & security → Microphone → "Let desktop apps access your microphone", then try again.'
          : 'Microphone access is blocked for this site. Click the lock icon in the address bar → Microphone → Allow, then try again.');
      } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
        setStatus('error');
        setError('No microphone was found. Plug one in (or pick it in Windows Settings → Sound → Input), or type your question.');
      } else if (name === 'NotReadableError' || name === 'AbortError') {
        setStatus('error');
        setError('The microphone is busy or unavailable — close other apps using it (Teams, Zoom, another tab) and try again.');
      } else {
        setStatus('error');
        setError(`Couldn't start the microphone${msg ? ` (${msg})` : ''}. Please try again.`);
      }
      return false;
    } finally {
      starting.current = false;
    }
  }, [teardown]);

  const finish = useCallback((save: boolean) => new Promise<Recording | null>((resolve) => {
    const r = rec.current;
    if (!r || r.state !== 'recording') { resolve(null); return; }
    keep.current = save;
    settle.current = (out) => { keep.current = true; resolve(out); };
    r.stop();
  }), []);
  const stop = useCallback(() => finish(true), [finish]);
  const cancel = useCallback(() => finish(false), [finish]);

  // at the limit it sends itself instead of cutting the admin off silently
  useEffect(() => {
    if (status === 'recording' && seconds >= MAX_SECONDS) void stop().then((r) => r && setAutoRecording(r));
  }, [seconds, status, stop]);

  return {
    status, isRecording: status === 'recording', seconds, level, maxSeconds: MAX_SECONDS,
    error, clearError: () => setError(''),
    start, stop, cancel,
    /** set when the recording hit the limit and stopped itself; take it, then clear */
    autoRecording, clearAutoRecording: () => setAutoRecording(null),
  };
}
