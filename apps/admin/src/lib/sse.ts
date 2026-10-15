import { streamUrl } from './auth';
import type { LiveChange } from './changes';

export interface LiveEventMessage {
  /** a live change (lib/changes), or 'connected' when the stream (re)opens. Parsed from the wire, so an
   *  older console can still meet a name it doesn't know — useEventStream falls back for those. */
  type: LiveChange | 'connected';
  timestamp: string;
  data: Record<string, unknown>;
}

export type StreamStatus = 'connecting' | 'open' | 'offline';

/**
 * Subscribe to the admin SSE feed at `path`. EventSource can't send headers, so each connect first
 * gets a single-use ticket (POST /api/admin/live/ticket, via apiFetch, which refreshes the token on a
 * 401) and connects with ?ticket= — the access token never goes in the URL. When the connection
 * drops (or no ticket can be had) it reconnects with backoff, each time with a new ticket.
 */
export function subscribeLiveEventStream(
  path: string,
  onMessage: (msg: LiveEventMessage) => void,
  onStatus?: (status: StreamStatus) => void
): () => void {
  if (typeof window === 'undefined' || !('EventSource' in window)) return () => {};

  let es: EventSource | null = null;
  let retryTimer: ReturnType<typeof setTimeout> | undefined;
  let attempt = 0;
  let closed = false;

  const retry = () => {
    if (closed) return;
    onStatus?.('offline');
    const delay = Math.min(30_000, 1_000 * 2 ** attempt);
    attempt += 1;
    retryTimer = setTimeout(() => { void connect(); }, delay);
  };

  const connect = async () => {
    if (closed) return;
    onStatus?.('connecting');
    let url: string | null;
    try {
      url = await streamUrl(path);
    } catch {
      retry();
      return;
    }
    // signed out, or unsubscribed while the ticket was on its way
    if (!url || closed) return;
    es = new EventSource(url);
    es.onopen = () => {
      attempt = 0;
      onStatus?.('open');
    };
    es.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data) as LiveEventMessage);
      } catch (err) {
        console.error('Failed to parse SSE payload', err);
      }
    };
    es.onerror = () => {
      // the ticket was single use: close (no built-in retry with a spent ticket) and reconnect with a new one
      es?.close();
      es = null;
      retry();
    };
  };

  void connect();
  return () => {
    closed = true;
    clearTimeout(retryTimer);
    es?.close();
  };
}
