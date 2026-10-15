import type { RetrievedContact } from '../../lib/api';

/** One turn of a conversation started elsewhere (the contact's Ask half sheet). */
export interface HandoffTurn {
  role: 'user' | 'assistant';
  content: string;      // what the user sees
  prompt?: string | undefined;      // what was actually sent (e.g. with the contact's details), used as history
  format?: 'openui' | 'text' | undefined;
  contacts?: RetrievedContact[] | undefined;
}

// In-memory only: "Open in Ask" leaves the conversation here and the Ask screen takes it once.
let pending: HandoffTurn[] | null = null;

export const askHandoff = {
  put(turns: HandoffTurn[]) {
    pending = turns.length ? turns : null;
  },
  /** Returns the waiting conversation (if any) and clears it. */
  take(): HandoffTurn[] | null {
    const t = pending;
    pending = null;
    return t;
  },
};
