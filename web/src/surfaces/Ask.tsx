import { useEffect, useRef, useState } from "react";
import { type Api, type Contact, jsonBody, message } from "@/api/client";
import { ContactRows, Notice } from "@/components/shared";
import { Suggestions } from "@/components/ui/suggestions";
import {
  ChatInput,
  ChatInputTextArea,
  ChatInputSubmit,
} from "@/components/ui/chat-input";
type Turn = {
  role: "user" | "assistant";
  content: string;
  contacts?: Contact[];
};
const starters = [
  { id: "follow-up", label: "Who should I follow up with?" },
  { id: "summary", label: "Summarize my contacts" },
  { id: "pitch", label: "Help me prepare a pitch" },
];
export function Ask({
  api,
  onOpen,
}: {
  api: Api;
  onOpen: (id: string) => void;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const lock = useRef(false);
  const conversation = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const pane = conversation.current;
    if (!pane) return;
    if (busy || error) pane.scrollTop = pane.scrollHeight;
    else {
      const question = pane.querySelector(".turn-user:nth-last-child(2)");
      if (question)
        pane.scrollTop += question.getBoundingClientRect().top - pane.getBoundingClientRect().top;
    }
  }, [turns, busy, error]);
  async function send(value = query) {
    const q = value.trim();
    if (!q || lock.current) return;
    lock.current = true;
    setBusy(true);
    setError("");
    try {
      const result = await api.json<{
        response: string;
        retrieved_contacts: Contact[];
      }>(
        "/converse/",
        jsonBody("POST", {
          user_id: new URLSearchParams(api.userQuery).get("user_id"),
          query: q,
          conversation_history: turns.map(({ role, content }) => ({
            role,
            content,
          })),
          top_k: 4,
        }),
      );
      if (typeof result.response !== "string")
        throw new Error("No reply was returned. Please try again.");
      setTurns((t) => [
        ...t,
        { role: "user", content: q },
        {
          role: "assistant",
          content: result.response,
          contacts: result.retrieved_contacts,
        },
      ]);
      setQuery("");
    } catch (e) {
      setError(message(e));
      setQuery(q);
    } finally {
      setBusy(false);
      lock.current = false;
    }
  }
  return (
    <div className="ask-page">
      <header className="page-header">
        <div>
          <h1>Ask Event Scout</h1>
          <p className="muted">Make sense of the people you have met.</p>
        </div>
      </header>
      <div
        className="conversation"
        ref={conversation}
        tabIndex={0}
        role="log"
        aria-label="Conversation"
        aria-live="polite"
      >
        {!turns.length && (
          <section className="ask-start">
            <h2>Start with your contacts</h2>
            <p>
              Find someone, plan a follow-up or prepare for your next
              conversation.
            </p>
            <Suggestions
              items={starters}
              onSelect={(item) => {
                setQuery(item.label);
                void send(item.label);
              }}
              disabled={busy}
              itemClassName="suggestion"
            />
          </section>
        )}
        {turns.map((turn, i) => (
          <article className={`turn turn-${turn.role}`} key={i}>
            <h2>{turn.role === "user" ? "You" : "Event Scout"}</h2>
            <p>{turn.content}</p>
            {!!turn.contacts?.length && (
              <>
                <h3>Related contacts</h3>
                <ContactRows contacts={turn.contacts} onOpen={onOpen} />
              </>
            )}
          </article>
        ))}
        {busy && <p role="status">Finding an answer from your contacts...</p>}
        {error && <Notice error={error} retry={() => void send()} />}
      </div>
      <div className="composer">
        <label htmlFor="ask-query">Ask about your contacts</label>
        <ChatInput
          variant="unstyled"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onSubmit={() => void send()}
          loading={busy}
        >
          <ChatInputTextArea
            id="ask-query"
            placeholder="What would you like to know?"
            disabled={busy}
            maxLength={8000}
          />
          <ChatInputSubmit
            aria-label="Send question"
            disabled={busy || !query.trim()}
          />
        </ChatInput>
        <p className="meta">
          Answers can be incomplete. Check details before following up.
        </p>
      </div>
    </div>
  );
}
