import { useEffect, useRef, useState, type FormEvent } from "react";
import { Camera, Images, Check, ScanLine } from "lucide-react";
import { toast } from "sonner";
import { type Api, ApiError, jsonBody, message } from "@/api/client";
import {
  putCard,
  queuedCards,
  removeCard,
  queueLock,
  type QueuedCard,
} from "@/lib/scan-queue";
import { Button, Notice, Loading } from "@/components/shared";
import { ContactFields, type Draft } from "./ContactSheet";
export function validateImage(file: File): string | null {
  if (
    ![
      "image/jpeg",
      "image/png",
      "image/webp",
      "image/heic",
      "image/heif",
    ].includes(file.type)
  )
    return `${file.name}: Choose a JPEG, PNG, WebP or HEIC image.`;
  if (file.size > 15e6) return `${file.name}: The limit is 15 MB per image.`;
  if (file.size === 0) return `${file.name}: This image is empty.`;
  return null;
}
function Thumbnail({ blob }: { blob: Blob }) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    const u = URL.createObjectURL(blob);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [blob]);
  return (
    <img
      width={56}
      height={72}
      className="thumbnail"
      src={url || undefined}
      alt="Queued business card"
    />
  );
}
type ScanReply = {
  status: "success" | "duplicate";
  existing_name?: string;
  contact_id?: string;
  extracted_fields?: { name?: string; company_name?: string };
};
// Upload Stack 44136: failures first, then active and queued cards, then a collapsed saved ledger.
// File Dropzone 19201: image validation, preview and drag handling adapted to camera-first capture.
export function Scan({
  api,
  userId,
  onSaved,
}: {
  api: Api;
  userId: string;
  onSaved: () => void;
}) {
  const [cards, setCards] = useState<QueuedCard[]>([]);
  const [ready, setReady] = useState(false);
  const [reading, setReading] = useState("");
  const [ledger, setLedger] = useState<
    { id: string; text: string; duplicate: boolean }[]
  >([]);
  const [error, setError] = useState("");
  const [tick, setTick] = useState(0);
  const [online, setOnline] = useState(navigator.onLine);
  const [manual, setManual] = useState(false);
  const [draft, setDraft] = useState<Draft>({});
  const [saving, setSaving] = useState(false);
  const [manualError, setManualError] = useState("");
  const camera = useRef<HTMLInputElement>(null);
  const gallery = useRef<HTMLInputElement>(null);
  const mounted = useRef(false);
  const active = useRef<AbortController | null>(null);
  useEffect(() => {
    mounted.current = true;
    queuedCards(userId)
      .then((c) => {
        if (mounted.current) {
          setCards(c);
          setReady(true);
        }
      })
      .catch((e) =>
        setError(`Could not open the offline queue. ${message(e)}`),
      );
    const update = () => setOnline(navigator.onLine);
    window.addEventListener("online", update);
    window.addEventListener("offline", update);
    return () => {
      mounted.current = false;
      active.current?.abort();
      window.removeEventListener("online", update);
      window.removeEventListener("offline", update);
    };
  }, [userId]);
  useEffect(() => {
    if (!ready || !online || queueLock.busy) return;
    const card = cards.find((c) => c.status !== "failed");
    if (!card) return;
    queueLock.busy = true;
    setReading(card.id);
    const ctl = new AbortController();
    active.current = ctl;
    (async () => {
      try {
        const form = new FormData();
        form.append("file", card.blob, card.filename || "business-card.jpg");
        const result = await api.json<ScanReply>(
          `/add_contact_from_image/?${api.userQuery}`,
          {
            method: "POST",
            body: form,
            signal: AbortSignal.any([ctl.signal, AbortSignal.timeout(30000)]),
          },
        );
        if (
          result.status !== "duplicate" &&
          (!result.contact_id || result.status !== "success")
        )
          throw new Error("The card was not saved. Please try again.");
        await removeCard(card.id);
        const text =
          result.status === "duplicate"
            ? `Already saved: ${result.existing_name || "contact"}`
            : [
                result.extracted_fields?.name || "Contact saved",
                result.extracted_fields?.company_name,
              ]
                .filter(Boolean)
                .join(" · ");
        if (mounted.current) {
          setCards((c) => c.filter((x) => x.id !== card.id));
          setLedger((l) => [
            ...l,
            { id: card.id, text, duplicate: result.status === "duplicate" },
          ]);
          toast.success(text);
          onSaved();
        }
      } catch (e) {
        if (ctl.signal.aborted) return;
        const failed = {
          ...card,
          status: "failed" as const,
          attempts: card.attempts + 1,
          error: message(e),
          unreadable: e instanceof ApiError && e.status === 400,
        };
        try {
          await putCard(failed);
        } catch {
          if (mounted.current)
            setError(
              "Could not update the offline queue. Keep this page open and try again.",
            );
        }
        if (mounted.current)
          setCards((c) => c.map((x) => (x.id === card.id ? failed : x)));
      } finally {
        queueLock.busy = false;
        active.current = null;
        if (mounted.current) {
          setReading("");
          setTick((t) => t + 1);
        }
      }
    })();
  }, [cards, ready, online, api, tick, onSaved]);
  async function add(files: File[]) {
    setError("");
    for (const file of files) {
      const invalid = validateImage(file);
      if (invalid) {
        setError((e) => [e, invalid].filter(Boolean).join(" "));
        continue;
      }
      const card: QueuedCard = {
        id: crypto.randomUUID(),
        userId,
        blob: file,
        filename: file.name,
        created: Date.now(),
        attempts: 0,
        status: "queued",
      };
      try {
        await putCard(card);
        setCards((c) => [...c, card]);
      } catch (e) {
        setError(
          `The photo could not be queued. Keep the original and try again. ${message(e)}`,
        );
      }
    }
  }
  async function retry(card: QueuedCard) {
    try {
      const next = { ...card, status: "queued" as const, error: undefined };
      await putCard(next);
      setCards((c) => c.map((x) => (x.id === card.id ? next : x)));
    } catch (e) {
      setError(message(e));
    }
  }
  async function remove(card: QueuedCard) {
    try {
      await removeCard(card.id);
      setCards((c) => c.filter((x) => x.id !== card.id));
    } catch (e) {
      setError(message(e));
    }
  }
  async function saveManual(e: FormEvent) {
    e.preventDefault();
    setSaving(true);
    setManualError("");
    try {
      await api.json(
        "/add_contact/",
        jsonBody("POST", {
          user_id: userId,
          contact: {
            ...draft,
            email: draft.email || "N/A",
            phone: draft.phone || "N/A",
            linkedin: draft.linkedin || "N/A",
          },
        }),
      );
      toast.success("Contact saved");
      setDraft({});
      setManual(false);
      onSaved();
    } catch (e) {
      setManualError(message(e));
    } finally {
      setSaving(false);
    }
  }
  const failed = cards.filter((c) => c.status === "failed");
  const pending = cards.filter((c) => c.status !== "failed");
  return (
    <>
      <header className="page-header">
        <div>
          <h1>Scan cards</h1>
          <p className="muted">Meet someone. Keep the connection.</p>
        </div>
        <ScanLine className="page-mark" aria-hidden="true" />
      </header>
      <section
        className="capture"
        aria-label="Add card photos"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          void add(Array.from(e.dataTransfer.files));
        }}
      >
        <Camera size={32} aria-hidden="true" />
        <h2>Save a business card</h2>
        <p>Take a clear photo or choose images from your gallery.</p>
        <div className="capture-actions">
          <Button onClick={() => camera.current?.click()} disabled={!ready}>
            <Camera />
            Take photo
          </Button>
          <Button
            variant="outline"
            onClick={() => gallery.current?.click()}
            disabled={!ready}
          >
            <Images />
            Choose photos
          </Button>
        </div>
        <p className="meta">JPEG, PNG, WebP or HEIC. Up to 15 MB each.</p>
        <input
          hidden
          ref={camera}
          type="file"
          accept="image/*"
          capture="environment"
          aria-label="Take card photo"
          onChange={(e) => {
            void add(Array.from(e.target.files || []));
            e.target.value = "";
          }}
        />
        <input
          hidden
          ref={gallery}
          type="file"
          multiple
          accept="image/*"
          aria-label="Choose card photos"
          onChange={(e) => {
            void add(Array.from(e.target.files || []));
            e.target.value = "";
          }}
        />
      </section>
      {error && <Notice error={error} />}{" "}
      {!ready && !error && <Loading>Opening your scan queue...</Loading>}
      {!online && (
        <p className="notice" role="status">
          You are offline. Queued photos stay on this device and will be read
          when you reconnect.
        </p>
      )}
      {failed.length > 0 && (
        <section className="queue-group">
          <h2>Needs attention · {failed.length}</h2>
          <ul>
            {failed.map((c) => (
              <li className="queue-row" key={c.id}>
                <Thumbnail blob={c.blob} />
                <div className="queue-content">
                  <strong>{c.filename}</strong>
                  <p className="danger">{c.error}</p>
                  <div className="row-actions">
                    <Button variant="outline" onClick={() => retry(c)}>
                      Retry
                    </Button>
                    {c.unreadable && (
                      <Button variant="outline" onClick={() => setManual(true)}>
                        Enter manually
                      </Button>
                    )}
                    <Button variant="ghost" onClick={() => remove(c)}>
                      Remove
                    </Button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
      {pending.length > 0 && (
        <section className="queue-group">
          <h2>Scan queue · {pending.length}</h2>
          <ul>
            {pending.map((c) => (
              <li
                className="queue-row"
                key={c.id}
                data-status={reading === c.id ? "reading" : "queued"}
              >
                <Thumbnail blob={c.blob} />
                <div className="queue-content">
                  <strong>{c.filename}</strong>
                  <p role="status">
                    {reading === c.id
                      ? "Reading card..."
                      : "Queued on this device"}
                  </p>
                  {reading === c.id ? (
                    <progress aria-label="Reading card" />
                  ) : (
                    <Button variant="ghost" onClick={() => remove(c)}>
                      Remove
                    </Button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
      {ledger.length > 0 && (
        <details className="saved-ledger">
          <summary>Completed · {ledger.length}</summary>
          <ul>
            {ledger.map((item) => (
              <li key={item.id}>
                <Check aria-hidden="true" />
                <span>{item.text}</span>
                <span className="meta">
                  {item.duplicate ? "Duplicate" : "Saved"}
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}
      {!cards.length && !ledger.length && ready && (
        <p className="queue-empty">
          No photos waiting. Your scan queue will appear here.
        </p>
      )}
      <section className="manual-section">
        <Button
          variant="link"
          aria-expanded={manual}
          onClick={() => setManual(!manual)}
        >
          {manual ? "Close manual entry" : "Enter a contact manually"}
        </Button>
        {manual && (
          <form onSubmit={saveManual}>
            <h2>New contact</h2>
            <ContactFields draft={draft} onChange={setDraft} />
            {manualError && <Notice error={manualError} />}
            <Button type="submit" disabled={saving}>
              {saving ? "Saving..." : "Save new contact"}
            </Button>
          </form>
        )}
      </section>
    </>
  );
}
