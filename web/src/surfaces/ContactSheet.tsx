import { useEffect, useState, type FormEvent } from "react";
import { Dialog } from "@base-ui/react/dialog";
import { X, Download } from "lucide-react";
import { toast } from "sonner";
import {
  type Api,
  type Contact,
  jsonBody,
  message,
  downloadBlob,
  vCard,
  safeUrl,
} from "@/api/client";
import { Button, Field, Notes, Notice, Loading } from "@/components/shared";
export const contactFields = [
  ["name", "Name"],
  ["title", "Job title"],
  ["company_name", "Company"],
  ["email", "Email"],
  ["phone", "Phone"],
  ["linkedin", "LinkedIn URL"],
] as const;
export type Draft = Record<string, string>;
export function ContactFields({
  draft,
  onChange,
}: {
  draft: Draft;
  onChange: (draft: Draft) => void;
}) {
  return (
    <>
      {contactFields.map(([key, label]) => (
        <Field
          key={key}
          label={label}
          value={draft[key] || ""}
          required={key === "name"}
          type={key === "email" ? "email" : "text"}
          onChange={(e) => onChange({ ...draft, [key]: e.target.value })}
        />
      ))}
      <Notes
        label="Notes"
        value={draft.notes || ""}
        onChange={(notes) => onChange({ ...draft, notes })}
      />
    </>
  );
}
// Adapted from cnippet-dev/v-sheet-1, id 26142, using the existing Base UI foundation.
export function ContactSheet({
  api,
  id,
  onClose,
  onSaved,
}: {
  api: Api;
  id: string;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [contact, setContact] = useState<Contact | null>(null);
  const [draft, setDraft] = useState<Draft>({});
  const [links, setLinks] = useState<{ label?: string; url: string }[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [version, setVersion] = useState(0);
  useEffect(() => {
    let current = true;
    setContact(null);
    setError("");
    api
      .json<{ contact: Contact }>(
        `/contact/${encodeURIComponent(id)}?${api.userQuery}`,
      )
      .then(({ contact }) => {
        if (!current) return;
        setContact(contact);
        setDraft(
          Object.fromEntries([
            ...contactFields.map(([k]) => [
              k,
              contact[k] === "N/A" ? "" : String(contact[k] || ""),
            ]),
            ["notes", contact.notes || ""],
          ]),
        );
        setLinks(contact.links || []);
      })
      .catch((e) => {
        if (current) setError(message(e));
      });
    return () => {
      current = false;
    };
  }, [api, id, version]);
  async function save(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (links.some((l) => !safeUrl(l.url)))
        throw new Error("Each link needs a full http or https URL.");
      const changed = Object.fromEntries(
        Object.entries(draft)
          .filter(([k, v]) => v !== (contact?.[k as keyof Contact] || ""))
          .map(([k, v]) => [
            k,
            ["email", "phone", "linkedin"].includes(k) ? v || "N/A" : v,
          ]),
      );
      await api.json(
        `/contact/${encodeURIComponent(id)}?${api.userQuery}`,
        jsonBody("PUT", { ...changed, links }),
      );
      toast.success("Contact saved");
      onSaved();
      onClose();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  async function remove() {
    setBusy(true);
    setError("");
    try {
      await api.json(`/contact/${encodeURIComponent(id)}?${api.userQuery}`, {
        method: "DELETE",
      });
      toast.success("Contact deleted");
      onSaved();
      onClose();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Dialog.Root
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <Dialog.Portal>
        <Dialog.Backdrop className="sheet-backdrop" />
        <Dialog.Popup className="contact-sheet">
          <header className="sheet-header">
            <div>
              <Dialog.Title>Contact details</Dialog.Title>
              <Dialog.Description>
                Edit details and keep your conversation notes.
              </Dialog.Description>
            </div>
            <Dialog.Close
              render={
                <Button
                  variant="ghost"
                  aria-label="Close contact"
                  disabled={busy}
                />
              }
            >
              <X />
            </Dialog.Close>
          </header>
          {error && (
            <Notice
              error={error}
              retry={!contact ? () => setVersion((v) => v + 1) : undefined}
            />
          )}{" "}
          {!contact && !error && <Loading>Loading contact...</Loading>}
          {contact && (
            <form onSubmit={save}>
              <ContactFields draft={draft} onChange={setDraft} />
              <section className="links-editor" aria-label="Contact links">
                <h3>Links</h3>
                {links.map((link, index) => (
                  <div className="link-row" key={index}>
                    <Field
                      label={`Link ${index + 1} label`}
                      value={link.label || ""}
                      onChange={(e) =>
                        setLinks(
                          links.map((l, i) =>
                            i === index ? { ...l, label: e.target.value } : l,
                          ),
                        )
                      }
                    />
                    <Field
                      label={`Link ${index + 1} URL`}
                      type="url"
                      required
                      value={link.url}
                      onChange={(e) =>
                        setLinks(
                          links.map((l, i) =>
                            i === index ? { ...l, url: e.target.value } : l,
                          ),
                        )
                      }
                    />
                    {safeUrl(link.url) && (
                      <a
                        href={safeUrl(link.url)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Open link {index + 1}
                      </a>
                    )}
                    <Button
                      variant="ghost"
                      aria-label={`Remove link ${index + 1}`}
                      onClick={() =>
                        setLinks(links.filter((_, i) => i !== index))
                      }
                    >
                      Remove
                    </Button>
                  </div>
                ))}
                <Button
                  variant="outline"
                  onClick={() => setLinks([...links, { label: "", url: "" }])}
                >
                  Add link
                </Button>
              </section>
              <footer className="sheet-footer">
                <Button type="submit" disabled={busy}>
                  {busy ? "Saving..." : "Save contact"}
                </Button>
                <Button
                  variant="outline"
                  disabled={busy}
                  onClick={() =>
                    downloadBlob(
                      new Blob([vCard(contact)], {
                        type: "text/vcard;charset=utf-8",
                      }),
                      "contact.vcf",
                    )
                  }
                >
                  <Download />
                  Download vCard
                </Button>
                {confirm ? (
                  <div className="delete-confirm" role="alert">
                    <p>Delete {contact.name}? This cannot be undone.</p>
                    <Button
                      variant="destructive"
                      disabled={busy}
                      onClick={remove}
                    >
                      Confirm delete
                    </Button>
                    <Button
                      variant="outline"
                      disabled={busy}
                      onClick={() => setConfirm(false)}
                    >
                      Keep contact
                    </Button>
                  </div>
                ) : (
                  <Button
                    variant="ghost"
                    className="danger"
                    disabled={busy}
                    onClick={() => setConfirm(true)}
                  >
                    Delete contact
                  </Button>
                )}
              </footer>
            </form>
          )}
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
