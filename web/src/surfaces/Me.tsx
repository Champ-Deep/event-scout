import { useEffect, useState, type FormEvent } from "react";
import { Tabs } from "@base-ui/react/tabs";
import { toast } from "sonner";
import {
  type Api,
  type Session,
  jsonBody,
  message,
  safeUrl,
} from "@/api/client";
import { Button, Field, Notes, Notice, Loading } from "@/components/shared";
import type { Draft } from "./ContactSheet";
const profileFields = [
  ["full_name", "Full name"],
  ["job_title", "Job title"],
  ["company_name", "Company"],
  ["company_website", "Company website"],
  ["linkedin_url", "LinkedIn URL"],
] as const;
const cardFields = [
  ...profileFields.slice(0, 3),
  ["email", "Email"],
  ["phone", "Phone"],
  ["website", "Website"],
  ["linkedin_url", "LinkedIn URL"],
] as const;
type Share = {
  shareable_url?: string;
  qr_code_base64?: string;
  qr_code_mime?: string;
};
// Settings1, felipemenezes098, 21st.dev id 28358. Product tabs replace notification and billing demos.
export function Me({
  api,
  user,
  onLogout,
}: {
  api: Api;
  user: Session;
  onLogout: () => Promise<void>;
}) {
  const [tab, setTab] = useState("profile");
  const [profile, setProfile] = useState<Draft | null>(null);
  const [card, setCard] = useState<Draft | null>(null);
  const [share, setShare] = useState<Share>({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let current = true;
    setError("");
    setProfile(null);
    setCard(null);
    Promise.all([
      api.json<{ profile: Draft }>(`/user/profile/?${api.userQuery}`),
      api.json<{ card: Draft; shareable_url?: string }>(
        `/user/card/?${api.userQuery}`,
      ),
    ])
      .then(([p, c]) => {
        if (current) {
          setProfile(p.profile);
          setCard(c.card);
          setShare({ shareable_url: c.shareable_url });
        }
      })
      .catch((e) => {
        if (current) setError(message(e));
      });
    return () => {
      current = false;
    };
  }, [api, retry]);
  async function save(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const keys =
        tab === "event"
          ? ["current_event_name", "current_event_description"]
          : tab === "profile"
            ? profileFields.map(([k]) => k)
            : [...cardFields.map(([k]) => k), "bio"];
      const data = tab === "card" ? card : profile;
      const body = Object.fromEntries(keys.map((k) => [k, data?.[k] || ""]));
      const r = await api.json<{
        profile?: Draft;
        card?: Draft;
        shareable_url?: string;
      }>(
        `/user/${tab === "card" ? "card" : "profile"}/?${api.userQuery}`,
        jsonBody("PUT", body),
      );
      if (tab === "card") {
        setCard(r.card!);
        setShare((s) => ({ ...s, shareable_url: r.shareable_url }));
      } else setProfile(r.profile!);
      toast.success(
        tab === "event"
          ? "Event saved"
          : tab === "card"
            ? "Card saved"
            : "Profile saved",
      );
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  async function generate() {
    setBusy(true);
    setError("");
    try {
      const result = await api.json<Share>(
        `/user/card/generate_token/?${api.userQuery}`,
        { method: "POST" },
      );
      setShare(result);
      toast.success("Card link ready");
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  async function copy() {
    try {
      await navigator.clipboard.writeText(share.shareable_url || "");
      toast.success("Link copied");
    } catch {
      setError(
        "Could not copy the link. Select the link below and copy it manually.",
      );
    }
  }
  return (
    <>
      <header className="page-header">
        <div>
          <h1>Me</h1>
          <p className="muted">
            {user.name} · {user.email}
          </p>
        </div>
      </header>
      <section className="settings-card">
        <Tabs.Root
          value={tab}
          onValueChange={(v) => {
            setTab(String(v));
            setError("");
          }}
        >
          <Tabs.List className="settings-tabs" aria-label="My settings">
            <Tabs.Tab value="profile">Profile</Tabs.Tab>
            <Tabs.Tab value="event">Event</Tabs.Tab>
            <Tabs.Tab value="card">My card</Tabs.Tab>
          </Tabs.List>
          {error && (
            <Notice
              error={error}
              retry={
                !profile || !card ? () => setRetry((v) => v + 1) : undefined
              }
            />
          )}{" "}
          {(!profile || !card) && !error && (
            <Loading>Loading your settings...</Loading>
          )}
          {profile && card && (
            <form onSubmit={save}>
              <Tabs.Panel value="profile">
                <h2>Your profile</h2>
                <p className="muted">
                  Help Event Scout understand who you work with.
                </p>
                {profileFields.map(([key, label]) => (
                  <Field
                    key={key}
                    label={label}
                    value={profile[key] || ""}
                    onChange={(e) =>
                      setProfile({ ...profile, [key]: e.target.value })
                    }
                  />
                ))}
              </Tabs.Panel>
              <Tabs.Panel value="event">
                <h2>Current event</h2>
                <p className="muted">
                  New contacts are tagged with this event.
                </p>
                <Field
                  label="Event name"
                  value={profile.current_event_name || ""}
                  onChange={(e) =>
                    setProfile({
                      ...profile,
                      current_event_name: e.target.value,
                    })
                  }
                />
                <Notes
                  label="Event description"
                  value={profile.current_event_description || ""}
                  onChange={(v) =>
                    setProfile({ ...profile, current_event_description: v })
                  }
                />
              </Tabs.Panel>
              <Tabs.Panel value="card">
                <h2>Your digital card</h2>
                <p className="muted">
                  Only add details you want to share. Blank fields keep saved
                  values.
                </p>
                {cardFields.map(([key, label]) => (
                  <Field
                    key={key}
                    label={label}
                    type={key === "email" ? "email" : "text"}
                    value={card[key] || ""}
                    onChange={(e) =>
                      setCard({ ...card, [key]: e.target.value })
                    }
                  />
                ))}
                <Notes
                  label="Bio"
                  value={card.bio || ""}
                  onChange={(bio) => setCard({ ...card, bio })}
                />
              </Tabs.Panel>
              <Button type="submit" disabled={busy}>
                {busy
                  ? "Please wait..."
                  : tab === "event"
                    ? "Save event"
                    : tab === "card"
                      ? "Save card"
                      : "Save profile"}
              </Button>
            </form>
          )}
          {tab === "card" && card && (
            <section className="card-share" aria-label="Share my card">
              <h3>Share your saved card</h3>
              <p>Save your details first, then generate your link.</p>
              <Button variant="outline" disabled={busy} onClick={generate}>
                {share.qr_code_base64 ? "Show QR again" : "Generate card link"}
              </Button>
              {safeUrl(share.shareable_url) && (
                <div className="share-result">
                  <a
                    className="share-url"
                    href={safeUrl(share.shareable_url)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {share.shareable_url}
                  </a>
                  <Button variant="outline" onClick={copy}>
                    Copy link
                  </Button>
                </div>
              )}
              {share.qr_code_base64 && share.qr_code_mime && (
                <img
                  width={220}
                  height={220}
                  className="card-qr"
                  src={`data:${share.qr_code_mime};base64,${share.qr_code_base64}`}
                  alt="QR code for your digital card"
                />
              )}
            </section>
          )}
        </Tabs.Root>
      </section>
      <footer className="account-actions">
        {user.is_admin && <a href="/admin.html">Admin console</a>}
        <Button
          variant="outline"
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            try {
              await onLogout();
            } catch (e) {
              setError(message(e));
              setBusy(false);
            }
          }}
        >
          Sign out
        </Button>
      </footer>
    </>
  );
}
