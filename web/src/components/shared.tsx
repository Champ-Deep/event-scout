import { useId, useState, type ComponentProps, type ReactNode } from "react";
import {
  ChevronRight,
  ScanLine,
  Users,
  MessageCircle,
  UserRound,
  Eye,
  EyeOff,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/coss-badge";
import type { Contact } from "@/api/client";
export { Button };
export function Field({
  label,
  multiline,
  ...props
}: ComponentProps<"input"> & { label: string; multiline?: false }) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <Input id={id} {...props} />
    </div>
  );
}
export function Notes({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <textarea
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={4}
      />
    </div>
  );
}
export function Notice({
  error,
  retry,
}: {
  error: string;
  retry?: () => void;
}) {
  return (
    <div className="notice" role="alert">
      <p>{error}</p>
      {retry && (
        <Button variant="outline" onClick={retry}>
          Try again
        </Button>
      )}
    </div>
  );
}
export function Loading({ children = "Loading..." }: { children?: ReactNode }) {
  return (
    <p className="loading" role="status">
      {children}
    </p>
  );
}
export function Password({ register = false }: { register?: boolean }) {
  const [show, setShow] = useState(false);
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>Password</label>
      <div className="password">
        <Input
          id={id}
          name="password"
          required
          minLength={register ? 8 : undefined}
          autoComplete={register ? "new-password" : "current-password"}
          type={show ? "text" : "password"}
        />
        <Button
          type="button"
          variant="ghost"
          aria-label={show ? "Hide password" : "Show password"}
          aria-pressed={show}
          onClick={() => setShow(!show)}
        >
          {show ? <EyeOff /> : <Eye />}
        </Button>
      </div>
    </div>
  );
}
// Adapted from 21st.dev arunachalam/bottom-nav-bar, id 8343. No decorative entry or label motion.
const tabs = [
  { id: "scan", label: "Scan", icon: ScanLine },
  { id: "contacts", label: "Contacts", icon: Users },
  { id: "ask", label: "Ask", icon: MessageCircle },
  { id: "me", label: "Me", icon: UserRound },
] as const;
export type Tab = (typeof tabs)[number]["id"];
export function BottomNavBar({
  value,
  onChange,
}: {
  value: Tab;
  onChange: (value: Tab) => void;
}) {
  return (
    <nav className="bottom-nav" aria-label="Main navigation">
      {tabs.map(({ id, label, icon: Icon }) => (
        <Button
          key={id}
          variant="ghost"
          aria-label={label}
          aria-current={value === id ? "page" : undefined}
          onClick={() => onChange(id)}
        >
          <Icon aria-hidden="true" />
          {value === id && <span>{label}</span>}
        </Button>
      ))}
    </nav>
  );
}
export function LeadBadge({ contact }: { contact: Contact }) {
  const t = contact.lead_temperature;
  const score = contact.lead_score;
  if (
    !t ||
    typeof score !== "number" ||
    !Number.isFinite(score) ||
    score < 0 ||
    score > 100
  )
    return null;
  return (
    <Badge className={`lead lead-${t}`}>
      {score} {t[0].toUpperCase() + t.slice(1)}
    </Badge>
  );
}
// Adapted from bundui/item7, id 28596. Data replaces sample people, primary action opens the record.
export function ContactRows({
  contacts,
  onOpen,
}: {
  contacts: Contact[];
  onOpen: (id: string) => void;
}) {
  return (
    <ul className="contact-list">
      {contacts.map((c) => (
        <li key={c.id}>
          <button className="contact-row" onClick={() => onOpen(c.id)}>
            <span className="avatar" aria-hidden="true">
              {c.photo_base64 ? (
                <img
                  width={44}
                  height={44}
                  alt=""
                  src={`data:image/jpeg;base64,${c.photo_base64}`}
                />
              ) : (
                c.name
                  .split(/\s+/)
                  .map((s) => s[0])
                  .slice(0, 2)
                  .join("")
              )}
            </span>
            <span className="contact-content">
              <strong>{c.name}</strong>
              <span>
                {[c.title, c.company_name]
                  .filter((s) => s && s !== "N/A")
                  .join(" · ") || "No company details"}
              </span>
            </span>
            <LeadBadge contact={c} />
            <ChevronRight size={18} aria-hidden="true" />
          </button>
        </li>
      ))}
    </ul>
  );
}
