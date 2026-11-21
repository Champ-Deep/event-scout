import { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Toaster, toast } from "sonner";
import { createApi, readSession, type Session } from "@/api/client";
import { BottomNavBar, type Tab } from "@/components/shared";
import { Auth } from "@/surfaces/Auth";
import { Scan } from "@/surfaces/Scan";
import { Contacts } from "@/surfaces/Contacts";
import { ContactSheet } from "@/surfaces/ContactSheet";
import { Ask } from "@/surfaces/Ask";
import { Me } from "@/surfaces/Me";
import "./index.css";
export function App() {
  const [user, setUser] = useState<Session | null>(readSession);
  const [notice, setNotice] = useState("");
  const [tab, setTab] = useState<Tab>("scan");
  const [selected, setSelected] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const expired = useCallback(() => {
    toast.dismiss();
    localStorage.removeItem("eventScoutUser");
    setUser(null);
    setSelected(null);
    setTab("scan");
    setNotice("Your session has expired. Please log in again.");
  }, []);
  const api = useMemo(() => createApi(user, expired), [user, expired]);
  const changed = useCallback(() => setVersion((v) => v + 1), []);
  useEffect(() => {
    const sync = () => {
      const session = readSession();
      setUser(session);
      setSelected(null);
      setTab("scan");
    };
    window.addEventListener("storage", sync);
    return () => window.removeEventListener("storage", sync);
  }, []);
  async function logout() {
    await api.json("/logout/", { method: "POST" });
    toast.dismiss();
    localStorage.removeItem("eventScoutUser");
    setUser(null);
    setSelected(null);
    setTab("scan");
    setNotice("");
  }
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      {user ? (
        <div className="app-shell" key={user.user_id}>
          <header className="brand-bar">
            <img
              width={40}
              height={40}
              src="/Circle%20Scout%20Mascot.jpeg"
              alt=""
            />
            <span>Event Scout</span>
          </header>
          <main id="main" tabIndex={-1}>
            <div hidden={tab !== "scan"}>
              <Scan api={api} userId={user.user_id} onSaved={changed} />
            </div>
            {tab === "contacts" && (
              <Contacts
                api={api}
                version={version}
                onOpen={setSelected}
                onScan={() => setTab("scan")}
              />
            )}
            <div hidden={tab !== "ask"}>
              <Ask api={api} onOpen={setSelected} />
            </div>
            {tab === "me" && <Me api={api} user={user} onLogout={logout} />}
          </main>
          <BottomNavBar
            value={tab}
            onChange={(value) => {
              setTab(value);
              window.scrollTo(0, 0);
            }}
          />
          {selected && (
            <ContactSheet
              key={selected}
              api={api}
              id={selected}
              onClose={() => setSelected(null)}
              onSaved={changed}
            />
          )}
        </div>
      ) : (
        <Auth
          notice={notice}
          onLogin={(s) => {
            setUser(s);
            setNotice("");
          }}
        />
      )}
      <Toaster
        position="bottom-center"
        offset="calc(96px + env(safe-area-inset-bottom))"
        toastOptions={{ className: "scout-toast" }}
      />
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
