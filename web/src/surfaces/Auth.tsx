import { useState, type FormEvent } from "react";
import { toast } from "sonner";
import { createApi, jsonBody, message, type Session } from "@/api/client";
import { Button, Field, Password, Notice } from "@/components/shared";
// MIT, Mohammad Shehadeh, Hirael login-01, 21st.dev id 28353. Providers and demo branding omitted.
export function Auth({
  onLogin,
  notice,
}: {
  onLogin: (session: Session) => void;
  notice: string;
}) {
  const [register, setRegister] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(e.currentTarget));
    setBusy(true);
    setError("");
    try {
      const result = await createApi(null, () => {}).json<Session>(
        register ? "/register/" : "/login/",
        jsonBody("POST", data),
      );
      if (!result.session_token || !result.user_id)
        throw new Error("Sign in did not return a session. Please try again.");
      localStorage.setItem("eventScoutUser", JSON.stringify(result));
      onLogin(result);
      toast.success(register ? "Account created" : "Signed in");
    } catch (err) {
      setError(message(err));
    } finally {
      setBusy(false);
    }
  }
  return (
    <main id="main" tabIndex={-1} className="auth-page">
      <section className="auth-card" aria-labelledby="auth-title">
        <img
          width={72}
          height={72}
          className="mascot"
          src="/Circle%20Scout%20Mascot.jpeg"
          alt="Circle Scout mascot"
        />
        <h1 id="auth-title">
          {register ? "Join Event Scout" : "Sign in to Event Scout"}
        </h1>
        <p className="muted">Your contacts, ready for the next conversation.</p>
        {notice && <Notice error={notice} />}
        <form onSubmit={submit}>
          {register && (
            <Field
              label="Name"
              name="name"
              autoComplete="name"
              required
              maxLength={200}
            />
          )}
          <Field
            label="Email"
            name="email"
            type="email"
            autoComplete="email"
            required
          />
          <Password register={register} />
          {error && <Notice error={error} />}
          <Button type="submit" disabled={busy}>
            {busy ? "Please wait..." : register ? "Create account" : "Sign in"}
          </Button>
        </form>
        <p className="auth-switch">
          {register ? "Already have an account?" : "New to Event Scout?"}{" "}
          <Button
            variant="link"
            onClick={() => {
              setRegister(!register);
              setError("");
            }}
          >
            {register ? "Sign in" : "Create an account"}
          </Button>
        </p>
      </section>
    </main>
  );
}
