import { DurableObject } from "cloudflare:workers";
import { createAuth, loadAppUser } from "./auth";

interface Env {
  BACKEND: DurableObjectNamespace<EventScoutBackend>;
  ASSETS: Fetcher;
  CONTAINER_INSTANCE: "lite" | "basic" | "standard-1" | "standard-2" | "standard-3" | "standard-4";
  AUTO_PIPELINE_ENABLED: string;
  DATABASE_URL: string;
  APP_API_KEY: string;
  BETTER_AUTH_SECRET: string;
  OPENROUTER_API_KEY?: string;
  GEMINI_API_KEY?: string;
  ADMIN_EMAILS?: string;
  WEBHOOK_URL?: string;
  GAMMA_API_KEY?: string;
  OCR_MODELS?: string;
  SCORING_MODELS?: string;
  CHAT_MODEL?: string;
  PIPELINE_MODEL?: string;
}

const PORT = 8000;
// Keep the backend warm through an event day; FAISS rebuilds from Postgres on each cold start.
const INACTIVITY_TIMEOUT_MS = 4 * 60 * 60 * 1000;
const PASS_THROUGH = [
  "DATABASE_URL", "APP_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "ADMIN_EMAILS",
  "WEBHOOK_URL", "GAMMA_API_KEY", "OCR_MODELS", "SCORING_MODELS", "CHAT_MODEL",
  "PIPELINE_MODEL", "AUTO_PIPELINE_ENABLED",
] as const;

export class EventScoutBackend extends DurableObject<Env> {
  private starting: Promise<void> | undefined;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    const container = ctx.container!;
    if (container.running) {
      void ctx.blockConcurrencyWhile(() => container.setInactivityTimeout(INACTIVITY_TIMEOUT_MS));
    }
  }

  async fetch(request: Request): Promise<Response> {
    this.starting ??= this.startAndWaitForPort().finally(() => {
      this.starting = undefined;
    });
    await this.starting;

    const url = new URL(request.url);
    url.protocol = "http:";
    url.host = "container";
    const forwarded = new Request(url, request);
    forwarded.headers.delete("host");
    return this.ctx.container!.getTcpPort(PORT).fetch(forwarded);
  }

  private async startAndWaitForPort(): Promise<void> {
    const container = this.ctx.container!;
    if (!container.running) {
      const env: Record<string, string> = { PORT: String(PORT) };
      for (const key of PASS_THROUGH) {
        const value = this.env[key];
        if (value) env[key] = value;
      }
      container.start({
        image: container.images.base,
        instance: this.env.CONTAINER_INSTANCE || "standard-1",
        enableInternet: true, // Postgres, OpenRouter and Gemini are outside Cloudflare
        env,
      });
    }
    await container.setInactivityTimeout(INACTIVITY_TIMEOUT_MS);

    // Model download is baked into the image, but torch + FAISS rebuild take a while: allow ~3 min.
    const port = container.getTcpPort(PORT);
    let lastError: unknown;
    for (let attempt = 0; attempt < 360; attempt++) {
      try {
        const response = await port.fetch("http://container/health/", {
          signal: AbortSignal.timeout(2000),
        });
        await response.body?.cancel();
        if (response.ok) return;
        lastError = new Error(`Health check returned ${response.status}`);
      } catch (error) {
        lastError = error;
      }
      await scheduler.wait(500);
    }
    throw new Error(`Backend did not become ready on port ${PORT}`, { cause: lastError });
  }
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

// user_id the request claims to act as: query string or JSON body (the app uses both).
async function claimedUserIds(request: Request, url: URL): Promise<string[]> {
  const ids = url.searchParams.getAll("user_id");
  if ((request.headers.get("content-type") || "").includes("application/json") && request.body) {
    try {
      const body = (await request.clone().json()) as { user_id?: unknown };
      if (body && typeof body.user_id === "string") ids.push(body.user_id);
    } catch {
      // Not JSON after all; the backend will reject it.
    }
  }
  return ids;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/app" || url.pathname === "/app/") {
      const page = await env.ASSETS.fetch(new Request(new URL("/mobile-frontend.html", url)));
      const headers = new Headers(page.headers);
      headers.set("Cache-Control", "no-cache, no-store, must-revalidate");
      return new Response(page.body, { status: page.status, headers });
    }
    if (url.pathname === "/manifest.json" || url.pathname.startsWith("/icon-")) {
      return env.ASSETS.fetch(request);
    }

    const { auth, pool } = createAuth({
      connectionString: env.DATABASE_URL,
      secret: env.BETTER_AUTH_SECRET,
      baseURL: url.origin,
    });
    try {
      if (url.pathname.startsWith("/api/auth/")) {
        return await auth.handler(request);
      }

      const session = request.headers.has("cookie")
        ? await auth.api.getSession({ headers: request.headers })
        : null;

      if (url.pathname === "/api/me") {
        if (!session) return json({ detail: "Not signed in" }, 401);
        const adminEmails = (env.ADMIN_EMAILS || "").split(",").map((e) => e.trim().toLowerCase()).filter(Boolean);
        const user = await loadAppUser(pool, session.user.id, adminEmails);
        return user ? json({ status: "success", ...user }) : json({ detail: "Account not found" }, 404);
      }

      // Sign-in now goes through Better Auth; the old password endpoints stay closed here.
      if (url.pathname === "/login/" || url.pathname === "/register/") {
        return json({ detail: "Use /api/auth/sign-in/email or /api/auth/sign-up/email" }, 410);
      }

      // The browser never holds the backend key. A signed-in user gets it added here, and only for
      // requests that act as that same user.
      const headers = new Headers(request.headers);
      headers.delete("x-api-key");
      if (session) {
        const claimed = await claimedUserIds(request, url);
        if (claimed.some((id) => id !== session.user.id)) {
          return json({ detail: "This request is for a different account" }, 403);
        }
        headers.set("X-API-Key", env.APP_API_KEY);
      }
      // One named instance: FAISS lives in process memory, so every request must reach the same backend.
      return await env.BACKEND.getByName("primary").fetch(new Request(request, { headers }));
    } finally {
      ctx.waitUntil(pool.end().catch(() => undefined));
    }
  },
} satisfies ExportedHandler<Env>;
