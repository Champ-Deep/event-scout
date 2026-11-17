import { DurableObject } from "cloudflare:workers";

interface Env {
  BACKEND: DurableObjectNamespace<EventScoutBackend>;
  ASSETS: Fetcher;
  CONTAINER_INSTANCE: "lite" | "basic" | "standard-1" | "standard-2" | "standard-3" | "standard-4";
  AUTO_PIPELINE_ENABLED: string;
  DATABASE_URL: string;
  APP_API_KEY: string;
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

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
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
    // One named instance: FAISS lives in process memory, so every request must reach the same backend.
    return env.BACKEND.getByName("primary").fetch(request);
  },
} satisfies ExportedHandler<Env>;
