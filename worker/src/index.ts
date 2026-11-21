// Event Scout API on Cloudflare (Worker + D1 + KV + Workers AI). Port of the Railway FastAPI app.
// The browser calls /api/<same path as the old FastAPI route>; everything else is static assets.
import { Ctx, Env, HttpError, User, json, first, sha256Hex, now } from "./lib";
import { routes as core } from "./routes/core";
import { routes as admin } from "./routes/admin";
import { routes as files } from "./routes/files";
import { routes as ai } from "./routes/ai";
import { sweep } from "./harbinger";

export type Auth = "public" | "user" | "admin";
export type Handler = (ctx: Ctx) => Promise<Response | unknown>;
export type Route = [method: string, path: string, handler: Handler, auth?: Auth];

interface Compiled {
  method: string;
  re: RegExp;
  keys: string[];
  handler: Handler;
  auth: Auth;
}

function compile(list: Route[]): Compiled[] {
  return list.map(([method, path, handler, auth = "user"]) => {
    const keys: string[] = [];
    const src = path.replace(/:(\w+)/g, (_, k) => {
      keys.push(k);
      return "([^/]+)";
    });
    // FastAPI routes here are declared with or without a trailing slash; accept both.
    return { method, re: new RegExp(`^${src.replace(/\/$/, "")}/?$`), keys, handler, auth };
  });
}

const TABLE = compile([...core, ...admin, ...files, ...ai]);
const PUBLIC_USER: User = { id: "", name: "", email: "", is_admin: 0, created_at: null };

async function sessionUser(env: Env, req: Request): Promise<User | null> {
  const bearer = req.headers.get("authorization")?.replace(/^Bearer\s+/i, "");
  const token = req.headers.get("x-api-key") || bearer;
  if (!token || token.length < 20) return null;
  const row = await first<User & { expires_at: string }>(
    env,
    `SELECT u.id, u.name, u.email, u.is_admin, u.created_at, s.expires_at
       FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?`,
    await sha256Hex(token),
  );
  if (!row || row.expires_at < now()) return null;
  return row;
}

/** user_id the request claims to act as (query string or JSON body): must be the session user. */
async function claimedUserIds(req: Request, url: URL): Promise<string[]> {
  const ids = url.searchParams.getAll("user_id");
  if ((req.headers.get("content-type") || "").includes("application/json")) {
    try {
      const b: any = await req.clone().json();
      if (b && typeof b.user_id === "string") ids.push(b.user_id);
    } catch {
      /* empty or invalid body: the handler reports it */
    }
  }
  return ids;
}

async function handleApi(req: Request, env: Env, exec: ExecutionContext, url: URL): Promise<Response> {
  const path = url.pathname.slice(4) || "/"; // strip "/api"
  let allowed: string[] = [];
  for (const r of TABLE) {
    const m = r.re.exec(path);
    if (!m) continue;
    if (r.method !== req.method) {
      allowed.push(r.method);
      continue;
    }
    const params: Record<string, string> = {};
    r.keys.forEach((k, i) => (params[k] = decodeURIComponent(m[i + 1])));

    let user = PUBLIC_USER;
    if (r.auth !== "public") {
      const u = await sessionUser(env, req);
      if (!u) return json({ detail: "Not signed in. Please log in again." }, 401);
      if (r.auth === "admin" && !u.is_admin) return json({ detail: "Admin access required" }, 403);
      const claimed = await claimedUserIds(req, url);
      if (claimed.some((id) => id !== u.id)) {
        return json({ detail: "This request is for a different account" }, 403);
      }
      user = u;
    }

    let cached: any;
    const ctx: Ctx = {
      req, env, exec, url, params, user,
      body: async () => {
        if (cached === undefined) {
          const t = await req.text();
          try {
            cached = t ? JSON.parse(t) : {};
          } catch {
            throw new HttpError(422, "Request body is not valid JSON");
          }
        }
        return cached;
      },
    };
    try {
      const out = await r.handler(ctx);
      return out instanceof Response ? out : json(out);
    } catch (e) {
      if (e instanceof HttpError) return json({ detail: e.detail }, e.status);
      console.log("unhandled", req.method, path, (e as Error)?.stack || String(e));
      return json({ detail: String((e as Error)?.message || e) }, 500);
    }
  }
  if (allowed.length) return json({ detail: "Method Not Allowed" }, 405);
  return json({ detail: "Not Found" }, 404);
}

const SECURITY_HEADERS: Record<string, string> = {
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Permissions-Policy": "camera=(self), geolocation=(), microphone=(self)",
};

export default {
  async fetch(req: Request, env: Env, exec: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    let res: Response;
    if (url.pathname === "/api" || url.pathname.startsWith("/api/")) {
      res = await handleApi(req, env, exec, url);
    } else if (url.pathname.startsWith("/card/")) {
      // Public share link: the viewer page reads the token from its own URL. Fetch the extensionless asset path:
      // "/card-viewer.html" answers 307 to "/card-viewer", which would send the browser there and drop the token.
      res = await env.ASSETS.fetch(new Request(new URL("/card-viewer", url), req));
    } else {
      res = await env.ASSETS.fetch(req);
    }
    const out = new Response(res.body, res);
    for (const [k, v] of Object.entries(SECURITY_HEADERS)) out.headers.set(k, v);
    if (url.pathname === "/" || url.pathname.endsWith(".html")) {
      out.headers.set("Cache-Control", "no-cache, no-store, must-revalidate");
      out.headers.set("X-Frame-Options", "DENY");
    }
    return out;
  },

  // Cron: push new or changed contacts (and their notes) to ChampHarbinger.
  async scheduled(_event: ScheduledController, env: Env, exec: ExecutionContext) {
    exec.waitUntil(sweep(env).then((r) => console.log("harbinger sweep", JSON.stringify(r))));
  },
} satisfies ExportedHandler<Env>;
