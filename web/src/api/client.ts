export interface Session {
  user_id: string;
  name: string;
  email: string;
  is_admin: boolean;
  session_token: string;
}
export interface Contact {
  id: string;
  name: string;
  email?: string;
  phone?: string;
  linkedin?: string;
  company_name?: string;
  title?: string;
  notes?: string;
  photo_base64?: string;
  lead_score?: number | null;
  lead_temperature?: "hot" | "warm" | "cold" | null;
  links?: { label?: string; url: string }[];
}
export function readSession(): Session | null {
  try {
    const s = JSON.parse(localStorage.getItem("eventScoutUser") || "null");
    return s?.session_token && s?.user_id ? s : null;
  } catch {
    return null;
  }
}
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}
export function createApi(session: Session | null, expired: () => void) {
  async function request(path: string, init: RequestInit = {}) {
    const headers = new Headers(init.headers);
    if (session) headers.set("X-API-Key", session.session_token);
    if (typeof init.body === "string")
      headers.set("Content-Type", "application/json");
    let response: Response;
    try {
      response = await fetch("/api" + path, {
        ...init,
        headers,
        signal: init.signal ?? AbortSignal.timeout(30000),
      });
    } catch (error) {
      throw new ApiError(
        0,
        error instanceof DOMException &&
          ["TimeoutError", "AbortError"].includes(error.name)
          ? "The request timed out. Please try again."
          : "Could not connect. Check your connection and try again.",
      );
    }
    if (response.status === 401 && session) {
      expired();
      throw new ApiError(401, "Your session has expired. Please log in again.");
    }
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        response.status,
        typeof data.detail === "string"
          ? data.detail
          : "The request failed. Please try again.",
      );
    }
    return response;
  }
  return {
    request,
    json: async <T = Record<string, unknown>>(
      path: string,
      init: RequestInit = {},
    ): Promise<T> => (await request(path, init)).json(),
    userQuery: session ? `user_id=${encodeURIComponent(session.user_id)}` : "",
  };
}
export type Api = ReturnType<typeof createApi>;
export const jsonBody = (method: string, body: unknown) => ({
  method,
  body: JSON.stringify(body),
});
export const message = (error: unknown) =>
  error instanceof Error
    ? error.message
    : "Something went wrong. Please try again.";
export function safeUrl(value?: string): string | undefined {
  if (!value) return;
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : undefined;
  } catch {
    return;
  }
}
export function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function vCard(contact: Contact) {
  const escape = (value: string = "") =>
    value
      .replace(/\\/g, "\\\\")
      .replace(/\r?\n/g, "\\n")
      .replace(/;/g, "\\;")
      .replace(/,/g, "\\,");
  const clean = (s?: string) => (s === "N/A" ? "" : s);
  const [first = "", ...last] = contact.name.split(" ");
  return [
    "BEGIN:VCARD",
    "VERSION:3.0",
    `N:${escape(last.join(" "))};${escape(first)};;;`,
    `FN:${escape(contact.name)}`,
    `ORG:${escape(clean(contact.company_name))}`,
    `TITLE:${escape(clean(contact.title))}`,
    `TEL:${escape(clean(contact.phone))}`,
    `EMAIL:${escape(clean(contact.email))}`,
    `URL:${escape(safeUrl(contact.linkedin))}`,
    `NOTE:${escape(contact.notes)}`,
    ...(contact.photo_base64
      ? [`PHOTO;ENCODING=b;TYPE=JPEG:${contact.photo_base64}`]
      : []),
    "END:VCARD",
  ].join("\r\n");
}
