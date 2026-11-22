import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createApi,
  readSession,
  safeUrl,
  vCard,
  type Session,
} from "./api/client";
import { validateImage } from "./surfaces/Scan";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { BottomNavBar, ContactRows, Password } from "./components/shared";
const user: Session = {
  user_id: "fixture-user",
  session_token: "local-test-token",
  name: "QA Scout",
  email: "qa@example.invalid",
  is_admin: false,
};
afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
  cleanup();
});
describe("session and API contract", () => {
  it("shares the legacy storage key and rejects incomplete sessions", () => {
    expect(readSession()).toBeNull();
    localStorage.setItem("eventScoutUser", "bad json");
    expect(readSession()).toBeNull();
    localStorage.setItem("eventScoutUser", JSON.stringify({ user_id: "x" }));
    expect(readSession()).toBeNull();
    localStorage.setItem("eventScoutUser", JSON.stringify(user));
    expect(readSession()).toEqual(user);
  });
  it("uses same origin, session header and JSON content type", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", fetcher);
    await createApi(user, vi.fn()).json("/converse/", {
      method: "POST",
      body: "{}",
    });
    const [path, init] = fetcher.mock.calls[0];
    expect(path).toBe("/api/converse/");
    expect(init.headers.get("X-API-Key")).toBe(user.session_token);
    expect(init.headers.get("Content-Type")).toBe("application/json");
  });
  it("does not set a multipart content type", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", fetcher);
    await createApi(user, vi.fn()).request("/scan/", {
      method: "POST",
      body: new FormData(),
    });
    expect(fetcher.mock.calls[0][1].headers.has("Content-Type")).toBe(false);
  });
  it("expires protected sessions on any 401", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("{}", { status: 401 })),
    );
    const expire = vi.fn();
    await expect(createApi(user, expire).json("/contacts/")).rejects.toThrow(
      "Your session has expired",
    );
    expect(expire).toHaveBeenCalledOnce();
  });
  it("does not expire on incorrect sign-in credentials", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response('{"detail":"Invalid credentials"}', { status: 401 }),
        ),
    );
    const expire = vi.fn();
    await expect(createApi(null, expire).json("/login/")).rejects.toThrow(
      "Invalid credentials",
    );
    expect(expire).not.toHaveBeenCalled();
  });
  it("preserves network and server failure recovery", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
    await expect(createApi(user, vi.fn()).json("/contacts/")).rejects.toThrow(
      "Check your connection",
    );
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("unavailable", { status: 503 })),
    );
    await expect(createApi(user, vi.fn()).json("/contacts/")).rejects.toThrow(
      "Please try again",
    );
  });
});
describe("scan boundaries", () => {
  it("accepts photos but not arbitrary image payloads", () => {
    expect(
      validateImage(new File(["photo"], "card.jpg", { type: "image/jpeg" })),
    ).toBeNull();
    expect(
      validateImage(
        new File(["<svg/>"], "card.svg", { type: "image/svg+xml" }),
      ),
    ).toContain("Choose a JPEG");
  });
  it("rejects empty files and more than 15 MB", () => {
    expect(
      validateImage(new File([], "empty.png", { type: "image/png" })),
    ).toContain("empty");
    const file = new File(["x"], "large.jpg", { type: "image/jpeg" });
    Object.defineProperty(file, "size", { value: 15000001 });
    expect(validateImage(file)).toContain("15 MB");
  });
});
describe("safe output and field controls", () => {
  it("rejects executable links", () => {
    expect(safeUrl("javascript:alert(1)")).toBeUndefined();
    expect(safeUrl("data:text/html,x")).toBeUndefined();
    expect(safeUrl("https://example.invalid/card")).toBe(
      "https://example.invalid/card",
    );
  });
  it("escapes vCard delimiters and normalizes missing fields", () => {
    const card = vCard({
      id: "qa",
      name: "QA; Scout, Test",
      notes: "line one\nline two",
      email: "N/A",
    });
    expect(card).toContain("FN:QA\\; Scout\\, Test");
    expect(card).toContain("NOTE:line one\\nline two");
    expect(card).toContain("EMAIL:\r\n");
    expect(card).not.toContain("N/A");
    expect(card.endsWith("END:VCARD")).toBe(true);
  });
  it("navigates four tabs with named icon-only controls", () => {
    const change = vi.fn();
    render(<BottomNavBar value="scan" onChange={change} />);
    expect(screen.getAllByRole("button")).toHaveLength(4);
    fireEvent.click(screen.getByRole("button", { name: "Contacts" }));
    expect(change).toHaveBeenCalledWith("contacts");
    expect(screen.getByRole("button", { name: "Scan" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });
  it("opens a contact from the primary row without invented scores", () => {
    const open = vi.fn();
    render(
      <ContactRows
        contacts={[
          {
            id: "qa",
            name: "QA Scout",
            company_name: "Fixture Labs",
            lead_score: null,
          },
        ]}
        onOpen={open}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /QA Scout/ }));
    expect(open).toHaveBeenCalledWith("qa");
    expect(screen.queryByText(/Hot|Warm|Cold/)).toBeNull();
  });
  it("shows band words with valid server scores", () => {
    render(
      <ContactRows
        contacts={[
          {
            id: "qa",
            name: "QA Scout",
            lead_score: 75,
            lead_temperature: "hot",
          },
        ]}
        onOpen={vi.fn()}
      />,
    );
    expect(screen.getByText("75 Hot")).toBeVisible();
  });
  it("toggles password visibility without changing its value", () => {
    render(<Password />);
    const input = screen.getByLabelText("Password");
    fireEvent.change(input, { target: { value: "qa-pass-123" } });
    fireEvent.click(screen.getByRole("button", { name: "Show password" }));
    expect(input).toHaveAttribute("type", "text");
    expect(input).toHaveValue("qa-pass-123");
    fireEvent.click(screen.getByRole("button", { name: "Hide password" }));
    expect(input).toHaveAttribute("type", "password");
  });
});
