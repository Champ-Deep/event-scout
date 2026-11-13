import React, { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError, api } from "../../lib/api";
import { q } from "../../lib/queries";
import { UiProvider } from "../../ui/kit";
import { EventsAdmin } from "./EventsAdmin";
import { OperationNotice } from "./OperationNotice";
import {
  detailFixture,
  eventFixture,
  filesFixture,
  plannedFixture,
} from "./admin.fixtures";

const route = vi.hoisted(() => ({
  event: "Test Industry Forum",
  blocker: null as null | { shouldBlockFn: () => Promise<boolean> },
}));
vi.mock("@tanstack/react-router", () => ({
  useSearch: () => ({ event: route.event }),
  useNavigate: () => vi.fn(),
  useBlocker: (opts: typeof route.blocker) => {
    route.blocker = opts;
  },
  Link: ({ children }: { children: React.ReactNode }) => <a>{children}</a>,
}));
let root: Root;
let host: HTMLDivElement;
let qc: QueryClient;
const tick = () =>
  act(async () => {
    await new Promise((r) => setTimeout(r, 30));
  });
const button = (name: string) =>
  [...document.querySelectorAll("button")].find((b) => b.textContent === name)!;
const click = async (name: string) => {
  await act(async () => {
    button(name).click();
  });
  await tick();
};
const input = async (
  el: HTMLInputElement | HTMLTextAreaElement,
  value: string,
) => {
  await act(async () => {
    const proto =
      el.tagName === "TEXTAREA"
        ? HTMLTextAreaElement.prototype
        : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value")!.set!.call(el, value);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  });
};
function seed() {
  const opts = { updatedAt: Date.parse("2026-11-12T10:00:00Z") };
  qc.setQueryData(q.adminEvents().queryKey, [eventFixture], opts);
  qc.setQueryData(q.plannerEvents().queryKey, [plannedFixture], opts);
  qc.setQueryData(q.eventDetail(route.event).queryKey, detailFixture, opts);
  qc.setQueryData(q.files().queryKey, filesFixture, opts);
  qc.setQueryData(q.users().queryKey, [], opts);
}
async function mount() {
  await act(async () =>
    root.render(
      <QueryClientProvider client={qc}>
        <UiProvider>
          <EventsAdmin />
        </UiProvider>
      </QueryClientProvider>,
    ),
  );
  await tick();
}
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  Object.defineProperty(navigator, "onLine", {
    configurable: true,
    value: true,
  });
  window.matchMedia = vi.fn().mockReturnValue({
    matches: false,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  });
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  qc = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  route.event = "Test Industry Forum";
  seed();
  vi.spyOn(api, "adminEvents").mockResolvedValue([eventFixture]);
  vi.spyOn(api, "plannerEvents").mockResolvedValue([plannedFixture]);
  vi.spyOn(api, "eventDetail").mockResolvedValue(detailFixture);
  vi.spyOn(api, "files").mockResolvedValue(filesFixture);
  vi.spyOn(api, "users").mockResolvedValue([]);
});
afterEach(async () => {
  await act(async () => root.unmount());
  qc.clear();
  host.remove();
  document.body.innerHTML = "";
  vi.restoreAllMocks();
});

describe("Events and Exhibitors admin", () => {
  it("blocks shared API mutations offline before any HTTP request", async () => {
    Object.defineProperty(navigator, "onLine", {
      configurable: true,
      value: false,
    });
    const fetch = vi.spyOn(globalThis, "fetch");
    await expect(
      api.addExhibitor({ event: route.event, name: "Unsent" }),
    ).rejects.toMatchObject({
      status: 0,
      message: "Unavailable offline. Reconnect to make changes.",
    });
    expect(fetch).not.toHaveBeenCalled();
  });
  it("requires a corrected source after an identified pre-commit rejection", async () => {
    const request = vi
      .spyOn(api, "importExhibitors")
      .mockRejectedValue(new ApiError("Name is required", 422));
    await mount();
    await click("Import list");
    await input(document.querySelector("textarea")!, "[1]");
    await act(async () =>
      [...document.querySelectorAll("button")]
        .filter((b) => b.textContent === "Import list")
        .at(-1)!
        .click(),
    );
    await tick();
    expect(host.textContent).toContain("Import not started");
    await click("Review source");
    expect(button("Retry import").disabled).toBe(true);
    await input(document.querySelector("textarea")!, '[{"name":"Corrected"}]');
    expect(document.body.textContent).toContain("Ready to try again");
    expect(button("Retry import").disabled).toBe(false);
    expect(request).toHaveBeenCalledTimes(1);
  });
  it("keeps cached data, local search and timestamps while writes and file downloads are disabled offline", async () => {
    Object.defineProperty(navigator, "onLine", {
      configurable: true,
      value: false,
    });
    await mount();
    expect(host.textContent).toContain(
      "You are offline. Showing cached data. Changes are unavailable until you reconnect.",
    );
    expect(host.textContent).toContain("File not available offline.");
    for (const name of [
      "Add event",
      "Edit event",
      "Import list",
      "Add exhibitor",
      "Add material",
    ])
      expect(button(name).disabled).toBe(true);
    expect(
      document.querySelector<HTMLButtonElement>(
        '[aria-label="Get forum-guide.pdf"]',
      )!.disabled,
    ).toBe(true);
    expect(button("Retry connection").disabled).toBe(false);
    expect(host.querySelectorAll(".es-cache")).toHaveLength(5);
    await input(host.querySelector("input[type=search]")!, "not listed");
    expect(host.textContent).not.toContain("Test Exhibitor");
    expect(host.textContent).toContain(
      "Cached results only. More may be available online.",
    );
  });
  it("allows one retry while navigator still says offline, and never advances failed timestamps", async () => {
    Object.defineProperty(navigator, "onLine", {
      configurable: true,
      value: false,
    });
    vi.mocked(api.adminEvents).mockRejectedValue(new ApiError("offline", 0));
    vi.mocked(api.plannerEvents).mockRejectedValue(new ApiError("offline", 0));
    vi.mocked(api.files).mockRejectedValue(new ApiError("offline", 0));
    vi.mocked(api.eventDetail).mockRejectedValue(new ApiError("offline", 0));
    await mount();
    const before = [...host.querySelectorAll(".es-cache")].map(
      (e) => e.textContent,
    );
    await click("Retry connection");
    expect(api.adminEvents).toHaveBeenCalledTimes(1);
    expect(host.textContent).toContain(
      "Still offline. Cached data has not changed.",
    );
    expect(
      [...host.querySelectorAll(".es-cache")].map((e) => e.textContent),
    ).toEqual(before);
    vi.mocked(api.adminEvents).mockResolvedValue([eventFixture]);
    vi.mocked(api.plannerEvents).mockResolvedValue([plannedFixture]);
    vi.mocked(api.files).mockResolvedValue(filesFixture);
    vi.mocked(api.eventDetail).mockResolvedValue(detailFixture);
    await click("Retry connection");
    expect(host.textContent).toContain("Back online. Data is up to date.");
    expect(button("Add event").disabled).toBe(false);
    expect(api.adminEvents).toHaveBeenCalledTimes(2);
  });
  it("renders a cold offline view without fabricated data or online empty claims", async () => {
    qc.clear();
    Object.defineProperty(navigator, "onLine", {
      configurable: true,
      value: false,
    });
    await mount();
    expect(host.textContent).toContain(
      "No cached data is available for this view.",
    );
    expect(host.textContent).not.toContain("Test Exhibitor");
    expect(host.textContent).not.toContain("No events yet");
    expect([...host.querySelectorAll("h2")].map((h) => h.textContent)).toEqual([
      "Event website",
      "Event info",
      "Exhibitor directory",
      "Event material catalog",
    ]);
  });
  it("does not select the first event when selection is absent", async () => {
    route.event = "";
    await mount();
    expect(host.textContent).toContain("Select an event");
    expect(host.textContent).not.toContain("Event website");
  });
  it("imports non-replacing exactly once and keeps existing rows outside the pending block", async () => {
    let resolve!: (v: Awaited<ReturnType<typeof api.importExhibitors>>) => void;
    const request = vi.spyOn(api, "importExhibitors").mockImplementation(
      () =>
        new Promise((r) => {
          resolve = r;
        }),
    );
    await mount();
    await click("Import list");
    expect(document.body.textContent).not.toContain("Replace existing");
    await input(
      document.querySelector("textarea")!,
      '[{"name":"New exhibitor"}]',
    );
    const buttons = [...document.querySelectorAll("button")].filter(
      (b) => b.textContent === "Import list",
    );
    await act(async () => {
      buttons.at(-1)!.click();
      buttons.at(-1)!.click();
    });
    await tick();
    expect(request).toHaveBeenCalledTimes(1);
    expect(request).toHaveBeenCalledWith(
      [{ name: "New exhibitor" }],
      route.event,
      false,
    );
    expect(host.textContent).toContain("Importing exhibitor list...");
    expect(host.textContent).toContain("No row results yet.");
    expect(host.querySelector("tbody")!.textContent).toContain(
      "Test Exhibitor",
    );
    expect(button("Cancel").disabled).toBe(true);
    expect(button("Add material").disabled).toBe(true);
    expect(
      document.querySelector<HTMLButtonElement>(
        '[aria-label="Get forum-guide.pdf"]',
      )!.disabled,
    ).toBe(false);
    await act(async () =>
      resolve({
        count: 1,
        skipped: 0,
        replaced: 0,
        events: { [route.event]: 1 },
      }),
    );
    await tick();
    expect(host.textContent).toContain("Imported 1 exhibitors. Skipped 0 rows");
  });
  it("keeps unknown outcomes read-only after reconciliation and never retries the mutation", async () => {
    const request = vi
      .spyOn(api, "importExhibitors")
      .mockRejectedValue(new ApiError("lost response", 0));
    await mount();
    await click("Import list");
    await input(document.querySelector("textarea")!, '[{"name":"New"}]');
    await act(async () =>
      [...document.querySelectorAll("button")]
        .filter((b) => b.textContent === "Import list")
        .at(-1)!
        .click(),
    );
    await tick();
    expect(host.textContent).toContain("Import outcome unknown");
    await click("Refresh directory");
    expect(host.textContent).toContain("outcome is still unconfirmed");
    expect(request).toHaveBeenCalledTimes(1);
    expect(button("Import list").disabled).toBe(true);
  });
  it("material upload submits once, locks other writes and does not create a pending table row", async () => {
    const request = vi
      .spyOn(api, "uploadFile")
      .mockImplementation(() => new Promise(() => {}));
    await mount();
    await click("Add material");
    const fileInput =
      document.querySelector<HTMLInputElement>("input[type=file]")!;
    await act(async () => {
      Object.defineProperty(fileInput, "files", {
        value: [
          new File(["test material"], "pending-guide.txt", {
            type: "text/plain",
          }),
        ],
      });
      fileInput.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await click("Upload");
    await tick();
    expect(request).toHaveBeenCalledTimes(1);
    expect(request).toHaveBeenCalledWith(
      expect.objectContaining({
        filename: "pending-guide.txt",
        event: route.event,
        category: "general",
      }),
    );
    expect(host.textContent).toContain("Uploading material...");
    expect(button("Cancel").disabled).toBe(true);
    expect(button("Edit event").disabled).toBe(true);
    expect(host.querySelector("select")!.disabled).toBe(true);
    // The server can announce the save before this request's response reaches the client.
    await act(async () =>
      qc.setQueryData(q.files().queryKey, [
        ...filesFixture,
        {
          ...filesFixture[0]!,
          id: "pending-file",
          filename: "pending-guide.txt",
        },
      ]),
    );
    await tick();
    expect(
      [...host.querySelectorAll("tbody")].every(
        (t) => !t.textContent?.includes("pending-guide.txt"),
      ),
    ).toBe(true);
    expect(
      document.querySelector<HTMLButtonElement>(
        '[aria-label="Get forum-guide.pdf"]',
      )!.disabled,
    ).toBe(false);
  });
  it("rejects a differently scoped event_name before any import request", async () => {
    const request = vi.spyOn(api, "importExhibitors");
    await mount();
    await click("Import list");
    await input(
      document.querySelector("textarea")!,
      '[{"name":"Wrong target","event_name":"Another event"}]',
    );
    await act(async () =>
      [...document.querySelectorAll("button")]
        .filter((b) => b.textContent === "Import list")
        .at(-1)!
        .click(),
    );
    await tick();
    expect(document.body.textContent).toContain(
      "Some rows name a different event.",
    );
    expect(request).not.toHaveBeenCalled();
  });
  it("material pending names one source, leaves saved files downloadable and disables cancellation", async () => {
    await act(async () =>
      root.render(
        <OperationNotice
          operation={{
            kind: "upload",
            phase: "pending",
            source: "new-guide.pdf",
            scope: route.event,
            category: "General",
          }}
          onRefresh={vi.fn()}
          onReview={vi.fn()}
          onClose={vi.fn()}
        />,
      ),
    );
    expect(host.textContent).toContain("Uploading material...");
    expect(host.textContent).toContain(
      "Pending file: new-guide.pdf | Category: General",
    );
    expect(host.querySelector("table")).toBeNull();
    expect(button("Cancel").disabled).toBe(true);
    expect(host.textContent).toContain(
      "This request has already been sent and cannot be cancelled here.",
    );
    expect(host.textContent).not.toMatch(/\d+%|ETA/);
  });
});
