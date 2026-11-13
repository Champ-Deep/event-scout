import React, { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Link,
  useBlocker,
  useNavigate,
  useSearch,
} from "@tanstack/react-router";
import {
  ApiError,
  api,
  base64ToBlob,
  fileToBase64,
  saveBlob,
} from "../../lib/api";
import { q } from "../../lib/queries";
import type { EventBooth, EventFile, PlannerEvent } from "../../lib/types";
import {
  Button,
  Modal,
  Spinner,
  parseTime,
  fmtBytes,
  safeHref,
  useUi,
} from "../../ui/kit";
import { DETAIL_FIELDS, EVENT_FIELDS, EventForm } from "./EventForm";
import { OperationNotice, type Operation } from "./OperationNotice";
import { reportConnection, useConnection } from "./useConnection";
import "./events-admin.css";

const CATEGORIES = [
  ["general", "General"],
  ["brochure", "Brochure / one-pager"],
  ["pricing", "Pricing"],
  ["catalog", "Deck / presentation"],
  ["floorplan", "Floor plan"],
];
const ACCEPT =
  ".pdf,.ppt,.pptx,.doc,.docx,.xls,.xlsx,.csv,.txt,.png,.jpg,.jpeg";
const categoryLabel = (c: string | null) =>
  CATEGORIES.find(([key]) => key === (c || "general"))?.[1] || c;
const MISSING = "Not available offline. Reconnect to load this section.";
const DISABLED = "Unavailable offline. Reconnect to make changes.";
const blankBooth = {
  name: "",
  booth: "",
  hall: "",
  category: "",
  website: "",
  description: "",
};
const stamp = (v: string | null) => {
  const d = parseTime(v);
  return d
    ? d.toLocaleString([], { dateStyle: "medium", timeStyle: "long" })
    : "Not provided";
};
function CacheLabel({ at }: { at: number }) {
  return (
    <p className="es-cache">
      Cached. Last synced:{" "}
      {at
        ? new Date(at).toLocaleString([], {
            dateStyle: "medium",
            timeStyle: "long",
          })
        : "unknown"}
      .
    </p>
  );
}
function Values({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="es-event-values">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value || "Not provided"}</dd>
        </div>
      ))}
    </dl>
  );
}
function TableRegion({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="es-table-region"
      role="region"
      tabIndex={0}
      aria-label={label}
    >
      {children}
    </div>
  );
}

export function EventsAdmin() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const search = useSearch({ from: "/_auth/exhibitors" });
  const online = useConnection();
  const { confirm } = useUi();
  const event = search.event ?? "";
  const readOptions = {
    retry: false as const,
    networkMode: "always" as const,
    enabled: online,
  };
  const events = useQuery({ ...q.adminEvents(), ...readOptions });
  const planned = useQuery({ ...q.plannerEvents(), ...readOptions });
  const files = useQuery({ ...q.files(), ...readOptions });
  const detail = useQuery({
    ...q.eventDetail(event),
    ...readOptions,
    enabled: online && !!event,
  });
  const users = useQuery({ ...q.users(), ...readOptions });
  const names = [
    ...new Set([
      ...(events.data ?? []).map((e) => e.name),
      ...(planned.data ?? []).map((e) => e.name),
    ]),
  ];
  const current = planned.data?.find((e) => e.name === event);
  const reads = [events, planned, files, ...(event ? [detail] : [])];
  const networkError = reads.some(
    (r) => r.error instanceof ApiError && r.error.status === 0,
  );
  const offline = !online || networkError;
  const [refreshing, setRefreshing] = useState(false);
  const [connection, setConnection] = useState("");
  const [refreshFailed, setRefreshFailed] = useState(false);
  const [editing, setEditing] = useState<
    PlannerEvent | "new" | "details" | null
  >(null);
  const [dialog, setDialog] = useState<"import" | "upload" | null>(null);
  const [json, setJson] = useState("");
  const [source, setSource] = useState("Entered JSON");
  const [file, setFile] = useState<File | null>(null);
  const [meta, setMeta] = useState({
    category: "general",
    event,
    description: "",
  });
  const [preflight, setPreflight] = useState("");
  const [op, setOp] = useState<Operation | null>(null);
  const pending = op?.phase === "pending";
  const unknown =
    !!op &&
    ["unknown", "refreshing", "refreshFailed", "reviewed"].includes(op.phase);
  const [booth, setBooth] = useState<EventBooth | "new" | null>(null);
  const [boothForm, setBoothForm] = useState(blankBooth);
  const [boothBusy, setBoothBusy] = useState(false);
  const [boothError, setBoothError] = useState("");
  const [teamBusy, setTeamBusy] = useState(false);
  const [teamError, setTeamError] = useState("");
  const [text, setText] = useState("");
  const [hall, setHall] = useState("");
  const [category, setCategory] = useState("");
  const [fileSearch, setFileSearch] = useState("");
  const [fileCategory, setFileCategory] = useState("");
  const [scope, setScope] = useState("both");
  const [saved, setSaved] = useState("");
  const inflight = useRef(false);
  const savedRows = useRef<{ booths: EventBooth[]; files: EventFile[] }>({
    booths: [],
    files: [],
  });
  const lastImport = useRef("");
  const lastFile = useRef<File | null>(null);
  const unresolvedImport = useRef<string | null>(null);
  const unresolvedFile = useRef<File | null>(null);
  const rejectedSource = useRef<string | null>(null);
  const sourceSignature =
    (dialog ?? op?.kind) === "import"
      ? json
      : JSON.stringify([file?.name, file?.size, file?.lastModified, meta]);
  const correctionRequired =
    op?.phase === "rejected" && rejectedSource.current === sourceSignature;
  const repeatedSaved =
    dialog === "import"
      ? !!json && json === lastImport.current
      : dialog === "upload" && !!file && file === lastFile.current;
  const repeatedUnknown =
    dialog === "import"
      ? !!json && json === unresolvedImport.current
      : dialog === "upload" && !!file && file === unresolvedFile.current;
  const sameFilename =
    dialog === "upload" &&
    !!file &&
    files.data?.some(
      (f) =>
        f.filename === file.name &&
        (f.eventName === meta.event || !f.eventName),
    );
  const sourceWarning = repeatedUnknown
    ? dialog === "import"
      ? "Import outcome unknown. Refresh and review the directory before trying again."
      : "Upload outcome unknown. Refresh and review material before trying again."
    : repeatedSaved
      ? dialog === "import"
        ? "This source was already saved in this session. Review the saved records instead of importing it again."
        : "This file was already uploaded in this session. Review material instead of uploading it again."
      : sameFilename
        ? `A file named ${file!.name} is already listed for ${meta.event || "All events"}. This may be a duplicate. Review it before uploading.`
        : "";
  const sourceMessage = preflight || sourceWarning;
  const preflightTitle = ["Import not started", "Upload not started"].find(
    (title) => sourceMessage.startsWith(title + "."),
  );
  const previousOnline = useRef(online);
  const retrying = useRef(false);
  const locked =
    offline ||
    refreshing ||
    refreshFailed ||
    pending ||
    boothBusy ||
    teamBusy ||
    (!!event && !names.includes(event)) ||
    reads.some((r) => !r.data || r.isError);
  const cached = reads.some((r) => r.data !== undefined);
  const go = (name: string) =>
    navigate({
      to: "/exhibitors",
      search: (s) => ({ ...s, event: name || undefined }) as never,
      replace: true,
    });
  const refresh = async () => {
    if (retrying.current) return;
    retrying.current = true;
    setRefreshing(true);
    setConnection(
      online
        ? "Connection restored. Refreshing data. Changes remain unavailable."
        : "Checking connection...",
    );
    const result = await Promise.all([
      events.refetch(),
      planned.refetch(),
      files.refetch(),
      ...(event ? [detail.refetch()] : []),
    ]);
    const failed = result.some((r) => r.isError);
    setRefreshFailed(failed);
    setConnection(
      failed
        ? !navigator.onLine ||
          result.some(
            (r) => r.error instanceof ApiError && r.error.status === 0,
          )
          ? cached
            ? "Still offline. Cached data has not changed. Check your connection and try again."
            : "Still offline. Check your connection and try again."
          : cached
            ? "Connected, but data could not be refreshed. Showing cached data. Changes remain unavailable."
            : "Connected, but data could not be loaded. Try again."
        : "Back online. Data is up to date.",
    );
    if (!failed) {
      previousOnline.current = true;
      reportConnection(true);
    }
    setRefreshing(false);
    retrying.current = false;
  };
  useEffect(() => {
    if (networkError) reportConnection(false);
  }, [networkError]);
  useEffect(() => {
    if (online && !previousOnline.current) void refresh();
    previousOnline.current = online;
  }, [online]);
  useEffect(() => {
    if (offline) setConnection("");
  }, [offline]);
  useEffect(() => {
    if (!online && pending && op) {
      if (op.kind === "import") unresolvedImport.current = json;
      else unresolvedFile.current = file;
      setOp({ ...op, phase: "unknown" });
    }
  }, [online, pending]);
  useBlocker({
    shouldBlockFn: async () =>
      pending
        ? !(await confirm({
            title: "Leave before confirmation?",
            message:
              "This request may still save data. Leaving will not cancel it, and its result may be unknown. Return and review saved records before trying again.",
            confirmLabel: "Leave anyway",
            cancelLabel: "Stay here",
          }))
        : false,
    enableBeforeUnload: pending,
  });
  useEffect(() => {
    const before = (e: BeforeUnloadEvent) => {
      if ((dialog && (json || file)) || editing || booth) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", before);
    return () => window.removeEventListener("beforeunload", before);
  }, [dialog, json, file, editing, booth]);
  const refreshOperation = async (operation = op) => {
    if (!operation || operation.phase === "refreshing") return;
    const known =
      operation.phase === "success" || operation.phase === "savedStale";
    setOp({ ...operation, phase: "refreshing" });
    const result = await Promise.all([
      events.refetch(),
      ...(operation.kind === "import"
        ? [detail.refetch()]
        : [files.refetch(), detail.refetch()]),
    ]);
    setOp({
      ...operation,
      phase: result.some((r) => r.isError)
        ? known
          ? "savedStale"
          : "refreshFailed"
        : known
          ? "success"
          : "reviewed",
    });
  };
  const closeSource = async () => {
    if (inflight.current) return;
    if (unknown) {
      setDialog(null);
      return;
    }
    if (
      (dialog === "import" ? !!json : !!file || !!meta.description) &&
      !(await confirm({
        title:
          dialog === "import" ? "Discard this import?" : "Discard this upload?",
        message:
          "Nothing has been sent. Discard the selected source and entered details?",
        confirmLabel: "Discard",
        cancelLabel: "Keep editing",
      }))
    )
      return;
    setDialog(null);
    setJson("");
    setSource("Entered JSON");
    setFile(null);
    setMeta({ category: "general", event, description: "" });
    setPreflight("");
  };
  const submit = async () => {
    if (
      locked ||
      inflight.current ||
      unknown ||
      correctionRequired ||
      sourceWarning ||
      !dialog
    )
      return;
    let list: unknown[] = [];
    if (dialog === "import") {
      try {
        const parsed: unknown = JSON.parse(json);
        list = Array.isArray(parsed) ? parsed : [parsed];
      } catch {
        setPreflight(
          "Import not started. Enter valid JSON before importing. Nothing has been sent.",
        );
        return;
      }
      if (
        list.some(
          (row) =>
            row &&
            typeof row === "object" &&
            (("event_name" in row && row.event_name) ||
              ("event" in row && row.event)) &&
            (("event_name" in row && row.event_name) ||
              ("event" in row && row.event)) !== event,
        )
      ) {
        setPreflight(
          "Import not started. Some rows name a different event. Use a list for the selected event before importing. Nothing has been sent.",
        );
        return;
      }
    } else {
      if (
        !file ||
        file.size > 20 * 1024 * 1024 ||
        !/\.(pdf|pptx?|docx?|xlsx?|csv|txt|png|jpe?g)$/i.test(file.name)
      ) {
        setPreflight(
          "Upload not started. Choose a supported file no larger than 20 MB. Nothing has been sent.",
        );
        return;
      }
    }
    savedRows.current = {
      booths: detail.data?.booths ?? [],
      files: files.data ?? [],
    };
    inflight.current = true;
    // The live feed can announce a commit before its request receives confirmation.
    const operation: Operation = {
      kind: dialog,
      phase: "pending",
      source: dialog === "import" ? source : file!.name,
      scope: dialog === "import" ? event : meta.event || "All events",
      category: categoryLabel(meta.category) ?? meta.category,
    };
    setOp(operation);
    setDialog(null);
    setPreflight("");
    try {
      let message: string;
      if (dialog === "import") {
        const result = await api.importExhibitors(list, event, false);
        lastImport.current = json;
        message = `Imported ${result.count} exhibitors. Skipped ${result.skipped} rows without an exhibitor name or event. Saved records have not been removed.`;
      } else {
        const result = await api.uploadFile({
          filename: file!.name,
          contentBase64: await fileToBase64(file!),
          mimeType: file!.type,
          category: meta.category,
          description: meta.description,
          event: meta.event || undefined,
        });
        lastFile.current = file;
        message = `${result.filename} is saved and visible to reps under Me, Team files.`;
      }
      await refreshOperation({ ...operation, phase: "success", message });
    } catch (error) {
      // Only the identified request-schema rejection is known to precede the handler's commit.
      const rejected = error instanceof ApiError && error.status === 422;
      if (rejected) rejectedSource.current = sourceSignature;
      if (!rejected) {
        if (operation.kind === "import") unresolvedImport.current = json;
        else unresolvedFile.current = file;
      }
      setOp({
        ...operation,
        phase: rejected ? "rejected" : "unknown",
        ...(rejected
          ? {
              message:
                error.message.split("\n")[0]?.slice(0, 300) ||
                "Review the source and required fields.",
            }
          : {}),
      });
    } finally {
      inflight.current = false;
    }
  };
  const operationNotice = op ? (
    <OperationNotice
      operation={op}
      onRefresh={() => void refreshOperation()}
      onReview={() => {
        if (op.phase === "success") {
          document
            .getElementById(
              op.kind === "import" ? "exhibitor-directory" : "event-materials",
            )
            ?.focus();
        } else {
          setDialog(op.kind);
          setPreflight(
            unknown
              ? "Review only. Do not resend the full source while the outcome is unknown."
              : "Review the corrected source and submit it again. Existing saved records will stay.",
          );
        }
      }}
      onClose={() => setOp(null)}
    />
  ) : null;
  const visibleFiles = (
    pending ? savedRows.current.files : (files.data ?? [])
  ).filter(
    (f) =>
      (!f.eventName || f.eventName === event) &&
      (scope !== "event" || f.eventName === event) &&
      (scope !== "shared" || !f.eventName) &&
      (!fileCategory || (f.category || "general") === fileCategory) &&
      (!fileSearch ||
        `${f.filename} ${f.description}`
          .toLowerCase()
          .includes(fileSearch.toLowerCase())),
  );
  const booths = pending
    ? savedRows.current.booths
    : (detail.data?.booths ?? []);
  const shownBooths = booths.filter(
    (b) =>
      (!hall || b.hall === hall) &&
      (!category || b.category === category) &&
      (!text ||
        [b.name, b.booth, b.hall, b.category, b.description, b.website]
          .join(" ")
          .toLowerCase()
          .includes(text.toLowerCase())),
  );
  const cacheLabel = (at: number) =>
    offline || refreshFailed ? <CacheLabel at={at} /> : null;
  const openUpload = () => {
    setMeta({ category: "general", event, description: "" });
    setFile(null);
    setPreflight("");
    setDialog("upload");
  };
  const download = async (id: string) => {
    if (offline) return;
    try {
      const f = await api.downloadFile(id);
      saveBlob(base64ToBlob(f.contentBase64, f.mimeType), f.filename);
      void files.refetch();
    } catch {
      setSaved("Could not download the file. Try again when connected.");
    }
  };
  if (
    reads.some(
      (r) =>
        r.error instanceof ApiError &&
        (r.error.status === 401 || r.error.status === 403),
    )
  )
    return (
      <div role="alert">
        Your account does not have access to this view. Cached content is
        unavailable.
      </div>
    );
  return (
    <div className="es-events">
      {offline || refreshing || refreshFailed || connection ? (
        <div className="es-notice" role="status">
          <p>
            {connection ||
              (cached
                ? "You are offline. Showing cached data. Changes are unavailable until you reconnect."
                : "You are offline. No cached data is available for this view. Reconnect to load events and materials.")}
          </p>
          {connection !== "Back online. Data is up to date." ? (
            <Button disabled={refreshing} onClick={() => void refresh()}>
              {refreshing
                ? "Checking connection..."
                : refreshFailed && !offline
                  ? "Retry refresh"
                  : "Retry connection"}
            </Button>
          ) : null}
        </div>
      ) : null}
      <div>
        <div className="es-section-title">
          <h1>Events and Exhibitors</h1>
          <Button
            variant="primary"
            disabled={locked}
            title={offline ? DISABLED : undefined}
            onClick={() => setEditing("new")}
          >
            Add event
          </Button>
        </div>
        {names.length ? (
          <label className="max-w-xl">
            Event
            <select
              aria-label="Event"
              value={event}
              disabled={pending || refreshing}
              onChange={(e) => void go(e.target.value)}
            >
              <option value="">Select an event</option>
              {names.map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
        ) : null}
        {offline ? (
          <p className="mt-3">
            Reconnect to add events, edit event info, or upload material.
          </p>
        ) : null}
        {cacheLabel(
          Math.min(
            events.dataUpdatedAt || Infinity,
            planned.dataUpdatedAt || Infinity,
          ) === Infinity
            ? 0
            : Math.min(
                events.dataUpdatedAt || Infinity,
                planned.dataUpdatedAt || Infinity,
              ),
        )}
      </div>
      {saved ? <p role="status">{saved}</p> : null}
      {reads.some(
        (r) =>
          r.isError && !(r.error instanceof ApiError && r.error.status === 0),
      ) ? (
        <div role="alert">
          Could not load this view.{" "}
          <Button onClick={() => void refresh()}>Retry refresh</Button>
        </div>
      ) : null}
      {!names.length && !offline && (events.isPending || planned.isPending) ? (
        <p role="status">
          <Spinner /> Loading events...
        </p>
      ) : !names.length && offline ? (
        <>
          <p>
            {event
              ? `Event: ${event}`
              : events.data && planned.data
                ? "No events in this cached snapshot."
                : "No cached event selection is available."}
          </p>
          {[
            { title: "Event website", actions: ["Edit event"] },
            { title: "Event info", actions: [] },
            {
              title: "Exhibitor directory",
              actions: ["Import list", "Add exhibitor"],
            },
            { title: "Event material catalog", actions: ["Add material"] },
          ].map((section) => (
            <section key={section.title}>
              <div className="es-section-title">
                <h2>{section.title}</h2>
                <div className="es-actions">
                  {section.actions.map((action) => (
                    <Button key={action} disabled title={DISABLED}>
                      {action}
                    </Button>
                  ))}
                </div>
              </div>
              <p>{MISSING}</p>
              {section.title === "Event website" ? (
                <p>
                  Scraper not connected. Reconnect to enter event info manually
                  or import an exhibitor list. No scraping or import is running.
                </p>
              ) : null}
              {section.title === "Event material catalog" ? (
                <p>
                  When online, uploads are visible to reps immediately. Upload
                  is unavailable offline.
                </p>
              ) : null}
            </section>
          ))}
        </>
      ) : !names.length ? (
        <>
          <section>
            <h2>
              {offline
                ? cached
                  ? "No events in this cached snapshot."
                  : "Events"
                : "No events yet"}
            </h2>
            <p>
              {offline
                ? MISSING
                : "Add an event website and details for your team."}
            </p>
            <p>Scraper not connected. You can enter event details manually.</p>
          </section>
          <section>
            <h2>Event material catalog</h2>
            <p>
              Shared material is still available.{" "}
              <Link to="/files">Open Team files</Link>
            </p>
          </section>
        </>
      ) : !event ? (
        <section>
          <h2>Select an event</h2>
          <p>
            Select an event to view its saved details, exhibitor directory and
            material.
          </p>
        </section>
      ) : (
        <>
          {!names.includes(event) ? (
            <p role="alert">
              The selected event is no longer available. Select another event to
              continue.
            </p>
          ) : null}
          <section>
            <div className="es-section-title">
              <h2>Event website</h2>
              <Button
                disabled={locked}
                title={offline ? DISABLED : undefined}
                onClick={() => setEditing(current ?? "details")}
              >
                Edit event
              </Button>
            </div>
            <Values
              rows={[
                [
                  "Website",
                  offline ? (
                    current?.website
                  ) : current?.website && safeHref(current.website) ? (
                    <a
                      className="underline"
                      href={safeHref(current.website)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {current.website}
                    </a>
                  ) : null,
                ],
              ]}
            />
            <div className="es-notice">
              <strong>Scraper not connected</strong>
              <p>
                {offline
                  ? "Reconnect to enter event info manually or import an exhibitor list."
                  : "Enter event info manually or import an exhibitor list. Saving a website does not start an import."}{" "}
                {!op || ["success", "rejected"].includes(op.phase)
                  ? "No scraping or import is running."
                  : ""}
              </p>
            </div>
            {!current && offline ? <p>{MISSING}</p> : null}
            {cacheLabel(planned.dataUpdatedAt)}
          </section>
          <section>
            <h2 className="mb-4">Event info</h2>
            {current ? (
              <>
                <Values
                  rows={[
                    ["Name", current.name],
                    ["Status", current.status],
                    [
                      "Dates",
                      [current.startDate, current.endDate]
                        .filter(Boolean)
                        .join(" to "),
                    ],
                    ...EVENT_FIELDS.filter(([key]) => key !== "name").map(
                      ([key, label]): [string, React.ReactNode] => [
                        label,
                        current[key],
                      ],
                    ),
                    ["Description", current.description],
                  ]}
                />
                <details>
                  <summary>Logistics and organiser contact</summary>
                  <Values
                    rows={DETAIL_FIELDS.map(([key, label]) => [
                      label,
                      current.details[key],
                    ])}
                  />
                </details>
                <details>
                  <summary>Team</summary>
                  {current.team.map((member) => (
                    <div key={member.userId} className="es-actions my-2">
                      <span>{member.name}</span>
                      <label>
                        Team role
                        <select
                          aria-label={`Team role for ${member.name}`}
                          disabled={locked || teamBusy}
                          value={member.role}
                          onChange={async (e) => {
                            if (locked) return;
                            setTeamBusy(true);
                            try {
                              await api.setPlannerTeam(
                                current.id,
                                current.team.map((m) => ({
                                  userId: m.userId,
                                  role:
                                    m.userId === member.userId
                                      ? (e.target.value as "lead" | "member")
                                      : m.role,
                                })),
                              );
                              await planned.refetch();
                            } catch {
                              setTeamError(
                                "Could not save the team. Try again.",
                              );
                            } finally {
                              setTeamBusy(false);
                            }
                          }}
                        >
                          <option value="lead">lead</option>
                          <option value="member">member</option>
                        </select>
                      </label>
                    </div>
                  ))}
                  <label>
                    Team member
                    <select
                      aria-label="Team member"
                      value=""
                      disabled={locked || users.isPending || users.isError}
                      onChange={async (e) => {
                        if (locked || !e.target.value) return;
                        setTeamBusy(true);
                        try {
                          await api.setPlannerTeam(current.id, [
                            ...current.team.map((m) => ({
                              userId: m.userId,
                              role: m.role,
                            })),
                            { userId: e.target.value, role: "member" },
                          ]);
                          await planned.refetch();
                        } catch {
                          setTeamError("Could not save the team. Try again.");
                        } finally {
                          setTeamBusy(false);
                        }
                      }}
                    >
                      <option value="">Assign a team member</option>
                      {(users.data ?? [])
                        .filter(
                          (u) => !current.team.some((m) => m.userId === u.id),
                        )
                        .map((u) => (
                          <option key={u.id} value={u.id}>
                            {u.name}
                          </option>
                        ))}
                    </select>
                  </label>
                  {teamError ? <p role="alert">{teamError}</p> : null}
                </details>
                <div className="mt-5">
                  <Values
                    rows={[
                      ["Source", current.source],
                      [
                        "Source URL",
                        current.sourceUrl &&
                        !offline &&
                        safeHref(current.sourceUrl) ? (
                          <a
                            className="underline"
                            href={safeHref(current.sourceUrl)}
                          >
                            {current.sourceUrl}
                          </a>
                        ) : (
                          current.sourceUrl
                        ),
                      ],
                      ["Created by", current.createdBy],
                      ["Created at", stamp(current.createdAt)],
                      ["Updated at", stamp(current.updatedAt)],
                    ]}
                  />
                </div>
              </>
            ) : (
              <>
                <p>{offline ? MISSING : "Not provided"}</p>
                <p>{event}</p>
              </>
            )}
            {cacheLabel(planned.dataUpdatedAt)}
          </section>
          <section id="exhibitor-directory" tabIndex={-1}>
            <div className="es-section-title">
              <h2>Exhibitor directory</h2>
              <div className="es-actions">
                {json && !dialog && !pending && !op ? (
                  <Button disabled={locked} onClick={() => setDialog("import")}>
                    Back to source
                  </Button>
                ) : null}
                <Button
                  disabled={locked || unknown}
                  title={offline ? DISABLED : undefined}
                  onClick={() => {
                    setDialog("import");
                    setPreflight("");
                  }}
                >
                  Import list
                </Button>
                <Button
                  disabled={locked}
                  title={offline ? DISABLED : undefined}
                  onClick={() => {
                    setBoothForm(blankBooth);
                    setBoothError("");
                    setBooth("new");
                  }}
                >
                  Add exhibitor
                </Button>
              </div>
            </div>
            {op?.kind === "import" ? operationNotice : null}
            <div className="es-filters">
              <label>
                Search exhibitors
                <input
                  type="search"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                />
              </label>
              <label>
                Hall
                <select
                  aria-label="Hall"
                  value={hall}
                  onChange={(e) => setHall(e.target.value)}
                >
                  <option value="">All</option>
                  {[...new Set(booths.map((b) => b.hall).filter(Boolean))].map(
                    (h) => (
                      <option key={h} value={h!}>
                        {h}
                      </option>
                    ),
                  )}
                </select>
              </label>
              <label>
                Category
                <select
                  aria-label="Category"
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                >
                  <option value="">All</option>
                  {[
                    ...new Set(booths.map((b) => b.category).filter(Boolean)),
                  ].map((c) => (
                    <option key={c} value={c!}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {offline ? <p>Searching cached exhibitors only.</p> : null}
            {!detail.data ? (
              <p>
                {offline
                  ? MISSING
                  : detail.isPending
                    ? "Loading exhibitors..."
                    : "Could not load exhibitors."}
              </p>
            ) : (
              <TableRegion label="Exhibitor directory, scroll horizontally">
                <table>
                  <thead>
                    <tr>
                      {["Name", "Booth", "Hall", "Category", "Action"].map(
                        (h) => (
                          <th key={h}>{h}</th>
                        ),
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {shownBooths.map((b) => (
                      <tr key={b.id}>
                        <td>{b.name}</td>
                        <td>{b.booth || "Not provided"}</td>
                        <td>{b.hall || "Not provided"}</td>
                        <td>{b.category || "Not provided"}</td>
                        <td>
                          <Button
                            aria-label={`Edit ${b.name}`}
                            disabled={locked}
                            onClick={() => {
                              setBoothForm({
                                name: b.name,
                                booth: b.booth || "",
                                hall: b.hall || "",
                                category: b.category || "",
                                website: b.website || "",
                                description: b.description || "",
                              });
                              setBooth(b);
                            }}
                          >
                            Edit
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!shownBooths.length ? (
                  <p className="p-4">
                    {text || hall || category
                      ? "No matching exhibitors."
                      : "No exhibitors in this saved directory."}
                  </p>
                ) : null}
              </TableRegion>
            )}
            {offline ? (
              <p>Cached results only. More may be available online.</p>
            ) : null}
            {pending && op?.kind === "import" ? (
              <p>
                Saved records only. Pending items are not shown as imported
                rows.
              </p>
            ) : null}
            {cacheLabel(detail.dataUpdatedAt)}
          </section>
          <section id="event-materials" tabIndex={-1}>
            <div className="es-section-title">
              <h2>Event material catalog</h2>
              <Button
                disabled={locked || unknown}
                title={offline ? DISABLED : undefined}
                onClick={openUpload}
              >
                Add material
              </Button>
            </div>
            {op?.kind === "upload" ? operationNotice : null}
            <div className="es-filters">
              <label>
                Search material
                <input
                  type="search"
                  value={fileSearch}
                  onChange={(e) => setFileSearch(e.target.value)}
                />
              </label>
              <label>
                Category
                <select
                  aria-label="Category"
                  value={fileCategory}
                  onChange={(e) => setFileCategory(e.target.value)}
                >
                  <option value="">All</option>
                  {CATEGORIES.map(([key, label]) => (
                    <option key={key} value={key}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Scope
                <select
                  aria-label="Scope"
                  value={scope}
                  onChange={(e) => setScope(e.target.value)}
                >
                  <option value="both">Event + shared</option>
                  <option value="event">This event only</option>
                  <option value="shared">Shared only</option>
                </select>
              </label>
            </div>
            {!files.data ? (
              <p>
                {offline
                  ? MISSING
                  : files.isPending
                    ? "Loading material..."
                    : "Could not load material."}
              </p>
            ) : (
              <TableRegion label="Event material catalog, scroll horizontally">
                <table>
                  <thead>
                    <tr>
                      {[
                        "File",
                        "Category",
                        "Event",
                        "Size",
                        "Downloads",
                        "Added",
                        "Action",
                      ].map((h, i) => (
                        <th
                          key={h}
                          className={
                            i > 2 && i < 6 ? "es-secondary-col" : undefined
                          }
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {visibleFiles.map((f) => (
                      <tr key={f.id}>
                        <td className="es-file-name">
                          <strong>{f.filename}</strong>
                          <small>
                            {f.fileType.toUpperCase()}
                            {f.placeholder ? " | Placeholder" : ""}
                          </small>
                          <small>{f.description}</small>
                          <details className="md:hidden">
                            <summary>File details</summary>
                            {fmtBytes(f.fileSize)}. Downloads: {f.downloadCount}
                            . Added: {stamp(f.createdAt)}
                          </details>
                        </td>
                        <td>{categoryLabel(f.category)}</td>
                        <td>{f.eventName || "All events"}</td>
                        <td className="es-secondary-col">
                          {fmtBytes(f.fileSize)}
                        </td>
                        <td className="es-secondary-col">{f.downloadCount}</td>
                        <td className="es-secondary-col">
                          <time dateTime={f.createdAt ?? undefined}>
                            {stamp(f.createdAt)}
                          </time>
                        </td>
                        <td>
                          <Button
                            disabled={offline || f.placeholder}
                            aria-label={`Get ${f.filename}`}
                            onClick={() => void download(f.id)}
                          >
                            Get
                          </Button>
                          {offline ? (
                            <small>File not available offline.</small>
                          ) : null}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!visibleFiles.length ? (
                  <div className="p-4">
                    <p>
                      {fileSearch || fileCategory || scope !== "both"
                        ? "No matching material"
                        : "No material for this event"}
                    </p>
                    {fileSearch || fileCategory || scope !== "both" ? (
                      <button
                        onClick={() => {
                          setFileSearch("");
                          setFileCategory("");
                          setScope("both");
                        }}
                      >
                        Clear filters
                      </button>
                    ) : null}
                  </div>
                ) : null}
              </TableRegion>
            )}
            {pending && op?.kind === "upload" ? (
              <p>
                Saved records only. The pending file is not a downloadable
                catalog row.
              </p>
            ) : null}
            {cacheLabel(files.dataUpdatedAt)}
            <p className="mt-4">
              {offline
                ? "When online, uploads are visible to reps immediately. Upload is unavailable offline."
                : "Uploads are visible to reps immediately."}
            </p>
          </section>
        </>
      )}
      {editing ? (
        <EventForm
          event={typeof editing === "string" ? null : editing}
          selectedName={editing === "details" ? event : ""}
          offline={offline}
          locked={locked}
          onClose={() => setEditing(null)}
          onSaved={async (e) => {
            qc.setQueryData(
              q.plannerEvents().queryKey,
              (old: PlannerEvent[] | undefined) => [
                ...(old ?? []).filter((p) => p.id !== e.id),
                e,
              ],
            );
            setEditing(null);
            setSaved("Saved");
            await go(e.name);
            void events.refetch();
          }}
          onConflict={async (name) => {
            const result = await planned.refetch();
            if (result.data?.some((e) => e.name === name)) {
              setEditing(null);
              void go(name);
            } else setSaved("Refresh events to find the saved event.");
          }}
        />
      ) : null}
      <Modal
        open={!!dialog}
        onOpenChange={(open) => {
          if (!open) void closeSource();
        }}
        title={dialog === "import" ? "Import list" : "Add material"}
        footer={
          <>
            <Button onClick={() => void closeSource()}>Cancel</Button>
            <Button
              variant="primary"
              disabled={
                locked || unknown || correctionRequired || !!sourceWarning
              }
              onClick={() => void submit()}
            >
              {dialog === "import"
                ? op?.phase === "rejected"
                  ? "Retry import"
                  : "Import list"
                : op?.phase === "rejected"
                  ? "Retry upload"
                  : "Upload"}
            </Button>
          </>
        }
      >
        <div className="es-event-form">
          {op?.phase === "rejected" && !correctionRequired ? (
            <div role="status">
              <h3>Ready to try again</h3>
              <p>
                Review the corrected source and submit it again. Existing saved
                records will stay.
              </p>
            </div>
          ) : null}
          {dialog === "import" ? (
            <>
              <p>
                Import list for {event}. Saved records have not been removed.
              </p>
              <label>
                JSON file
                <input
                  type="file"
                  accept=".json,application/json"
                  disabled={unknown || offline}
                  onChange={async (e) => {
                    const f = e.target.files?.[0];
                    if (f) {
                      setJson(await f.text());
                      setSource(f.name);
                      setPreflight("");
                    }
                  }}
                />
              </label>
              <label>
                JSON
                <textarea
                  id="es-exhibitor-json"
                  rows={8}
                  readOnly={unknown}
                  value={json}
                  onChange={(e) => {
                    setJson(e.target.value);
                    setSource("Entered JSON");
                    setPreflight("");
                  }}
                />
              </label>
              {booths.length && !sourceWarning ? (
                <>
                  <p>
                    This may be a repeated list. Importing it again can add
                    duplicate exhibitors. Review the directory before
                    submitting.
                  </p>
                  <Button
                    onClick={() => {
                      setDialog(null);
                      requestAnimationFrame(() =>
                        document.getElementById("exhibitor-directory")?.focus(),
                      );
                    }}
                  >
                    Review directory
                  </Button>
                </>
              ) : null}
            </>
          ) : (
            <>
              <label>
                File
                <input
                  type="file"
                  id="es-material-file"
                  accept={ACCEPT}
                  disabled={unknown || offline}
                  onChange={(e) => {
                    setFile(e.target.files?.[0] ?? null);
                    setPreflight("");
                  }}
                />
              </label>
              {file ? (
                <p>
                  {file.name}, {fmtBytes(file.size)}
                </p>
              ) : null}
              <label>
                Category
                <select
                  aria-label="Category"
                  disabled={unknown}
                  value={meta.category}
                  onChange={(e) =>
                    setMeta({ ...meta, category: e.target.value })
                  }
                >
                  {CATEGORIES.map(([key, label]) => (
                    <option key={key} value={key}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Event
                <select
                  aria-label="Event"
                  disabled={unknown}
                  value={meta.event}
                  onChange={(e) => setMeta({ ...meta, event: e.target.value })}
                >
                  <option value="">All events</option>
                  {names.map((name) => (
                    <option key={name}>{name}</option>
                  ))}
                </select>
              </label>
              <p>
                Upload scope:{" "}
                {meta.event || "All events. Shared with every event."}
              </p>
              <label>
                Description
                <textarea
                  readOnly={unknown}
                  value={meta.description}
                  onChange={(e) =>
                    setMeta({ ...meta, description: e.target.value })
                  }
                />
              </label>
              <p>Uploads are visible to reps immediately.</p>
            </>
          )}
          <p>
            Duplicate checking is limited. Repeated uploads or imports may
            create additional records. Review saved records before submitting.
          </p>
          {offline ? (
            <p>
              You are offline. Your changes have not been saved. Keep this tab
              open and reconnect to review them.
            </p>
          ) : null}
          {sourceMessage ? (
            <div role="alert">
              {preflightTitle ? (
                <>
                  <h3>{preflightTitle}</h3>
                  <p>{sourceMessage.slice(preflightTitle.length + 2)}</p>
                  <Button
                    onClick={() =>
                      document
                        .getElementById(
                          dialog === "import"
                            ? "es-exhibitor-json"
                            : "es-material-file",
                        )
                        ?.focus()
                    }
                  >
                    {dialog === "import" ? "Edit JSON" : "Choose file"}
                  </Button>
                </>
              ) : (
                <p>{sourceMessage}</p>
              )}
              {sourceWarning ? (
                <Button
                  onClick={() => {
                    const kind = dialog;
                    setDialog(null);
                    requestAnimationFrame(() =>
                      document
                        .getElementById(
                          kind === "import"
                            ? "exhibitor-directory"
                            : "event-materials",
                        )
                        ?.focus(),
                    );
                  }}
                >
                  {repeatedSaved
                    ? dialog === "import"
                      ? "View directory"
                      : "View material"
                    : dialog === "import"
                      ? "Review directory"
                      : "Review material"}
                </Button>
              ) : null}
              {sameFilename ? (
                <Button
                  onClick={() =>
                    document.getElementById("es-material-file")?.click()
                  }
                >
                  Choose a different file
                </Button>
              ) : null}
            </div>
          ) : null}
        </div>
      </Modal>
      <Modal
        open={!!booth}
        onOpenChange={(open) => {
          if (!open && !boothBusy) setBooth(null);
        }}
        title={booth === "new" ? "Add exhibitor" : "Edit exhibitor"}
      >
        <form
          className="es-event-form"
          onSubmit={async (e) => {
            e.preventDefault();
            if (locked || boothBusy) return;
            setBoothBusy(true);
            setBoothError("");
            try {
              if (booth === "new")
                await api.addExhibitor({ ...boothForm, event });
              else if (booth) await api.updateExhibitor(booth.id, boothForm);
              setBooth(null);
              await detail.refetch();
            } catch {
              setBoothError(
                "Could not confirm this save. Refresh and review the directory before trying again.",
              );
            } finally {
              setBoothBusy(false);
            }
          }}
        >
          {Object.entries(boothForm).map(([key, value]) => (
            <label key={key}>
              {key.charAt(0).toUpperCase() + key.slice(1)}
              {key === "description" ? (
                <textarea
                  value={value}
                  onChange={(e) =>
                    setBoothForm({ ...boothForm, [key]: e.target.value })
                  }
                />
              ) : (
                <input
                  required={key === "name"}
                  type={key === "website" ? "url" : "text"}
                  value={value}
                  onChange={(e) =>
                    setBoothForm({ ...boothForm, [key]: e.target.value })
                  }
                />
              )}
            </label>
          ))}
          {boothError ? <p role="alert">{boothError}</p> : null}
          {offline ? (
            <p>
              You are offline. Your changes have not been saved. Keep this tab
              open and reconnect to review them.
            </p>
          ) : null}
          <Button type="submit" disabled={locked || boothBusy || !!boothError}>
            Save exhibitor
          </Button>
        </form>
      </Modal>
    </div>
  );
}
