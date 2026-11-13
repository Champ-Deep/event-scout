import React, { useRef, useState } from "react";
import { useBlocker } from "@tanstack/react-router";
import { ApiError, api } from "../../lib/api";
import type {
  PlannerEvent,
  PlannerEventInput,
  PlannerStatus,
} from "../../lib/types";
import { Button, Modal, useUi } from "../../ui/kit";

export const EVENT_FIELDS = [
  ["name", "Event name", 255],
  ["venue", "Venue", 255],
  ["organizer", "Organizer", 255],
  ["address", "Address", 500],
  ["city", "City", 120],
  ["country", "Country", 120],
  ["industry", "Industry", 255],
] as const;
export const DETAIL_FIELDS = [
  ["booth", "Our booth"],
  ["hours", "Opening hours"],
  ["registration", "Registration"],
  ["tickets", "Tickets"],
  ["dressCode", "Dress code"],
  ["contactName", "Contact name"],
  ["contactEmail", "Contact email"],
  ["contactPhone", "Contact phone"],
  ["notes", "Notes"],
] as const;
const STATUSES: PlannerStatus[] = [
  "suggested",
  "interested",
  "planned",
  "attending",
  "done",
  "dismissed",
];
const EMPTY: PlannerEventInput = {
  name: "",
  status: "planned",
  website: "",
  startDate: "",
  endDate: "",
  details: {},
};

export function EventForm({
  event,
  selectedName = "",
  locked,
  offline,
  onClose,
  onSaved,
  onConflict,
}: {
  event: PlannerEvent | null;
  selectedName?: string;
  locked: boolean;
  offline: boolean;
  onClose: () => void;
  onSaved: (event: PlannerEvent) => Promise<void>;
  onConflict: (name: string) => void;
}) {
  const { confirm } = useUi();
  const initialValue = event ?? { ...EMPTY, name: selectedName };
  const [value, setValue] = useState<PlannerEventInput>(initialValue);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [conflict, setConflict] = useState(false);
  const [uncertain, setUncertain] = useState(false);
  const inflight = useRef(false);
  const committed = useRef(false);
  const initial = JSON.stringify(initialValue);
  const dirty = JSON.stringify(value) !== initial;
  useBlocker({
    shouldBlockFn: async () =>
      dirty && !committed.current
        ? !(await confirm({
            title: "Discard changes?",
            message: "Your changes have not been saved.",
            confirmLabel: "Discard",
            cancelLabel: "Keep editing",
          }))
        : false,
    enableBeforeUnload: dirty || busy,
  });
  const set = (patch: PlannerEventInput) => {
    setValue((v) => ({ ...v, ...patch }));
    setError("");
    setConflict(false);
  };
  const close = async () => {
    if (busy) return;
    if (
      dirty &&
      !(await confirm({
        title: "Discard changes?",
        message: "Your changes have not been saved.",
        confirmLabel: "Discard",
        cancelLabel: "Keep editing",
      }))
    )
      return;
    onClose();
  };
  const save = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (locked || inflight.current || uncertain) return;
    const form = e.currentTarget;
    const name = form.elements.namedItem("name") as HTMLInputElement;
    name.setCustomValidity(!value.name?.trim() ? "Enter an event name." : "");
    if (!form.reportValidity()) return;
    inflight.current = true;
    setBusy(true);
    setError("");
    try {
      const {
        name: eventName,
        status,
        website,
        startDate,
        endDate,
        venue,
        address,
        city,
        country,
        organizer,
        industry,
        description,
        details,
      } = value;
      const body: PlannerEventInput = {
        name: eventName?.trim() ?? "",
        status: status ?? "planned",
        website: website?.trim() ?? "",
        startDate: startDate || null,
        endDate: endDate || null,
        venue: venue ?? "",
        address: address ?? "",
        city: city ?? "",
        country: country ?? "",
        organizer: organizer ?? "",
        industry: industry ?? "",
        description: description ?? "",
        details: details ?? {},
      };
      const saved = event
        ? await api.updatePlannerEvent(event.id, body)
        : await api.createPlannerEvent(body);
      committed.current = true;
      await onSaved(saved);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409 && !event) {
        setConflict(true);
        setError(
          `An event named ${value.name?.trim()} already exists. Open it instead of creating another event.`,
        );
      } else if (err instanceof ApiError && err.status === 0) {
        setUncertain(true);
        setError(
          "Connection lost. We could not confirm whether this change was saved. Reconnect to check before trying again.",
        );
      } else
        setError(
          "Could not save the event. Review your details and try again.",
        );
    } finally {
      inflight.current = false;
      setBusy(false);
    }
  };
  return (
    <Modal
      open
      onOpenChange={(open) => {
        if (!open) void close();
      }}
      title={event || selectedName ? "Edit event" : "Add event"}
      size="lg"
      footer={
        <>
          <Button disabled={busy} onClick={() => void close()}>
            Cancel
          </Button>
          <Button
            form="es-event-form"
            type="submit"
            variant="primary"
            disabled={locked || uncertain || busy}
            loading={busy}
          >
            Save event
          </Button>
        </>
      }
    >
      <form
        id="es-event-form"
        className="es-event-form"
        onSubmit={save}
        onInvalid={(e) => {
          const details = (e.target as HTMLElement).closest("details");
          if (details) details.open = true;
        }}
      >
        <h2>Event website</h2>
        <label>
          Event website
          <input
            type="url"
            pattern="https?://.*"
            maxLength={500}
            value={value.website ?? ""}
            onChange={(e) => set({ website: e.target.value })}
          />
        </label>
        <p>Scraper not connected. Enter the details below.</p>
        <h2>Event info</h2>
        <div className="es-event-fields">
          <label>
            Event name *
            <input
              name="name"
              required
              readOnly={!!event || !!selectedName}
              maxLength={255}
              value={value.name ?? ""}
              onChange={(e) => {
                e.target.setCustomValidity("");
                set({ name: e.target.value });
              }}
            />
          </label>
          <label>
            Status
            <select
              aria-label="Status"
              value={value.status ?? "planned"}
              onChange={(e) => set({ status: e.target.value as PlannerStatus })}
            >
              {STATUSES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label>
            Start date
            <input
              name="startDate"
              type="date"
              value={value.startDate ?? ""}
              onChange={(e) => set({ startDate: e.target.value })}
            />
          </label>
          <label>
            End date
            <input
              name="endDate"
              type="date"
              min={value.startDate || undefined}
              value={value.endDate ?? ""}
              onChange={(e) => {
                e.target.setCustomValidity("");
                set({ endDate: e.target.value });
              }}
            />
          </label>
          {EVENT_FIELDS.filter(([key]) => key !== "name").map(
            ([key, label, max]) => (
              <label
                key={key}
                className={
                  key === "address" || key === "industry"
                    ? "es-wide"
                    : undefined
                }
              >
                {label}
                <input
                  name={key}
                  maxLength={max}
                  value={value[key] ?? ""}
                  onChange={(e) => set({ [key]: e.target.value })}
                />
              </label>
            ),
          )}
          <label className="es-wide">
            Description
            <textarea
              rows={3}
              value={value.description ?? ""}
              onChange={(e) => set({ description: e.target.value })}
            />
          </label>
        </div>
        <details>
          <summary>Logistics and organiser contact</summary>
          <div className="es-event-fields">
            {DETAIL_FIELDS.map(([key, label]) => (
              <label key={key}>
                {label}
                {key === "notes" ? (
                  <textarea
                    value={value.details?.[key] ?? ""}
                    onChange={(e) =>
                      set({
                        details: { ...value.details, [key]: e.target.value },
                      })
                    }
                  />
                ) : (
                  <input
                    type={
                      key === "contactEmail"
                        ? "email"
                        : key === "contactPhone"
                          ? "tel"
                          : "text"
                    }
                    value={value.details?.[key] ?? ""}
                    onChange={(e) =>
                      set({
                        details: { ...value.details, [key]: e.target.value },
                      })
                    }
                  />
                )}
              </label>
            ))}
          </div>
        </details>
        {offline ? (
          <p role="status">
            You are offline. Your changes have not been saved. Keep this tab
            open and reconnect to review them.
          </p>
        ) : null}
        {error ? (
          <div role="alert">
            <h3>{conflict ? "Event already exists" : "Event not saved"}</h3>
            <p>{error}</p>
            {conflict ? (
              <Button onClick={() => onConflict(value.name?.trim() ?? "")}>
                Open existing event
              </Button>
            ) : null}
            {uncertain && !locked ? (
              <Button onClick={() => onConflict(value.name?.trim() ?? "")}>
                Review saved event
              </Button>
            ) : null}
          </div>
        ) : null}
      </form>
    </Modal>
  );
}
