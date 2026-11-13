import React from "react";
import { Button, Spinner } from "../../ui/kit";

export type Operation = {
  kind: "import" | "upload";
  phase:
    | "pending"
    | "success"
    | "unknown"
    | "refreshing"
    | "refreshFailed"
    | "reviewed"
    | "savedStale"
    | "rejected";
  source: string;
  scope: string;
  category?: string;
  message?: string;
};
export function OperationNotice({
  operation: o,
  onRefresh,
  onReview,
  onClose,
}: {
  operation: Operation;
  onRefresh: () => void;
  onReview: () => void;
  onClose: () => void;
}) {
  const importing = o.kind === "import";
  const noun = importing ? "import" : "upload";
  const pending = o.phase === "pending";
  const title = pending
    ? importing
      ? "Importing exhibitor list..."
      : "Uploading material..."
    : o.phase === "success"
      ? importing
        ? "Exhibitor import complete"
        : "Material uploaded"
      : o.phase === "savedStale"
        ? "Saved, but the list did not refresh"
        : o.phase === "rejected"
          ? importing
            ? "Import not started"
            : "Upload not started"
          : o.phase === "refreshing"
            ? "Checking saved records..."
            : o.phase === "refreshFailed"
              ? "Could not refresh saved records"
              : o.phase === "reviewed"
                ? "Review saved records"
                : importing
                  ? "Import outcome unknown"
                  : "Upload outcome unknown";
  const body = pending
    ? `Waiting for confirmation. Keep this page open until the ${noun} finishes.`
    : o.phase === "savedStale"
      ? "The server confirmed this save. Refresh the list to see the latest records. Do not import or upload it again."
      : o.phase === "rejected"
        ? "The server rejected this request before saving. Review the details below and correct the source before trying again."
        : o.phase === "refreshing"
          ? "Reviewing the latest saved records. Nothing is being imported or uploaded."
          : o.phase === "refreshFailed"
            ? "The result is still unknown. Try refreshing again before resubmitting this source."
            : o.phase === "reviewed"
              ? "The list is refreshed, but this request's outcome is still unconfirmed. Compare the saved records with your source. Do not resend the full source while the outcome is unknown."
              : o.phase === "unknown"
                ? `We could not confirm whether ${importing ? "the list" : "the file"} was saved. Refresh and review ${importing ? "the directory" : "material"} before trying again. Saved records have not been removed.`
                : o.message;
  return (
    <div
      className="es-notice"
      role={
        ["unknown", "rejected", "refreshFailed", "savedStale"].includes(o.phase)
          ? "alert"
          : "status"
      }
    >
      <h3 className="flex items-center gap-2">
        {pending || o.phase === "refreshing" ? <Spinner /> : null}
        {title}
      </h3>
      <p>{body}</p>
      {!pending ? (
        <p>
          Source: {o.source} | Event: {o.scope}
        </p>
      ) : null}
      {o.phase === "rejected" && o.message ? <p>{o.message}</p> : null}
      {pending ? (
        <>
          <p>
            {importing
              ? `Pending request: ${o.source} for ${o.scope}. No row results yet.`
              : `Pending file: ${o.source} | Category: ${o.category} | Event: ${o.scope}`}
          </p>
          {!importing ? <p>Uploads are visible to reps immediately.</p> : null}
          <div className="es-actions mt-3">
            <Button disabled>
              {importing ? "Importing..." : "Uploading..."}
            </Button>
            <Button disabled>Cancel</Button>
          </div>
          <p>
            This request has already been sent and cannot be cancelled here.
          </p>
          <p>
            Keep this page open until the request finishes. Leaving does not
            cancel it.
          </p>
        </>
      ) : (
        <div className="es-actions mt-3">
          {["unknown", "refreshFailed", "savedStale"].includes(o.phase) ? (
            <Button onClick={onRefresh}>
              {o.phase === "savedStale"
                ? "Refresh list"
                : o.phase === "refreshFailed"
                  ? "Refresh again"
                  : importing
                    ? "Refresh directory"
                    : "Refresh material"}
            </Button>
          ) : null}
          {o.phase !== "refreshing" ? (
            <Button onClick={onReview}>
              {o.phase === "success"
                ? importing
                  ? "View directory"
                  : "View material"
                : "Review source"}
            </Button>
          ) : null}
          {["success", "reviewed"].includes(o.phase) ? (
            <Button onClick={onClose}>Close</Button>
          ) : null}
        </div>
      )}
    </div>
  );
}
