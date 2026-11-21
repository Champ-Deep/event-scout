import { useEffect, useState } from "react";
import { Download } from "lucide-react";
import { toast } from "sonner";
import { type Api, type Contact, downloadBlob, message } from "@/api/client";
import {
  Button,
  ContactRows,
  Field,
  Notice,
  Loading,
} from "@/components/shared";
import { ChipGroup } from "@/components/ui/chip-group";
const filters = [
  { value: "all", label: "All" },
  { value: "today", label: "Today" },
  { value: "yesterday", label: "Yesterday" },
  { value: "last_week", label: "Last week" },
  { value: "last_month", label: "Last month" },
];
export function Contacts({
  api,
  onOpen,
  version,
  onScan,
}: {
  api: Api;
  onOpen: (id: string) => void;
  version: number;
  onScan: () => void;
}) {
  const [contacts, setContacts] = useState<Contact[] | null>(null);
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [exporting, setExporting] = useState(false);
  useEffect(() => {
    let current = true;
    setContacts(null);
    setError("");
    api
      .json<{ contacts: Contact[] }>(
        `/list_contacts/?${api.userQuery}${filter === "all" ? "" : `&date_filter=${filter}`}`,
      )
      .then((r) => {
        if (current) setContacts(r.contacts);
      })
      .catch((e) => {
        if (current) setError(message(e));
      });
    return () => {
      current = false;
    };
  }, [api, filter, version, retry]);
  async function exportCSV() {
    setExporting(true);
    try {
      const response = await api.request(
        `/export_contacts/?${api.userQuery}&format=csv`,
      );
      downloadBlob(await response.blob(), "event-scout-contacts.csv");
      toast.success("CSV downloaded");
    } catch (e) {
      setError(message(e));
    } finally {
      setExporting(false);
    }
  }
  const shown = contacts?.filter((c) =>
    [c.name, c.email, c.phone, c.company_name, c.title, c.notes].some((v) =>
      v?.toLowerCase().includes(search.toLowerCase()),
    ),
  );
  return (
    <>
      <header className="page-header">
        <div>
          <h1>Contacts</h1>
          <p className="muted" role="status">
            {contacts === null
              ? "Loading contacts..."
              : `${shown?.length} ${search ? "matching " : ""}${shown?.length === 1 ? "contact" : "contacts"}`}
          </p>
        </div>
        <Button variant="outline" onClick={exportCSV} disabled={exporting}>
          <Download />
          {exporting ? "Exporting..." : "Export CSV"}
        </Button>
      </header>
      <div className="contacts-tools">
        <Field
          label="Search contacts"
          type="search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Name, company or notes"
        />
        <ChipGroup
          label="Date added"
          options={filters}
          value={[filter]}
          multiple={false}
          onValueChange={(v) => setFilter(v[0] || "all")}
        />
      </div>
      {error && <Notice error={error} retry={() => setRetry((v) => v + 1)} />}{" "}
      {!contacts && !error && <Loading>Loading your contacts...</Loading>}
      {shown && shown.length > 0 && (
        <ContactRows contacts={shown} onOpen={onOpen} />
      )}{" "}
      {shown?.length === 0 && (
        <section className="empty">
          <h2>
            {search
              ? "No matching contacts"
              : filter === "all"
                ? "Scan your first card"
                : "No contacts in this date range"}
          </h2>
          <p>
            {search
              ? "Try a different name or company."
              : filter === "all"
                ? "Keep the people you meet in one place."
                : "Choose All to see your other contacts."}
          </p>
          {!search && filter === "all" && (
            <Button onClick={onScan}>Scan a card</Button>
          )}
        </section>
      )}
    </>
  );
}
