import React, { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient, useSuspenseQuery } from '@tanstack/react-query';
import { useNavigate, useSearch } from '@tanstack/react-router';
import { api } from '../lib/api';
import { q } from '../lib/queries';
import { applyChange } from '../lib/changes';
import { clean } from '../lib/search';
import type { WorkspaceDeletedContact } from '../lib/types';
import { DeletedContactDrawer } from '../features/contacts/DeletedContactDrawer';
import { FilterPill, SearchField, type PillOption } from '../features/contacts/Toolbar';
import { Avatar } from '../ui/Avatar';
import { Icon } from '../ui/icons';
import { Badge, Button, Card, EmptyState, PageHeader, Table, Td, Th, cx, errorMessage, fmtDateTime, relTime, useUi } from '../ui/kit';

type Page = { contacts: WorkspaceDeletedContact[]; retentionDays: number };

/** Every rep's Recently deleted — contacts deleted or merged away, restorable for the server's retentionDays. */
export function DeletedComponent() {
  const qc = useQueryClient();
  const { toast } = useUi();
  const { data } = useSuspenseQuery(q.deleted());
  // search + owner filter live in the URL (?q=…&owner=…)
  const search = useSearch({ from: '/_auth/deleted' });
  const navigate = useNavigate();
  const owner = search.owner ?? '';
  // the contact being looked at (?view=<id>), opened from its name
  const viewing = search.view ? data.contacts.find((c) => c.id === search.view) ?? null : null;
  const setView = (id: string | undefined) => navigate({ to: '/deleted', search: (s) => clean({ ...s, view: id }) });
  const setOwner = (o: string) => navigate({ to: '/deleted', search: (s) => clean({ ...s, owner: o || undefined }), replace: true });
  const [text, setText] = useState(search.q ?? '');
  useEffect(() => {
    const id = setTimeout(() => {
      if ((search.q ?? '') !== text.trim()) navigate({ to: '/deleted', search: (s) => clean({ ...s, q: text.trim() || undefined }), replace: true });
    }, 300);
    return () => clearTimeout(id);
  }, [text]);

  // owner filter options: every rep with something in the bin, their photo and how many items they have there
  const { data: users } = useQuery(q.users());
  const ownerOptions: PillOption[] = useMemo(() => {
    const counts = new Map<string, { name: string; n: number }>();
    for (const c of data.contacts) counts.set(c.ownerId, { name: c.owner, n: (counts.get(c.ownerId)?.n ?? 0) + 1 });
    return [...counts.entries()].sort((a, b) => a[1].name.localeCompare(b[1].name)).map(([id, { name, n }]) => ({
      value: id, label: name, count: n,
      icon: <Avatar src={users?.find((u) => u.id === id)?.photoUrl || null} name={name} size="xs" className="size-6 shrink-0 border-0 text-[9px] shadow-none" />,
    }));
  }, [data, users]);
  const shown = useMemo(() => {
    const t = (search.q ?? '').toLowerCase();
    return data.contacts.filter((c) => (!owner || c.ownerId === owner)
      && (!t || [c.name, c.company, c.title, c.owner, c.reason].join(' ').toLowerCase().includes(t)));
  }, [data, owner, search.q]);

  // optimistic: the row leaves at once; it comes back if the restore fails
  const restore = async (c: WorkspaceDeletedContact) => {
    if (search.view === c.id) setView(undefined);
    await qc.cancelQueries({ queryKey: q.deleted().queryKey });
    const before = qc.getQueryData<Page>(q.deleted().queryKey);
    qc.setQueryData<Page>(q.deleted().queryKey, (d) => (d ? { ...d, contacts: d.contacts.filter((x) => x.id !== c.id) } : d));
    try {
      await api.restoreDeleted(c.id, c.ownerId);
      toast('success', `${c.name} restored`, `Back in ${c.owner}'s contacts.`);
    } catch (e) {
      qc.setQueryData(q.deleted().queryKey, before);
      toast('error', "Couldn't restore — put back", errorMessage(e));
    } finally {
      void applyChange(qc, 'contact_restored');
    }
  };

  return (
    <>
      <PageHeader title="Recently deleted"
        description={`Contacts your team deleted or merged away. Each one can be restored to its owner for ${data.retentionDays} days, then it's removed for good.`} />
      <Card>
        <div className="flex flex-wrap items-center gap-3 border-b border-border p-4">
          <SearchField value={text} onChange={setText} placeholder="Search name, company, owner, reason…" className="w-full sm:w-[340px]" />
          <FilterPill icon="users" tint="var(--brand-600)" title="Owner" noun="rep" options={ownerOptions}
            value={owner || undefined} onChange={(v) => setOwner(v ?? '')} />
          {text || owner ? (
            <button onClick={() => { setText(''); setOwner(''); }}
              className="inline-flex h-9 shrink-0 cursor-pointer items-center gap-1 rounded-full px-2.5 text-xs font-bold text-brand-700 transition-colors hover:bg-brand-50">
              <Icon name="eraser" size={13} />Clear all
            </button>
          ) : null}
          <span className="ml-auto rounded-full bg-secondary px-2.5 py-1 text-xs font-bold tabular-nums text-body">{shown.length} of {data.contacts.length}</span>
        </div>
        {!shown.length ? (
          <EmptyState icon="trash" title={data.contacts.length ? 'Nothing matches' : 'Nothing deleted recently'}
            description={data.contacts.length ? undefined : `Deleted and merged contacts appear here for ${data.retentionDays} days.`} />
        ) : (
          <Table>
            <thead><tr><Th>Contact</Th><Th>Owner</Th><Th>What happened</Th><Th>When</Th><Th>Kept for</Th><Th className="w-28" /></tr></thead>
            <tbody>
              {shown.map((c) => (
                <tr key={c.id} className={cx('cursor-pointer hover:bg-secondary', viewing?.id === c.id && 'bg-accent hover:bg-accent')} onClick={() => setView(c.id)}>
                  <Td>
                    <div className="flex items-center gap-3">
                      <Avatar src={c.photoUrl} name={c.name} size="sm" />
                      <div className="min-w-0">
                        <button onClick={(e) => { e.stopPropagation(); setView(c.id); }} title={`See ${c.name}'s details`}
                          className="group/name inline-flex max-w-full cursor-pointer items-center gap-1 truncate text-left font-semibold text-foreground underline-offset-2 hover:text-brand-700 hover:underline">
                          <span className="truncate">{c.name}</span>
                          <Icon name="chevronRight" size={13} className="shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover/name:opacity-100" />
                        </button>
                        <div className="truncate text-xs text-muted-foreground">{[c.title, c.company].filter(Boolean).join(' · ') || '—'}</div>
                      </div>
                    </div>
                  </Td>
                  <Td className="text-body">{c.owner}</Td>
                  <Td>{c.reason ? <Badge tone="violet" icon="merge">{c.reason}</Badge> : <Badge tone="red" icon="trash">Deleted</Badge>}</Td>
                  <Td className="whitespace-nowrap text-muted-foreground" title={fmtDateTime(c.deletedAt)}>{relTime(c.deletedAt)}</Td>
                  <Td className={cx('whitespace-nowrap tabular-nums', c.daysLeft <= 3 ? 'font-medium text-rose-600 dark:text-rose-400' : 'text-muted-foreground')}>
                    {c.daysLeft} day{c.daysLeft === 1 ? '' : 's'}
                  </Td>
                  <Td><Button size="sm" icon="refresh" onClick={(e) => { e.stopPropagation(); restore(c); }}>Restore</Button></Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
      <DeletedContactDrawer summary={viewing} onClose={() => setView(undefined)} onRestore={restore} />
    </>
  );
}
