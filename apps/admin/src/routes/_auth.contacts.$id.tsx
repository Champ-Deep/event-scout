import React, { useMemo } from 'react';
import { useSuspenseQuery } from '@tanstack/react-query';
import { useNavigate, useParams, useSearch } from '@tanstack/react-router';
import { q } from '../lib/queries';
import { clean, type ContactTab, type ContactsSearch } from '../lib/search';
import { ContactDrawer } from '../features/contacts/ContactDrawer';

/** /contacts/$id — the row drawer over the grid. Its loader ensured the contact; this reads the cache. ?tab= picks the tab. */
export function ContactDrawerRoute() {
  const { id } = useParams({ from: '/_auth/contacts/$id' });
  const search = useSearch({ from: '/_auth/contacts/$id' });
  const navigate = useNavigate();
  const { data: contact } = useSuspenseQuery(q.contact(id));
  const { data: users } = useSuspenseQuery(q.users());
  const { data: events } = useSuspenseQuery(q.events());
  const { data: dash } = useSuspenseQuery(q.dashboard(7));
  const ownerUser = users.find((u) => u.id === contact.userId);
  const owner = ownerUser?.name;
  const eventNames = useMemo(() => [...new Set([...events.map((e) => e.name), ...dash.byEvent.map((e) => e.event), contact.eventId])].filter(Boolean).sort(), [events, dash, contact.eventId]);

  const patch = (p: Partial<ContactsSearch>) => clean({ ...search, ...p }) as ContactsSearch;
  const setTab = (tab: ContactTab) => navigate({ to: '/contacts/$id', params: { id }, search: patch({ tab: tab === 'overview' ? undefined : tab }), replace: true });
  // closing keeps every filter (search params) of the grid, but not the tab or full-page mode
  const close = () => navigate({ to: '/contacts', search: patch({ tab: undefined, full: undefined }) });
  const openOther = (other: string) => navigate({ to: '/contacts/$id', params: { id: other }, search: patch({ tab: undefined }) });
  const filter = (f: { event?: string; owner?: string }) => navigate({ to: '/contacts/$id', params: { id }, search: patch(f), replace: true });
  return (
    <ContactDrawer contact={contact} ownerName={owner} ownerPhoto={ownerUser?.photoUrl} ownerTitle={ownerUser?.title} tab={search.tab ?? 'overview'} onTab={setTab} onClose={close}
      onOpenContact={openOther} onFilter={filter} eventNames={eventNames} />
  );
}
