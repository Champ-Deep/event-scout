import React, { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../../lib/api';
import { applyChange } from '../../lib/changes';
import { Button, ErrorState, Field, Input, Modal, Select, Skeleton, errorMessage, useUi } from '../../ui/kit';

/** Add the given contacts to an existing list, or to a new one. */
export function AddToListModal({ contactIds, onClose, onDone }: { contactIds: string[] | null; onClose: () => void; onDone?: () => void }) {
  const qc = useQueryClient();
  const { toast } = useUi();
  const lists = useQuery({ queryKey: ['admin', 'lists'], queryFn: api.lists, enabled: !!contactIds });
  const [target, setTarget] = useState('');
  const [newName, setNewName] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => { if (contactIds) { setTarget(''); setNewName(''); } }, [contactIds]);
  useEffect(() => { if (!target && lists.data?.length) setTarget(lists.data[0]!.id); }, [lists.data]);
  const loadFailed = !!lists.error && !lists.data;
  const creating = target === '__new' || !lists.data?.length;

  const save = async () => {
    if (!contactIds) return;
    setBusy(true);
    try {
      let id = target;
      let name = lists.data?.find((l) => l.id === target)?.name || '';
      if (creating) {
        const l = await api.createList(newName);
        id = l.id;
        name = l.name;
      }
      const r = await api.addToList(id, contactIds);
      toast('success', `Added ${r.count} to “${name}”`, r.alreadyInList ? `${r.alreadyInList} were already in it` : undefined);
      void applyChange(qc, 'list_changed');
      onDone?.();
      onClose();
    } catch (e) {
      toast('error', "Couldn't add to the list", errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const n = contactIds?.length || 0;
  return (
    <Modal open={!!contactIds} onOpenChange={(o) => { if (!o) onClose(); }} size="sm" title={`Add ${n} contact${n === 1 ? '' : 's'} to a list`}
      footer={<>
        <Button variant="ghost" onClick={onClose}>Cancel</Button>
        <Button variant="primary" loading={busy} disabled={loadFailed || (creating ? !newName.trim() : !target)} onClick={save}>Add to list</Button>
      </>}>
      {lists.isLoading ? <Skeleton className="h-9 w-full" /> : loadFailed ? (
        // a failed load must not look like "no lists yet" (which would offer to create a new one)
        <ErrorState error={lists.error} onRetry={() => lists.refetch()} />
      ) : (
        <div className="space-y-4">
          {lists.data?.length ? (
            <Field label="List">
              <Select value={target} onChange={(e) => setTarget(e.target.value)}>
                {lists.data.map((l) => <option key={l.id} value={l.id}>{l.name} ({l.memberCount})</option>)}
                <option value="__new">+ New list…</option>
              </Select>
            </Field>
          ) : null}
          {creating ? (
            <Field label="New list name"><Input autoFocus value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="e.g. Day 1 hot leads" /></Field>
          ) : null}
        </div>
      )}
    </Modal>
  );
}
