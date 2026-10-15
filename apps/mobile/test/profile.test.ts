import { describe, expect, it, vi } from 'vitest';
import { createProfileStore, eventRules, type Profile } from '../src/features/profile/core';

function setup(server: Profile) {
  let copy: string | undefined = 'Old Expo';
  const update = vi.fn(async (_userId: string, patch: any) => ({ status: 'success', profile: { ...server, ...patch } }));
  const store = createProfileStore({
    remote: { get: async () => ({ profile: server }), update },
    activeEvent: { get: () => copy, set: (name) => { copy = name; } },
  });
  return { store, update, copy: () => copy };
}

describe('event-list rules', () => {
  it('activating an event adds it to your events and reopens it if it was ended', () => {
    const profile: Profile = { events: ['A'], ended_events: ['b'], current_event_name: 'A' };
    expect(eventRules.activate(profile, 'B')).toEqual({ current_event_name: 'B', events: ['A', 'B'], ended_events: [] });
    // already listed (any case), not ended: only the active event changes
    expect(eventRules.activate(profile, 'A')).toEqual({ current_event_name: 'A', events: ['A'] });
    expect(eventRules.activate({ events: ['expo'] }, 'Expo').events).toEqual(['expo']);
  });

  it('adding the first event makes it the active one', () => {
    expect(eventRules.add({ events: [] }, 'Expo')).toEqual({ current_event_name: 'Expo', events: ['Expo'] });
    expect(eventRules.add({ events: ['A'], current_event_name: 'A' }, 'B')).toEqual({ events: ['A', 'B'] });
    expect(eventRules.add({ events: ['A'], current_event_name: 'A' }, 'B', { makeCurrent: true }).current_event_name).toBe('B');
  });

  it('ending the active event clears it; reopening takes it off the ended list', () => {
    const profile: Profile = { events: ['A'], current_event_name: 'A', ended_events: [] };
    expect(eventRules.toggleEnded(profile, 'A')).toEqual({ ended_events: ['A'], current_event_name: '' });
    expect(eventRules.toggleEnded({ ended_events: ['a'] }, 'A')).toEqual({ ended_events: [] });
    expect(eventRules.toggleEnded({ current_event_name: 'B' }, 'A')).toEqual({ ended_events: ['A'] });
  });

  it('ensureListed adds an event only when it is missing', () => {
    expect(eventRules.ensureListed({ events: ['Expo'] }, 'expo')).toEqual({});
    expect(eventRules.ensureListed({ events: ['A'] }, 'B')).toEqual({ events: ['A', 'B'] });
  });
});

describe('profile store', () => {
  it('activateEvent saves the same rule the Events page uses and updates the offline copy', async () => {
    const server: Profile = { events: ['A'], ended_events: ['B'], current_event_name: 'A' };
    const { store, update, copy } = setup(server);

    const saved = await store.activateEvent('u1', server, 'B');
    expect(update).toHaveBeenCalledWith('u1', { current_event_name: 'B', events: ['A', 'B'], ended_events: [] });
    expect(saved.current_event_name).toBe('B');
    expect(copy()).toBe('B'); // offline scans are now tagged with B
  });

  it('loading the profile refreshes the offline copy of the active event', async () => {
    const { store, copy } = setup({ current_event_name: 'Expo 2026' });
    expect(await store.load('u1')).toEqual({ current_event_name: 'Expo 2026' });
    expect(copy()).toBe('Expo 2026');
    expect(store.activeEventName()).toBe('Expo 2026');
    expect(store.peek('u1')?.current_event_name).toBe('Expo 2026');
    expect(store.peek('someone-else')).toBeNull();
  });

  it('clearing the active event clears the offline copy too', async () => {
    const { store, copy } = setup({ current_event_name: 'A' });
    await store.clearActiveEvent('u1');
    expect(copy()).toBe('');
    expect(store.activeEventName()).toBeNull();
  });

  it('a change that does not touch the active event leaves the offline copy alone', async () => {
    const remote = { get: async () => ({}), update: async () => ({ status: 'success', profile: { job_title: 'CTO' } }) };
    let copy = 'Expo';
    const store = createProfileStore({ remote, activeEvent: { get: () => copy, set: (n) => { copy = n; } } });
    const saved = await store.save('u1', { job_title: 'CTO' });
    expect(saved).toEqual({ job_title: 'CTO' });
    expect(copy).toBe('Expo');
  });

  it('refuses to save without an account', async () => {
    const { store } = setup({});
    await expect(store.save('', { job_title: 'x' })).rejects.toThrow('Sign in again to save.');
  });
});
