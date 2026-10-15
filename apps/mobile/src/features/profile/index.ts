import { api } from '../../lib/api';
import { fastStore } from '../../lib/mmkv';
import { createProfileStore } from './core';

export {
  eventRules, createProfileStore,
  type Profile, type ProfilePatch, type Product, type EventDetailEntry, type ProfileStore,
} from './core';

/**
 * The profile & active event module (see ./core): the only way screens load / save the profile.
 * Every load and save keeps the phone's copy of the active event (fastStore ACTIVE_EVENT_NAME)
 * current, so scans saved offline are tagged with the event that's really active.
 */
export const profileStore = createProfileStore({
  remote: {
    get: (userId) => api.getProfile(userId),
    update: (userId, patch) => api.updateProfile(userId, patch),
  },
  activeEvent: {
    get: () => fastStore.getActiveEventName(),
    set: (name) => fastStore.setActiveEventName(name),
  },
});
