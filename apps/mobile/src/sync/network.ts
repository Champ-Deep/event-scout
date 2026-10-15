import { Platform } from 'react-native';
import * as Network from 'expo-network';

/**
 * Real connectivity (expo-network). The API client used to assume native apps were always
 * online, so going offline only showed up as failed requests.
 */
let online = true;
const listeners = new Set<(online: boolean) => void>();

function apply(state: { isConnected?: boolean | null; isInternetReachable?: boolean | null }) {
  // isInternetReachable can be null while the OS is still checking — treat that as connected.
  const next = !!state.isConnected && state.isInternetReachable !== false;
  if (next !== online) {
    online = next;
    listeners.forEach((l) => l(next));
  }
}

if (Platform.OS !== 'web') {
  Network.getNetworkStateAsync().then(apply).catch(() => {});
  try {
    Network.addNetworkStateListener(apply);
  } catch {
    /* listener unavailable: getNetworkStateAsync is still used on demand */
  }
}

export async function isOnline(): Promise<boolean> {
  if (Platform.OS === 'web') return typeof navigator === 'undefined' || navigator.onLine !== false;
  try {
    apply(await Network.getNetworkStateAsync());
  } catch {
    /* keep the last known state */
  }
  return online;
}

export function isOnlineNow(): boolean {
  return online;
}

/** Called with true when the connection comes back, false when it drops. */
export function onConnectivityChange(listener: (online: boolean) => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}
