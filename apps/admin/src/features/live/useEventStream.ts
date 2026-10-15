import { useEffect, useState, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { subscribeLiveEventStream, LiveEventMessage, StreamStatus } from '../../lib/sse';
import { liveKeys } from '../../lib/changes';

// Which page data each live event moves: lib/changes (the same map every write in the console uses).

export function useEventStream(streamUrl?: string) {
  const queryClient = useQueryClient();
  const [recentEvents, setRecentEvents] = useState<LiveEventMessage[]>([]);
  const [status, setStatus] = useState<StreamStatus>('connecting');
  const queryClientRef = useRef(queryClient);

  useEffect(() => {
    queryClientRef.current = queryClient;
  }, [queryClient]);

  useEffect(() => {
    if (!streamUrl) return;
    let connectedBefore = false;
    const cleanup = subscribeLiveEventStream(streamUrl, (msg) => {
      if (msg.type === 'connected') {
        // events sent while the stream was down are not replayed: after a reconnect refetch everything
        if (connectedBefore) queryClientRef.current.invalidateQueries({ queryKey: ['admin'] });
        connectedBefore = true;
        return;
      }
      setRecentEvents((prev) => [msg, ...prev].slice(0, 20));
      // an event this console doesn't know yet refreshes the feed only (and says so in dev)
      liveKeys(msg.type).forEach((queryKey) => {
        queryClientRef.current.invalidateQueries({ queryKey });
      });
    }, setStatus);
    return cleanup;
  }, [streamUrl]);

  return { recentEvents, status };
}
