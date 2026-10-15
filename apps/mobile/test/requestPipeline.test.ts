import { describe, expect, it, vi } from 'vitest';
import { SESSION_REVOKED_DETAIL } from '../src/lib/apiError';
import { createRequestPipeline, type PipelineDeps } from '../src/lib/requestPipeline';

function setup(over: Partial<PipelineDeps> = {}) {
  const tokens = { access: 'old' as string | null, refresh: 'r1' as string | null };
  const postRefresh = vi.fn(async (rt: string) => ({ token: `new-from-${rt}`, refresh_token: 'r2' }));
  const deps: PipelineDeps = {
    getToken: async () => tokens.access,
    getRefreshToken: async () => tokens.refresh,
    saveTokens: async (a, r) => { tokens.access = a; tokens.refresh = r; },
    postRefresh,
    isTokenExpired: (t) => t === 'old',
    apiKey: () => 'KEY',
    ...over,
  };
  const pipeline = createRequestPipeline(deps);
  const revoked = vi.fn();
  pipeline.setSessionRevokedHandler(revoked);
  return { pipeline, tokens, postRefresh, revoked };
}

describe('request pipeline', () => {
  it('adds the bearer token and the API key (not on sign-in routes)', async () => {
    const { pipeline } = setup();
    expect(await pipeline.headers('/api/contacts')).toEqual({ Authorization: 'Bearer old', 'X-API-Key': 'KEY' });
    expect(await pipeline.headers('/login/')).toEqual({ Authorization: 'Bearer old' });
  });

  it('streams get a token renewed first when it already expired', async () => {
    const { pipeline, postRefresh } = setup();
    expect((await pipeline.headers('/converse/stream/', { fresh: true })).Authorization).toBe('Bearer new-from-r1');
    expect(postRefresh).toHaveBeenCalledTimes(1);
  });

  it('a 401 renews the token once, shared by requests that failed together', async () => {
    const { pipeline, postRefresh } = setup();
    const [a, b] = await Promise.all([
      pipeline.recoverFrom401({ url: '/api/broadcasts/stream', detail: 'Token expired', sentToken: 'old' }),
      pipeline.recoverFrom401({ url: '/converse/stream/', detail: 'Token expired', sentToken: 'old' }),
    ]);
    expect(a).toBe('new-from-r1');
    expect(b).toBe('new-from-r1');
    expect(postRefresh).toHaveBeenCalledTimes(1);
    // a request still carrying the old token replays with the one already renewed
    expect(await pipeline.recoverFrom401({ url: '/api/contacts', detail: 'Token expired', sentToken: 'old' })).toBe('new-from-r1');
    expect(postRefresh).toHaveBeenCalledTimes(1);
  });

  it('a revoked session is never renewed: it is reported once', async () => {
    const { pipeline, postRefresh, revoked } = setup();
    expect(await pipeline.recoverFrom401({ url: '/converse/stream/', detail: SESSION_REVOKED_DETAIL, sentToken: 'old' })).toBeNull();
    expect(pipeline.checkRevoked({ url: '/api/contacts', status: 401, detail: SESSION_REVOKED_DETAIL })).toBe(true);
    expect(postRefresh).not.toHaveBeenCalled();
    expect(revoked).toHaveBeenCalledTimes(1);
    pipeline.resetSessionRevoked();
    pipeline.checkRevoked({ url: '/auth/refresh', status: 401, detail: 'Invalid refresh token' });
    expect(revoked).toHaveBeenCalledTimes(2);
  });

  it('sign-in routes and a missing refresh token are not renewed', async () => {
    const { pipeline, tokens } = setup();
    expect(await pipeline.recoverFrom401({ url: '/auth/refresh', detail: 'x', sentToken: 'old' })).toBeNull();
    tokens.refresh = null;
    expect(await pipeline.recoverFrom401({ url: '/api/contacts', detail: 'x', sentToken: 'old' })).toBeNull();
  });
});
