import { describe, expect, it } from 'vitest';
import {
  ApiError, SESSION_REVOKED_DETAIL, asApiError, classifyError, errorMessage, isApiError, isNetworkError, isOffline,
} from '../src/lib/apiError';

const axiosLike = (status: number, detail?: unknown) => Object.assign(new Error(`Request failed with status code ${status}`), {
  response: { status, data: detail === undefined ? {} : { detail } },
});

describe('classifyError', () => {
  it('sorts every failure into one kind, keeping the original response', () => {
    expect(classifyError(new Error('Network Error'), { deviceOffline: true }).kind).toBe('offline');
    expect(classifyError(new Error('Network Error')).kind).toBe('unreachable');
    expect(classifyError(Object.assign(new Error('timeout of 15000ms exceeded'), { code: 'ECONNABORTED' })).kind).toBe('timeout');
    expect(classifyError(Object.assign(new Error('canceled'), { code: 'ERR_CANCELED' })).kind).toBe('canceled');
    expect(classifyError(axiosLike(401, SESSION_REVOKED_DETAIL)).kind).toBe('revoked');
    expect(classifyError(axiosLike(401, 'Expired'), { sessionRoute: true }).kind).toBe('revoked');
    expect(classifyError(axiosLike(401, 'Expired')).kind).toBe('unauthorized');
    expect(classifyError(axiosLike(404, 'Contact not found')).kind).toBe('not_found');
    expect(classifyError(axiosLike(422, [{ msg: 'bad' }])).kind).toBe('validation');
    expect(classifyError(axiosLike(409, 'Already exists')).kind).toBe('validation');
    expect(classifyError(axiosLike(503, 'AI unavailable')).kind).toBe('server');

    const conflict = classifyError(axiosLike(409, { message: 'Changed', conflicts: [{ field: 'email' }] }));
    expect(conflict.kind).toBe('conflict');
    expect(conflict.conflicts).toEqual([{ field: 'email' }]);
    expect(conflict.displayMessage).toBe('Changed');

    const server = classifyError(axiosLike(500, 'Boom'));
    expect(server.response).toEqual({ status: 500, data: { detail: 'Boom' }, headers: undefined });
    expect(server.status).toBe(500);
    expect(server.displayMessage).toBe('Boom');
    expect(server).toBeInstanceOf(ApiError);
    expect(isApiError(server)).toBe(true);
    expect(classifyError(server)).toBe(server);
  });

  it("keeps the old 'OFFLINE' message so existing checks still work", () => {
    const e = classifyError(new Error('x'), { deviceOffline: true });
    expect(e.message).toBe('OFFLINE');
    expect(isOffline(e)).toBe(true);
    expect(isOffline(new Error('OFFLINE'))).toBe(true);
  });

  it('a response wins over a flaky "offline" reading', () => {
    expect(classifyError(axiosLike(400, 'Bad'), { deviceOffline: true }).kind).toBe('validation');
  });
});

describe('isNetworkError', () => {
  it('is true only when the request never got an answer', () => {
    expect(isNetworkError(new Error('OFFLINE'))).toBe(true);
    expect(isNetworkError(Object.assign(new Error('x'), { code: 'ECONNABORTED' }))).toBe(true);
    expect(isNetworkError(new Error('Network request failed'))).toBe(true);
    expect(isNetworkError(axiosLike(500))).toBe(false);
    expect(isNetworkError(new Error('Please log in to scan cards.'))).toBe(false);
    expect(isNetworkError(null)).toBe(false);
  });
});

describe('errorMessage', () => {
  it("shows the server's detail, else the screen's fallback", () => {
    expect(errorMessage(axiosLike(400, 'Email is invalid'), 'Could not save.')).toBe('Email is invalid');
    expect(errorMessage(axiosLike(500), 'Could not save.')).toBe('Could not save.');
    expect(errorMessage(axiosLike(422, [{ msg: 'x' }]), 'Could not save.')).toBe('Could not save.');
  });

  it('uses the shared wording for connection problems, or the screen’s own', () => {
    expect(errorMessage(new Error('OFFLINE'), 'f')).toBe("You're offline.");
    expect(errorMessage(new Error('Network Error'), 'f')).toBe("Can't reach the server.");
    const timeout = Object.assign(new Error('t'), { code: 'ECONNABORTED' });
    expect(errorMessage(timeout, 'f')).toBe("Can't reach the server.");
    expect(errorMessage(timeout, 'f', { timeout: 'Too slow.' })).toBe('Too slow.');
    expect(errorMessage(new Error('OFFLINE'), 'f', { offline: "You're offline — nothing was saved." })).toBe("You're offline — nothing was saved.");
    expect(errorMessage(new Error('Network Error'), 'f', { unreachable: 'No server.' })).toBe('No server.');
  });

  it('screen-specific answers for 401 / 413 / 422 lists / 501', () => {
    expect(errorMessage(axiosLike(401, 'Token expired'), 'f', { unauthorized: 'Sign in again.' })).toBe('Sign in again.');
    expect(errorMessage(axiosLike(401, 'Token expired'), 'f')).toBe('Token expired');
    expect(errorMessage(axiosLike(413), 'f', { tooLarge: 'Too big.' })).toBe('Too big.');
    expect(errorMessage(axiosLike(422, [{ msg: 'x' }]), 'f', { invalid: 'Check the email.' })).toBe('Check the email.');
    expect(errorMessage(axiosLike(501), 'f', { notAvailable: 'Not set up.' })).toBe('Not set up.');
    expect(errorMessage(axiosLike(401, 'Invalid API Key'), 'f')).toMatch(/API key mismatch/);
  });

  it("shows the app's own error messages as they are", () => {
    expect(errorMessage(new Error('Sign in again to save.'), 'f')).toBe('Sign in again to save.');
    expect(asApiError(new Error('Sign in again to save.'))).toBeNull();
    expect(errorMessage(undefined, 'Fallback')).toBe('Fallback');
  });
});
