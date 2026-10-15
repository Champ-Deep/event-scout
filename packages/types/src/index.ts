// @es/types — the wire contract between apps/backend and the two clients (admin console, mobile app).
// The backend (FastAPI, hand-built dicts) is the source of truth; each schema names the handler it mirrors.
// Each app's HTTP adapter checks responses against these in development (createResponseCheck).
export * from './wire';
export * from './contact';
export * from './files';
export * from './event';
export * from './broadcast';
export * from './converse';
