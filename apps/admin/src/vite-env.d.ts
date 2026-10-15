/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend origin for a production build hosted apart from the API (e.g. https://api.example.com). Unset = same origin / Vite dev proxy. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
