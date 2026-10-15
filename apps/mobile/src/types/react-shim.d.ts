// Compatibility shim: Augment React 18 types so ReactNode permits bigint,
// which prevents JSX type mismatch errors with libraries (like expo-router)
// that resolve types against React 19.
import 'react';

declare module 'react' {
  interface DO_NOT_USE_OR_YOU_WILL_BE_FIRED_EXPERIMENTAL_REACT_NODES {
    bigints: bigint;
  }
}

declare global {
  namespace React {
    interface DO_NOT_USE_OR_YOU_WILL_BE_FIRED_EXPERIMENTAL_REACT_NODES {
      bigints: bigint;
    }
  }
}
