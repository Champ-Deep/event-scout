// The app's response check against the wire contract (@es/types), run in lib/api.ts.
// Development (__DEV__): warns in the console when a response drifts from the contract. Release builds:
// a no-op (nothing is parsed). It never throws and always returns the data it was given.
// No React Native imports here, so the unit tests (Node) can load it.
import { createResponseCheck, type ResponseCheck } from '@es/types';

declare const __DEV__: boolean | undefined;

export const checkResponse: ResponseCheck = createResponseCheck({ enabled: typeof __DEV__ !== 'undefined' && !!__DEV__ });

/** `.then(checked(schema, 'GET /api/…'))`: checks a response and passes it on unchanged. */
export const checked = (schema: Parameters<ResponseCheck>[0], endpoint: string) =>
  <T>(data: T): T => checkResponse(schema, data, endpoint);
