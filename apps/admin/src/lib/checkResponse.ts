// The admin console's response check against the wire contract (@es/types), run in lib/api.ts.
// Development: warns in the console when a response drifts from the contract. Production: a no-op
// (nothing is parsed). It never throws and always returns the data it was given.
import { createResponseCheck, type ResponseCheck } from '@es/types';

export const checkResponse: ResponseCheck = createResponseCheck({ enabled: import.meta.env.DEV });

/** `.then(checked(schema, 'GET /api/…'))`: checks a response and passes it on unchanged. */
export const checked = (schema: Parameters<ResponseCheck>[0], endpoint: string) =>
  <T>(data: T): T => checkResponse(schema, data, endpoint);
