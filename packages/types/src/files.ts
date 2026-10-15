import { z } from 'zod';
import { loose, type WireOf } from './wire';

/** A document attached to a contact (GET/POST /api/contact/{id}/files). */
export const contactFileSchema = loose({
  id: z.string(),
  filename: z.string(),
  fileType: z.string(),
  /** "general" for uploads; older rows may have none */
  category: z.string().nullable(),
  fileSize: z.number(),
  description: z.string(),
  createdAt: z.string().nullable(),
});
export type ContactFile = WireOf<typeof contactFileSchema>;

/** A team document the admins uploaded (GET /api/files/list). */
export const eventFileSchema = loose({
  id: z.string(),
  filename: z.string(),
  fileType: z.string(),
  mimeType: z.string(),
  category: z.string().nullable(),
  fileSize: z.number(),
  description: z.string(),
  downloadCount: z.number(),
  /** null = for every event */
  eventName: z.string().nullable(),
  /** demo row without the real file — can't be downloaded */
  placeholder: z.boolean(),
  createdAt: z.string().nullable(),
});
export type EventFile = WireOf<typeof eventFileSchema>;

/** A file's content (GET /api/files/{id}/download, GET /api/contact/files/{id}/download). */
export const fileContentSchema = loose({
  filename: z.string(),
  mimeType: z.string(),
  fileSize: z.number(),
  contentBase64: z.string(),
});
export type FileContent = WireOf<typeof fileContentSchema>;
