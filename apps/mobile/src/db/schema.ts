import { sqliteTable, text, integer } from 'drizzle-orm/sqlite-core';

export const contacts = sqliteTable('contacts', {
  id: text('id').primaryKey(),
  eventId: text('event_id').notNull(),
  firstName: text('first_name'),
  lastName: text('last_name'),
  company: text('company'),
  title: text('title'),
  email: text('email'),
  phone: text('phone'),
  notes: text('notes'),
  score: integer('score'),
  scoreBand: text('score_band', { enum: ['A', 'B', 'C', 'D', 'F'] }),
  source: text('source', { enum: ['scan', 'manual', 'import'] }).notNull().default('scan'),
  stage: text('stage', { enum: ['new', 'contacted', 'qualified', 'converted'] }).notNull().default('new'),
  version: integer('version').notNull().default(1),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  scannedAt: text('scanned_at'),
  // JSON of the full contact (photoUrl, linkedin, real source, eventId…) — see hydrateContact.
  raw: text('raw'),
});

export const events = sqliteTable('events', {
  id: text('id').primaryKey(),
  name: text('name').notNull(),
  startDate: text('start_date').notNull(),
  endDate: text('end_date').notNull(),
  timezone: text('timezone').notNull(),
  location: text('location'),
  isActive: integer('is_active', { mode: 'boolean' }).notNull().default(true),
});

export const outbox = sqliteTable('outbox', {
  id: text('id').primaryKey(),
  entityType: text('entity_type', { enum: ['contact', 'event'] }).notNull(),
  entityId: text('entity_id').notNull(),
  action: text('action', { enum: ['create', 'update', 'delete'] }).notNull(),
  baseVersion: integer('base_version').notNull().default(0),
  payload: text('payload').notNull(), // JSON stringified patch
  createdAt: text('created_at').notNull(),
  attempts: integer('attempts').notNull().default(0),
  lastError: text('last_error'),
});

export const syncMeta = sqliteTable('sync_meta', {
  key: text('key').primaryKey(),
  value: text('value').notNull(),
  updatedAt: text('updated_at').notNull(),
});

// Scans taken while offline: the photo is kept on the phone and read by the server once back online.
export const scanQueue = sqliteTable('scan_queue', {
  id: text('id').primaryKey(),
  kind: text('kind', { enum: ['card', 'qr'] }).notNull(),
  imageUri: text('image_uri').notNull(),
  eventName: text('event_name'),
  createdAt: text('created_at').notNull(),
  attempts: integer('attempts').notNull().default(0),
  lastError: text('last_error'),
});
