import { Platform } from 'react-native';
import { eq } from 'drizzle-orm';
import { File, Paths } from 'expo-file-system';
import * as LegacyFS from 'expo-file-system/legacy';
import * as Sharing from 'expo-sharing';
import { api } from '../../lib/api';
import { db } from '../../db/client';
import { syncMeta } from '../../db/schema';

export type ExportFormat = 'csv' | 'json';

const TYPES: Record<ExportFormat, { mime: string; uti: string }> = {
  csv: { mime: 'text/csv', uti: 'public.comma-separated-values-text' },
  json: { mime: 'application/json', uti: 'public.json' },
};

/**
 * GET /export_contacts/?format=csv|json →
 *   CSV  — one row per contact (Excel / Google Sheets); starts with a BOM so Excel reads UTF-8 names.
 *   JSON — the full records ({ format, exported_at, total, contacts: [...] }), nulls for missing values.
 */
async function fetchExport(userId: string, format: ExportFormat) {
  const data: any = await api.exportContacts(userId, format);
  let body: string;
  let count: number;
  if (format === 'csv') {
    const csv = typeof data === 'string' ? data : '';
    if (!csv.trim()) throw new Error("The server didn't return any data to export.");
    count = Math.max(0, csv.trim().split(/\r?\n/).length - 1);
    body = csv.startsWith('﻿') ? csv : `﻿${csv}`;
  } else {
    if (!data || !Array.isArray(data.contacts)) throw new Error("The server didn't return any data to export.");
    count = data.contacts.length;
    const { status: _s, ...doc } = data;
    body = JSON.stringify(doc, null, 2);
  }
  const stamp = new Date().toISOString().slice(0, 16).replace('T', '_').replace(':', '-'); // 2026-09-30_13-05
  return { body, count, base: `eventscout-contacts-${stamp}`, name: `eventscout-contacts-${stamp}.${format}` };
}

function webDownload(body: string, name: string, format: ExportFormat) {
  const url = URL.createObjectURL(new Blob([body], { type: TYPES[format].mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

async function shareFile(body: string, name: string, format: ExportFormat) {
  const file = new File(Paths.cache, name);
  if (file.exists) file.delete();
  file.create();
  file.write(body);
  if (!(await Sharing.isAvailableAsync())) throw new Error("Sharing isn't available on this device.");
  await Sharing.shareAsync(file.uri, {
    mimeType: TYPES[format].mime,
    dialogTitle: `Export contacts (${format.toUpperCase()})`,
    UTI: TYPES[format].uti,
  });
}

/** Export and open the share sheet (Drive, email, WhatsApp…). On web it downloads. Resolves the count. */
export async function exportContacts(userId: string, format: ExportFormat): Promise<number> {
  const { body, count, name } = await fetchExport(userId, format);
  if (Platform.OS === 'web') webDownload(body, name, format);
  else await shareFile(body, name, format);
  return count;
}

/** Kept for older callers. */
export const exportContactsCsv = (userId: string) => exportContacts(userId, 'csv');

// ---- Save to Downloads ----

const DIR_KEY = 'downloads_dir_uri';

async function savedDir(): Promise<string | null> {
  try {
    const rows = await db.select().from(syncMeta).where(eq(syncMeta.key, DIR_KEY)).limit(1);
    return rows?.[0]?.value || null;
  } catch {
    return null;
  }
}

async function rememberDir(uri: string | null) {
  try {
    if (!uri) { await db.delete(syncMeta).where(eq(syncMeta.key, DIR_KEY)); return; }
    const now = new Date().toISOString();
    await db.insert(syncMeta).values({ key: DIR_KEY, value: uri, updatedAt: now })
      .onConflictDoUpdate({ target: syncMeta.key, set: { value: uri, updatedAt: now } });
  } catch { /* not remembered: the picker just opens again next time */ }
}

/** "Downloads/EventScout" for a folder inside Download, else the folder's path. */
function folderLabel(dirUri: string): string {
  const path = decodeURIComponent(dirUri.split('/tree/')[1] || dirUri).replace(/^[^:]*:/, '');
  const parts = path.split('/').filter(Boolean);
  if (!parts.length) return 'the folder you picked';
  if (/^downloads?$/i.test(parts[0]!)) parts[0] = 'Downloads';
  return parts.join('/');
}

export interface SavedExport { count: number; where: string; askedForFolder: boolean }

/**
 * Save the export as a real file on the phone.
 * Android: Storage Access Framework — the folder picker opens at Downloads the first time. Android 11+
 * doesn't let apps take the Download folder itself, so you pick / create a folder INSIDE it (e.g.
 * Download/EventScout) and tap "Use this folder"; it's remembered and later saves go straight there.
 * iOS has no Downloads folder: opens the share sheet, where "Save to Files" saves it. Web: downloads.
 */
export async function saveContactsToDownloads(userId: string, format: ExportFormat): Promise<SavedExport> {
  const { body, count, base, name } = await fetchExport(userId, format);
  if (Platform.OS === 'web') { webDownload(body, name, format); return { count, where: 'Downloads', askedForFolder: false }; }
  if (Platform.OS !== 'android') { await shareFile(body, name, format); return { count, where: 'Files', askedForFolder: false }; }

  const saved = await saveFileToDownloads(base, TYPES[format].mime, body, 'utf8');
  return { count, ...saved };
}

/**
 * Android: write any file into the remembered Downloads folder (Storage Access Framework), asking
 * for the folder once (picker opens at Downloads; pick / create e.g. Download/EventScout).
 * data = text (utf8) or base64 (e.g. a PNG). Shared by the contacts export and the business card.
 */
export async function saveFileToDownloads(base: string, mime: string, data: string, encoding: 'utf8' | 'base64'): Promise<{ where: string; askedForFolder: boolean }> {
  const SAF = LegacyFS.StorageAccessFramework;
  const write = async (dirUri: string) => {
    // Android adds the extension from the MIME type (and " (1)" if the name exists).
    const fileUri = await SAF.createFileAsync(dirUri, base, mime);
    await LegacyFS.writeAsStringAsync(fileUri, data, { encoding: encoding === 'base64' ? LegacyFS.EncodingType.Base64 : LegacyFS.EncodingType.UTF8 });
  };

  let dir = await savedDir();
  if (dir) {
    try {
      await write(dir);
      return { where: folderLabel(dir), askedForFolder: false };
    } catch (e: any) {
      console.warn('[export] saved folder no longer writable, asking again:', e?.message);
      await rememberDir(null); // access was revoked / folder removed
    }
  }
  const downloads = SAF.getUriForDirectoryInRoot('Download');
  const perm = await SAF.requestDirectoryPermissionsAsync(downloads);
  if (!perm.granted) {
    throw new Error('Not saved — no folder was chosen. In Downloads, create or open a folder (e.g. "EventScout"), then tap "Use this folder".');
  }
  dir = perm.directoryUri;
  await write(dir);
  await rememberDir(dir);
  return { where: folderLabel(dir), askedForFolder: true };
}

/** Forget the remembered folder (next save asks again). */
export const resetDownloadsFolder = () => rememberDir(null);

/** Where saves go now, e.g. "Downloads/EventScout" — null until a folder has been picked. */
export async function downloadsFolderLabel(): Promise<string | null> {
  const dir = await savedDir();
  return dir ? folderLabel(dir) : null;
}
