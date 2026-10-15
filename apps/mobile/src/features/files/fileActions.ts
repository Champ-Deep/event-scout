import { Platform } from 'react-native';
import * as LegacyFS from 'expo-file-system/legacy';
import * as Sharing from 'expo-sharing';
import type { FileContent } from '../../lib/api';
import { saveFileToDownloads } from '../me/export';

const safeName = (name: string) => (name || 'file').replace(/[\\/:*?"<>|]+/g, '_').trim() || 'file';

function webDownload(c: FileContent) {
  const bytes = Uint8Array.from(atob(c.contentBase64), (ch) => ch.charCodeAt(0));
  const url = URL.createObjectURL(new Blob([bytes], { type: c.mimeType }));
  const a = document.createElement('a');
  a.href = url;
  a.download = safeName(c.filename);
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Open the file with another app (PDF viewer, Slides, Drive, WhatsApp…) via the share sheet. */
export async function openFile(c: FileContent): Promise<void> {
  if (Platform.OS === 'web') { webDownload(c); return; }
  const uri = `${LegacyFS.cacheDirectory}${safeName(c.filename)}`;
  await LegacyFS.writeAsStringAsync(uri, c.contentBase64, { encoding: LegacyFS.EncodingType.Base64 });
  if (!(await Sharing.isAvailableAsync())) throw new Error("Opening files isn't available on this device.");
  await Sharing.shareAsync(uri, { mimeType: c.mimeType, dialogTitle: c.filename });
}

/** Save the file on the phone: Android → your Downloads folder; iOS → the share sheet ("Save to Files"). */
export async function saveFile(c: FileContent): Promise<string> {
  if (Platform.OS === 'web') { webDownload(c); return 'Downloads'; }
  if (Platform.OS !== 'android') { await openFile(c); return 'Files'; }
  // Android adds the extension from the MIME type, so pass the name without it.
  const base = safeName(c.filename).replace(/\.[a-z0-9]{1,5}$/i, '');
  const { where } = await saveFileToDownloads(base, c.mimeType, c.contentBase64, 'base64');
  return where;
}

export function formatSize(bytes: number): string {
  if (!bytes) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
