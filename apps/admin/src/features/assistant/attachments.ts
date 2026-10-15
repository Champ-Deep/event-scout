// Files for a question — the same rules as the server (read_chat_attachments) and the app:
// up to 3 per question, 10 MB each, images / PDF / text (CSV, Markdown, JSON, vCard).
export const MAX_ATTACHMENTS = 3;
export const MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024;
export const ACCEPT = 'image/jpeg,image/png,image/webp,image/heic,image/heif,application/pdf,text/plain,text/csv,text/markdown,application/json,text/tab-separated-values,.csv,.md,.txt,.json,.tsv,.vcf,.pdf,.heic,.heif';
const OK = /\.(jpe?g|png|webp|heic|heif|pdf|txt|csv|md|json|tsv|vcf)$/i;

/** What a question's message keeps about its files (the file itself isn't stored). */
export interface AttachmentMeta { name: string; size: number; type: string; thumb?: string | undefined }

export const isImage = (a: { type: string; name: string }) => a.type.startsWith('image/') || /\.(jpe?g|png|webp|heic|heif)$/i.test(a.name);

export function formatBytes(n?: number) {
  if (!n) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** Add `incoming` to `current`; returns the new list and a problem to show (or ''). */
export function addFiles(current: File[], incoming: File[]): { files: File[]; problem: string } {
  const files = [...current];
  let problem = '';
  for (const f of incoming) {
    if (files.length >= MAX_ATTACHMENTS) { problem = `You can attach up to ${MAX_ATTACHMENTS} files per question.`; break; }
    if (!OK.test(f.name) && !/^(image\/(jpeg|png|webp|heic|heif)|application\/pdf|text\/|application\/json)/.test(f.type)) { problem = `“${f.name}” isn't supported. Attach an image, PDF, or text/CSV file.`; continue; }
    if (f.size > MAX_ATTACHMENT_BYTES) { problem = `“${f.name}” is larger than 10 MB.`; continue; }
    if (!f.size) { problem = `“${f.name}” is empty.`; continue; }
    if (files.some((x) => x.name === f.name && x.size === f.size)) continue;
    files.push(f);
  }
  return { files, problem };
}

/** A small JPEG preview of an image, kept with the message so the chat history still shows it. */
export async function thumbnail(f: File, size = 96): Promise<string | undefined> {
  if (!f.type.startsWith('image/') || /hei[cf]/i.test(f.type)) return undefined;
  try {
    const bmp = await createImageBitmap(f);
    const k = size / Math.max(bmp.width, bmp.height);
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(bmp.width * k));
    c.height = Math.max(1, Math.round(bmp.height * k));
    c.getContext('2d')?.drawImage(bmp, 0, 0, c.width, c.height);
    bmp.close?.();
    return c.toDataURL('image/jpeg', 0.7);
  } catch {
    return undefined;
  }
}

export const metaOf = async (files: File[]): Promise<AttachmentMeta[]> =>
  Promise.all(files.map(async (f) => ({ name: f.name, size: f.size, type: f.type, thumb: await thumbnail(f) })));
