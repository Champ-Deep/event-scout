import { Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';
import type { ChatAttachment } from '../../lib/api';

// Must match the backend (/converse/with_files/): 3 files, 10 MB each, images / PDF / text.
export const MAX_ATTACHMENTS = 3;
export const MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024;
const DOC_TYPES = ['application/pdf', 'text/plain', 'text/csv', 'text/comma-separated-values', 'text/markdown', 'application/json', 'text/x-vcard', 'text/vcard', 'image/*'];

export type AttachSource = 'camera' | 'library' | 'document';

export class AttachmentError extends Error {}

export function isImage(a: Pick<ChatAttachment, 'mimeType' | 'name'>) {
  return (a.mimeType || '').startsWith('image/') || /\.(jpe?g|png|webp|heic|heif)$/i.test(a.name);
}

export function formatBytes(n?: number) {
  if (!n) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** Pick up to `room` files for a question. Resolves [] when cancelled. */
export async function pickAttachments(source: AttachSource, room: number): Promise<ChatAttachment[]> {
  if (room <= 0) throw new AttachmentError(`You can attach up to ${MAX_ATTACHMENTS} files per question.`);

  let picked: ChatAttachment[] = [];
  if (source === 'document') {
    const res = await DocumentPicker.getDocumentAsync({ type: DOC_TYPES, multiple: room > 1, copyToCacheDirectory: true });
    if (res.canceled) return [];
    picked = res.assets.map((a) => ({ uri: a.uri, name: a.name, mimeType: a.mimeType, size: a.size }));
  } else {
    if (source === 'camera') {
      const perm = await ImagePicker.requestCameraPermissionsAsync();
      if (!perm.granted) throw new AttachmentError('Camera access is off. Enable it in Settings to attach a photo.');
    } else if (Platform.OS !== 'web') {
      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) throw new AttachmentError('Photo library access is off. Enable it in Settings to attach a photo.');
    }
    const options: ImagePicker.ImagePickerOptions = {
      mediaTypes: ['images'],
      quality: 0.7, // plenty for the AI to read a slide / badge / booth sign; keeps uploads small
      allowsMultipleSelection: source === 'library' && room > 1,
      selectionLimit: room,
    };
    const res = source === 'camera'
      ? await ImagePicker.launchCameraAsync(options)
      : await ImagePicker.launchImageLibraryAsync(options);
    if (res.canceled) return [];
    picked = res.assets.map((a, i) => {
      const ext = (a.mimeType?.split('/')[1] || 'jpg').replace('jpeg', 'jpg');
      return {
        uri: a.uri,
        name: a.fileName || `photo-${Date.now()}-${i + 1}.${ext}`,
        mimeType: a.mimeType || 'image/jpeg',
        size: a.fileSize,
      };
    });
  }

  const tooBig = picked.find((a) => (a.size || 0) > MAX_ATTACHMENT_BYTES);
  if (tooBig) throw new AttachmentError(`"${tooBig.name}" is larger than 10 MB.`);
  if (picked.length > room) {
    throw new AttachmentError(`You can attach up to ${MAX_ATTACHMENTS} files per question.`);
  }
  return picked;
}
