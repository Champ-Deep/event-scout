import React, { useEffect, useState } from 'react';
import { View, ActivityIndicator, Platform } from 'react-native';
import Animated, { FadeIn, FadeInDown, LinearTransition } from 'react-native-reanimated';
import * as DocumentPicker from 'expo-document-picker';
import * as LegacyFS from 'expo-file-system/legacy';
import { Paperclip, Plus, AlertCircle } from 'lucide-react-native';
import { Text } from '../../ui/Text';
import { theme, makeStyles } from '../../ui/theme';
import { api, errorMessage, type ContactFile } from '../../lib/api';
import { PressableScale } from '../onboarding/components';
import { FileList } from './FileList';

// Same limits as the backend (POST /api/contact/{id}/files).
const MAX_BYTES = 20 * 1024 * 1024;
const EXT_BY_MIME: Record<string, string> = {
  'application/pdf': 'pdf',
  'application/vnd.ms-powerpoint': 'ppt',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
  'application/msword': 'doc',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
  'application/vnd.ms-excel': 'xls',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
  'text/csv': 'csv',
  'text/plain': 'txt',
  'image/png': 'png',
  'image/jpeg': 'jpg',
};
const ALLOWED = new Set([...Object.values(EXT_BY_MIME), 'jpeg']);


async function readBase64(uri: string): Promise<string> {
  if (Platform.OS !== 'web') return LegacyFS.readAsStringAsync(uri, { encoding: LegacyFS.EncodingType.Base64 });
  const blob = await (await fetch(uri)).blob();
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(',', 2)[1] || '');
    r.onerror = () => reject(new Error('Could not read the file.'));
    r.readAsDataURL(blob);
  });
}

/**
 * Documents attached to this contact — list (GET /api/contact/{id}/files), open / save
 * (GET /api/contact/files/{id}/download), attach (POST /api/contact/{id}/files) and remove
 * (DELETE /api/contact/files/{id}).
 */
export function ContactFilesCard({ contactId, reloadKey }: { contactId: string; reloadKey?: number }) {
  const [files, setFiles] = useState<ContactFile[] | null>(null);
  const [attaching, setAttaching] = useState(false);
  const [err, setErr] = useState('');

  useEffect(() => {
    let active = true;
    api.getContactFiles(contactId)
      .then((list) => { if (active) setFiles(Array.isArray(list) ? list : []); })
      .catch(() => { if (active) setFiles((prev) => prev ?? []); });
    return () => { active = false; };
  }, [contactId, reloadKey]);

  const attach = async () => {
    if (attaching) return;
    setErr('');
    try {
      const res = await DocumentPicker.getDocumentAsync({ type: Object.keys(EXT_BY_MIME), multiple: false, copyToCacheDirectory: true });
      if (res.canceled || !res.assets?.length) return;
      const a = res.assets[0]!;
      let name = (a.name || 'file').trim();
      let ext = name.includes('.') ? name.split('.').pop()!.toLowerCase() : '';
      if (!ALLOWED.has(ext) && a.mimeType && EXT_BY_MIME[a.mimeType]) {
        ext = EXT_BY_MIME[a.mimeType]!;
        name = `${name}.${ext}`;
      }
      if (!ALLOWED.has(ext)) { setErr('Use a PDF, slide deck, Word / Excel / CSV / text file, or a PNG / JPG image.'); return; }
      if ((a.size || 0) > MAX_BYTES) { setErr('That file is larger than 20 MB.'); return; }
      setAttaching(true);
      const added = await api.attachContactFile(contactId, { filename: name, contentBase64: await readBase64(a.uri) });
      setFiles((prev) => [added, ...(prev || [])]);
    } catch (e: any) {
      setErr(errorMessage(e, "Couldn't attach the file."));
    } finally {
      setAttaching(false);
    }
  };

  const remove = async (id: string) => {
    await api.deleteContactFile(id);
    setFiles((prev) => (prev || []).filter((f) => f.id !== id));
  };

  return (
    <Animated.View entering={FadeInDown.duration(260)} layout={LinearTransition.duration(240)} style={styles.card}>
      <View style={styles.cardHeader}>
        <View style={styles.cardIcon}><Paperclip size={16} color={theme.colors.brand[700]} /></View>
        <Text style={styles.cardTitle}>FILES</Text>
        {files?.length ? <Text style={styles.count}>{files.length}</Text> : null}
        <PressableScale onPress={attach} disabled={attaching} scaleTo={0.92} style={styles.attachBtn} accessibilityLabel="Attach a file to this contact"
          leadingIcon={attaching ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <Plus size={13} color={theme.colors.brand[700]} />}>
          <Text style={styles.attachText}>{attaching ? 'Attaching…' : 'Attach'}</Text>
        </PressableScale>
      </View>
      {files == null ? null : files.length ? (
        <FileList
          download={(id) => api.downloadContactFile(id)}
          onRemove={remove}
          files={files.map((f) => ({ id: f.id, filename: f.filename, fileType: f.fileType, fileSize: f.fileSize, description: f.description,
            meta: f.createdAt ? new Date(f.createdAt).toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric' }) : null }))}
        />
      ) : (
        <Text style={styles.empty}>No files yet. Attach a proposal, the brochure they gave you, or a photo — you can open or save it here later.</Text>
      )}
      {err ? (
        <Animated.View entering={FadeIn.duration(200)} style={styles.errRow}>
          <AlertCircle size={14} color={theme.colors.semantic.danger} />
          <Text style={styles.errText}>{err}</Text>
        </Animated.View>
      ) : null}
    </Animated.View>
  );
}

// Same look as the other cards on the contact screen.
const styles = makeStyles(() => ({
  card: { backgroundColor: theme.colors.bg.surface, borderRadius: 20, padding: 16, marginBottom: 12, borderWidth: 1, borderColor: theme.colors.bg.divider, ...theme.shadows.card },
  cardHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 10 },
  cardIcon: { width: 30, height: 30, borderRadius: 10, backgroundColor: theme.colors.brand[50], alignItems: 'center', justifyContent: 'center' },
  cardTitle: { flex: 1, fontSize: 11, fontWeight: '800', color: theme.colors.text.display, letterSpacing: 1 },
  count: { fontSize: 12, fontWeight: '800', color: theme.colors.text.muted },
  attachBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 11, paddingVertical: 6, borderRadius: 999, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  attachText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  empty: { fontSize: 13, lineHeight: 19, color: theme.colors.text.muted },
  errRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 8 },
  errText: { flex: 1, fontSize: 12, fontWeight: '600', color: theme.colors.semantic.danger },
}));
