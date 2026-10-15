import React, { useEffect, useState } from 'react';
import { View, Pressable, ActivityIndicator, Platform, Alert } from 'react-native';
import Animated, { FadeIn, FadeInDown, FadeOut, LinearTransition } from 'react-native-reanimated';
import { FileText, FileSpreadsheet, FileImage, Presentation, Download, ExternalLink, CheckCircle2, AlertCircle, Trash2 } from 'lucide-react-native';
import { Text } from '../../ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../ui/theme';
import { errorMessage, type FileContent } from '../../lib/api';
import { openFile, saveFile, formatSize } from './fileActions';

export interface FileItem {
  id: string;
  filename: string;
  fileType?: string | null;
  fileSize: number;
  description?: string | null;
  /** extra line, e.g. "Pricing · GITEX 2026" */
  meta?: string | null;
}

function FileIcon({ type }: { type?: string | null | undefined }) {
  const t = (type || '').toLowerCase();
  if (['ppt', 'pptx', 'key'].includes(t)) return <Presentation size={16} color={toneFg('#c2410c')} />;
  if (['xls', 'xlsx', 'csv'].includes(t)) return <FileSpreadsheet size={16} color={toneFg('#15803d')} />;
  if (['png', 'jpg', 'jpeg'].includes(t)) return <FileImage size={16} color={toneFg('#0369a1')} />;
  return <FileText size={16} color={toneFg('#b91c1c')} />;
}


/**
 * Tap a file to open it with another app (share sheet); the download button saves it to the phone
 * (Android: your Downloads folder). Content comes from `download(id)`.
 */
export function FileList({ files, download, onRemove }: {
  files: FileItem[];
  download: (id: string) => Promise<FileContent>;
  /** shows a remove button (asks first); resolve when removed */
  onRemove?: ((id: string) => Promise<void>) | undefined;
}) {
  const [busy, setBusy] = useState<{ id: string; action: 'open' | 'save' | 'remove' } | null>(null);
  const [status, setStatus] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    if (!status) return;
    const t = setTimeout(() => setStatus(null), 3500);
    return () => clearTimeout(t);
  }, [status]);

  const run = async (f: FileItem, action: 'open' | 'save') => {
    if (busy) return;
    setBusy({ id: f.id, action });
    setStatus(null);
    try {
      const content = await download(f.id);
      if (action === 'open') await openFile(content);
      else setStatus({ ok: true, text: `Saved “${content.filename}” to ${await saveFile(content)}` });
    } catch (e: any) {
      setStatus({ ok: false, text: errorMessage(e, `Couldn't ${action === 'open' ? 'open' : 'save'} the file.`) });
    } finally {
      setBusy(null);
    }
  };

  const remove = (f: FileItem) => {
    if (busy || !onRemove) return;
    const go = async () => {
      setBusy({ id: f.id, action: 'remove' });
      setStatus(null);
      try {
        await onRemove(f.id);
        setStatus({ ok: true, text: `Removed “${f.filename}”` });
      } catch (e: any) {
        setStatus({ ok: false, text: errorMessage(e, "Couldn't remove the file.") });
      } finally {
        setBusy(null);
      }
    };
    if (Platform.OS === 'web') { go(); return; }
    Alert.alert('Remove file?', `“${f.filename}” will be removed from this contact.`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Remove', style: 'destructive', onPress: go },
    ]);
  };

  return (
    <View style={{ gap: 8 }}>
      {files.map((f, i) => {
        const opening = busy?.id === f.id && busy.action === 'open';
        const saving = busy?.id === f.id && busy.action === 'save';
        const removing = busy?.id === f.id && busy.action === 'remove';
        const sub = [(f.fileType || '').toUpperCase(), formatSize(f.fileSize), f.meta].filter(Boolean).join(' · ');
        return (
          <Animated.View key={f.id} entering={FadeInDown.delay(Math.min(i, 8) * 40).duration(260)} layout={LinearTransition.duration(240)}>
            <Pressable onPress={() => run(f, 'open')} disabled={!!busy} accessibilityRole="button" accessibilityLabel={`Open ${f.filename}`}
              style={({ pressed }) => [styles.row, pressed && { opacity: 0.85 }]}>
              <View style={styles.icon}>{opening ? <ActivityIndicator size="small" color={theme.colors.brand[600]} /> : <FileIcon type={f.fileType} />}</View>
              <View style={{ flex: 1 }}>
                <Text style={styles.name} numberOfLines={1}>{f.filename}</Text>
                {sub ? <Text style={styles.sub} numberOfLines={1}>{sub}</Text> : null}
                {f.description ? <Text style={styles.desc} numberOfLines={2}>{f.description}</Text> : null}
              </View>
              <ExternalLink size={15} color={theme.colors.text.muted} />
              {Platform.OS !== 'ios' ? (
                <Pressable onPress={() => run(f, 'save')} disabled={!!busy} hitSlop={8} style={styles.saveBtn}
                  accessibilityRole="button" accessibilityLabel={`Save ${f.filename} to Downloads`}>
                  {saving ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <Download size={15} color={theme.colors.brand[700]} />}
                </Pressable>
              ) : null}
              {onRemove ? (
                <Pressable onPress={() => remove(f)} disabled={!!busy} hitSlop={8} style={styles.removeBtn}
                  accessibilityRole="button" accessibilityLabel={`Remove ${f.filename}`}>
                  {removing ? <ActivityIndicator size="small" color={theme.colors.semantic.danger} /> : <Trash2 size={14} color={theme.colors.semantic.danger} />}
                </Pressable>
              ) : null}
            </Pressable>
          </Animated.View>
        );
      })}
      {status ? (
        <Animated.View entering={FadeIn.duration(200)} exiting={FadeOut.duration(160)} style={styles.status}>
          {status.ok ? <CheckCircle2 size={14} color={theme.colors.semantic.successText} /> : <AlertCircle size={14} color={theme.colors.semantic.danger} />}
          <Text style={[styles.statusText, { color: status.ok ? theme.colors.semantic.successText : theme.colors.semantic.danger }]}>{status.text}</Text>
        </Animated.View>
      ) : null}
    </View>
  );
}

const styles = makeStyles(() => ({
  row: { flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: toneBg('#ffffff'), borderRadius: 14, padding: 11, borderWidth: 1, borderColor: theme.colors.bg.divider },
  icon: { width: 34, height: 34, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.bg.surfaceHover },
  name: { fontSize: 14, fontWeight: '700', color: theme.colors.text.display },
  sub: { fontSize: 12, color: theme.colors.text.muted, marginTop: 1 },
  desc: { fontSize: 12, color: theme.colors.text.body, marginTop: 3, lineHeight: 17 },
  saveBtn: { width: 32, height: 32, borderRadius: 16, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  removeBtn: { width: 32, height: 32, borderRadius: 16, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.semantic.dangerBg, borderWidth: 1, borderColor: theme.colors.semantic.dangerBorder },
  status: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 4 },
  statusText: { flex: 1, fontSize: 12, fontWeight: '600' },
}));
