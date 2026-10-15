import React, { useRef, useState } from 'react';
import type { Contact } from '@es/types';
import { View, Pressable, ScrollView, ActivityIndicator, useWindowDimensions } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Animated from 'react-native-reanimated';
import { X, GitMerge, Check, RotateCcw, Layers } from 'lucide-react-native';
import { Text } from '../../../ui/Text';
import { theme, toneBg, makeStyles } from '../../../ui/theme';
import { useSheetMotion, SheetOverlay } from '../../../ui/SheetOverlay';
import { PressableScale } from '../../onboarding/components';
import { contactsFeature, getEditConflict, type ContactPatch, type EditConflict } from '../index';
import { errorMessage } from '../../../lib/api';

const FIELD_NAMES: Record<string, string> = {
  firstName: 'First name', lastName: 'Last name', company: 'Company', title: 'Title', email: 'Email',
  phone: 'Phone', notes: 'Notes', linkedin: 'LinkedIn', eventId: 'Event',
};

const show = (v: unknown) => (v == null || String(v).trim() === '' ? '(empty)' : String(v));

/** "Keep both" for notes: the server's text, then what this edit added to the notes it started from. */
export function mergeNotes(server: unknown, base: unknown, mine: unknown): string {
  const s = String(server ?? '').trimEnd();
  const b = String(base ?? '').trimEnd();
  const m = String(mine ?? '').trimEnd();
  const added = b && m.startsWith(b) ? m.slice(b.length).trim() : m.trim();
  if (!added || s.includes(added)) return s;
  return s ? `${s}\n\n${added}` : added;
}

/**
 * Wires a screen's contact save to the conflict sheet:
 *   catch (e) { if (conflict.handle(e, contact.id, patch)) return; …show the error… }
 * and render `conflict.sheet`. "Keep mine" / "Keep both" save with force and call onSaved;
 * "Use theirs" calls onUseTheirs with the server's contact and the conflicting fields (the
 * screen drops its edits to those fields; nothing is saved).
 */
type ConflictCallbacks = {
  onSaved: (contact: Contact & { pendingSync?: boolean }) => void;
  onUseTheirs: (server: Contact | null, fields: string[]) => void;
};

export function useEditConflict(opts: ConflictCallbacks) {
  const [state, setState] = useState<{
    contactId: string; patch: ContactPatch; conflict: EditConflict; extra?: Partial<ConflictCallbacks> | undefined;
  } | null>(null);
  const [busy, setBusy] = useState<'mine' | 'theirs' | 'both' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const optsRef = useRef(opts);
  optsRef.current = opts;

  /** `extra`: callbacks for the part of the screen that saved (e.g. the notes card), run after the screen's. */
  const handle = (e: unknown, contactId: string, patch: ContactPatch, extra?: Partial<ConflictCallbacks>): boolean => {
    const conflict = getEditConflict(e);
    if (!conflict) return false;
    setError(null);
    setBusy(null);
    setState({ contactId, patch, conflict, extra });
    return true;
  };

  const saveForced = async (kind: 'mine' | 'both') => {
    if (!state || busy) return;
    setBusy(kind);
    setError(null);
    let patch = state.patch;
    if (kind === 'both') {
      const n = state.conflict.conflicts.find((c) => c.field === 'notes');
      if (n) patch = { ...patch, notes: mergeNotes(n.server, n.base, n.mine) || null };
    }
    try {
      const saved = await contactsFeature.updateContact(state.contactId, patch, { force: true });
      setState(null);
      optsRef.current.onSaved(saved);
      state.extra?.onSaved?.(saved);
    } catch (err: any) {
      const unreachable = "Can't reach the server — nothing was saved.";
      setError(errorMessage(err, "Couldn't save. Please try again.", { offline: unreachable, unreachable }));
    } finally {
      setBusy(null);
    }
  };

  const takeTheirs = async () => {
    if (!state || busy) return;
    setBusy('theirs');
    let server = state.conflict.contact;
    if (server) contactsFeature.acceptServerCopy(server); // a change still queued on the phone wins
    else server = await contactsFeature.getContact(state.contactId).catch(() => null);
    const fields = state.conflict.conflicts.map((c) => String(c.field));
    setState(null);
    setBusy(null);
    optsRef.current.onUseTheirs(server, fields);
    state.extra?.onUseTheirs?.(server, fields);
  };

  const sheet = (
    <ConflictSheet
      visible={!!state}
      conflict={state?.conflict ?? null}
      busy={busy}
      error={error}
      onKeepMine={() => saveForced('mine')}
      onKeepBoth={() => saveForced('both')}
      onUseTheirs={takeTheirs}
      onClose={() => { if (!busy) setState(null); }}
    />
  );

  return { handle, sheet, open: !!state };
}

export type EditConflictHandler = ReturnType<typeof useEditConflict>;

interface Props {
  visible: boolean;
  conflict: EditConflict | null;
  busy?: 'mine' | 'theirs' | 'both' | null;
  error?: string | null;
  onKeepMine: () => void;
  onUseTheirs: () => void;
  /** Offered when the notes conflict: saves the other device's notes plus this edit's additions. */
  onKeepBoth?: (() => void) | undefined;
  onClose: () => void;
}

/**
 * "Changed somewhere else" sheet for a contact edit (in-screen overlay like QrDialog — not an RN
 * <Modal>). Lists each field changed elsewhere since the edit started, with what to keep.
 */
export function ConflictSheet({ visible, conflict, busy, error, onKeepMine, onUseTheirs, onKeepBoth, onClose }: Props) {
  const insets = useSafeAreaInsets();
  const { height: winH } = useWindowDimensions();
  // Slide-in, back button and slide-out: the shared sheet motion. Closing is blocked while saving.
  const motion = useSheetMotion({ open: visible && !!conflict, onClose, dismissible: !busy });
  // Keep the last conflict on screen while the sheet slides out (the parent clears it on close).
  const lastConflict = useRef(conflict);
  if (conflict) lastConflict.current = conflict;

  const shown = conflict ?? lastConflict.current;
  if (!motion.mounted || !shown) return null;
  const notesConflict = shown.conflicts.some((c) => c.field === 'notes');

  return (
    <SheetOverlay motion={motion} backdropColor="rgba(15,23,42,0.55)" zIndex={60}>
      <View style={[styles.wrap, { maxHeight: winH - Math.max(insets.top, 12) - 24 }]}>
        <Animated.View style={[styles.sheet, { paddingBottom: Math.max(insets.bottom, 16) }, motion.sheetStyle]}>
          <View style={styles.handleArea}><View style={styles.handle} /></View>
          <View style={styles.header}>
            <View style={styles.headerIcon}><GitMerge size={16} color="white" /></View>
            <View style={{ flex: 1 }}>
              <Text style={styles.headerTitle}>Changed somewhere else</Text>
              <Text style={styles.headerSub}>Nothing was saved yet. Pick what to keep.</Text>
            </View>
            <Pressable onPress={motion.dismiss} hitSlop={12} style={({ pressed }) => [styles.closeBtn, pressed && { opacity: 0.6 }]}
              accessibilityRole="button" accessibilityLabel="Close">
              <X size={18} color={theme.colors.text.body} />
            </Pressable>
          </View>

          <ScrollView style={{ flexShrink: 1 }} contentContainerStyle={{ paddingBottom: 4 }} showsVerticalScrollIndicator={false} bounces={false}>
            {shown.conflicts.map((c) => (
              <View key={String(c.field)} style={styles.fieldCard}>
                <Text style={styles.fieldName}>{FIELD_NAMES[String(c.field)] || String(c.field)}</Text>
                <View style={styles.row}>
                  <Text style={styles.label}>Changed elsewhere</Text>
                  <Text style={styles.value} numberOfLines={c.field === 'notes' ? 4 : 2}>{show(c.server)}</Text>
                </View>
                <View style={[styles.row, styles.rowMine]}>
                  <Text style={[styles.label, { color: theme.colors.brand[700] }]}>Yours</Text>
                  <Text style={styles.value} numberOfLines={c.field === 'notes' ? 4 : 2}>{show(c.mine)}</Text>
                </View>
              </View>
            ))}

            {error ? <Text style={styles.error}>{error}</Text> : null}

            <View style={styles.actions}>
              {notesConflict && onKeepBoth ? (
                <PressableScale onPress={onKeepBoth} disabled={!!busy} style={styles.primaryBtn} scaleTo={0.97}
                  leadingIcon={busy === 'both' ? <ActivityIndicator size="small" color="white" /> : <Layers size={16} color="white" />}>
                  <Text style={styles.primaryText}>Keep both notes</Text>
                </PressableScale>
              ) : null}
              <PressableScale onPress={onKeepMine} disabled={!!busy} style={notesConflict && onKeepBoth ? styles.ghostBtn : styles.primaryBtn} scaleTo={0.97}
                leadingIcon={busy === 'mine'
                  ? <ActivityIndicator size="small" color={notesConflict && onKeepBoth ? theme.colors.brand[700] : 'white'} />
                  : <Check size={16} color={notesConflict && onKeepBoth ? theme.colors.brand[700] : 'white'} />}>
                <Text style={notesConflict && onKeepBoth ? styles.ghostText : styles.primaryText}>Keep mine</Text>
              </PressableScale>
              <PressableScale onPress={onUseTheirs} disabled={!!busy} style={styles.ghostBtn} scaleTo={0.97}
                leadingIcon={busy === 'theirs' ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <RotateCcw size={16} color={theme.colors.brand[700]} />}>
                <Text style={styles.ghostText}>Use theirs</Text>
              </PressableScale>
            </View>
          </ScrollView>
        </Animated.View>
      </View>
    </SheetOverlay>
  );
}

const styles = makeStyles(() => ({
  wrap: { width: '100%' },
  sheet: { backgroundColor: toneBg('#ffffff'), borderTopLeftRadius: 28, borderTopRightRadius: 28, paddingHorizontal: 18, flexShrink: 1 },
  handleArea: { alignItems: 'center', paddingTop: 10, paddingBottom: 4 },
  handle: { width: 42, height: 5, borderRadius: 3, backgroundColor: toneBg('#cbd5e1') },
  header: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 8, marginBottom: 6 },
  headerIcon: { width: 32, height: 32, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600] },
  headerTitle: { fontSize: 17, fontWeight: '800', color: theme.colors.text.display },
  headerSub: { fontSize: 12, color: theme.colors.text.muted, marginTop: 1 },
  closeBtn: { width: 38, height: 38, borderRadius: 19, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.bg.surfaceHover },
  fieldCard: { borderRadius: 16, borderWidth: 1, borderColor: theme.colors.bg.divider, padding: 12, marginBottom: 10, gap: 8 },
  fieldName: { fontSize: 11, fontWeight: '800', letterSpacing: 1, color: theme.colors.text.muted, textTransform: 'uppercase' },
  row: { gap: 2, padding: 10, borderRadius: 12, backgroundColor: theme.colors.bg.surfaceHover },
  rowMine: { backgroundColor: theme.colors.brand[50] },
  label: { fontSize: 11, fontWeight: '700', color: theme.colors.text.muted },
  value: { fontSize: 14, color: theme.colors.text.display },
  error: { fontSize: 13, fontWeight: '600', color: theme.colors.semantic.danger, marginTop: 2, marginBottom: 4 },
  actions: { gap: 10, marginTop: 6 },
  primaryBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, height: 48, borderRadius: 999, backgroundColor: theme.colors.brand[600], ...theme.shadows.active },
  primaryText: { fontSize: 15, fontWeight: '800', color: 'white' },
  ghostBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, height: 48, borderRadius: 999, borderWidth: 1.5, borderColor: theme.colors.brand[200], backgroundColor: toneBg('#ffffff') },
  ghostText: { fontSize: 14, fontWeight: '800', color: theme.colors.brand[700] },
}));
