import React, { useEffect, useRef, useState } from 'react';
import { View, TextInput, Pressable, ActivityIndicator } from 'react-native';
import Animated, {
  FadeIn, FadeInDown, FadeOut, ZoomIn, LinearTransition,
  useSharedValue, useAnimatedStyle, withSequence, withTiming, interpolateColor,
} from 'react-native-reanimated';
import { StickyNote, Pencil, Plus, Check, X, Clock, ListChecks, CalendarDays, AlertCircle, RotateCw } from 'lucide-react-native';
import type { Contact } from '@es/types';
import { Text } from '../../../ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../../ui/theme';
import { PressableScale } from '../../onboarding/components';
import { errorMessage, type ErrorMessageOptions } from '../../../lib/api';
import { contactsFeature } from '..';
import type { EditConflictHandler } from './ConflictSheet';


export const NOTES_MAX = 20000; // matches NOTES_MAX_CHARS on the backend

interface NotesCardProps {
  contactId: string;
  notes: string | null | undefined;
  eventName?: string | null | undefined;
  entering?: any;
  onSaved: (contact: Contact) => void;
  /** Lets the screen guard navigation and scroll the editor above the keyboard. */
  onEditingChange?: ((editing: boolean, dirty: boolean) => void) | undefined;
  /** Called when the editor gets focus with the Save row, so the screen can keep it above the keyboard. */
  onFocusEditor?: ((saveRow: View | null) => void) | undefined;
  /** The screen's conflict sheet (useEditConflict): shown when the notes were changed elsewhere meanwhile. */
  conflict?: EditConflictHandler | undefined;
}

function stamp() {
  return new Date().toLocaleString([], { day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' });
}

// How a failed note save is worded (see errorMessage).
const NOTE_ERRORS: ErrorMessageOptions = {
  offline: "You're offline — the note wasn't saved. Your text is still here.",
  timeout: 'Saving took too long. Please try again.',
  unreachable: "Can't reach the server — the note wasn't saved. Your text is still here.",
};
const NOTE_FAILED = "Couldn't save the note. Please try again.";

/**
 * Notes on the contact page, edited in place (PATCH /api/contacts/:id { notes }).
 * Previously "Add"/"Edit" jumped to the full edit screen.
 */
export function NotesCard({ contactId, notes, eventName, entering, onSaved, onEditingChange, onFocusEditor, conflict }: NotesCardProps) {
  const saved = notes || '';
  // 'add' = write a NEW note (appended with today's date; existing notes untouched)
  // 'edit' = change or delete anything in the existing notes. Tapping the notes does nothing.
  const [mode, setMode] = useState<'add' | 'edit' | null>(null);
  const editing = mode !== null;
  const [draft, setDraft] = useState(saved);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [justSaved, setJustSaved] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const inputRef = useRef<TextInput>(null);
  const footerRef = useRef<View>(null);
  const selection = useRef({ start: saved.length, end: saved.length });
  // The notes as they were when "Edit all" opened: the base the server checks for a change made elsewhere.
  const editBaseRef = useRef<string | null>(null);

  const dirty = mode === 'add' ? draft.trim().length > 0 : mode === 'edit' && draft.trimEnd() !== saved.trimEnd();
  useEffect(() => { onEditingChange?.(editing, dirty); }, [editing, dirty]);

  // Keep the draft in sync when the contact reloads (e.g. after coming back from the edit screen).
  useEffect(() => { if (!editing) setDraft(saved); }, [saved, editing]);

  // Save flash: the card border glows green for a moment.
  const flash = useSharedValue(0);
  const focus = useSharedValue(0);
  const DIVIDER = theme.colors.bg.divider; // plain strings for the animation (a worklet keeps its first copy of the theme)
  const SUCCESS = theme.colors.semantic.success;
  const BRAND_500 = theme.colors.brand[500];
  const HOVER = theme.colors.bg.surfaceHover;
  const cardStyle = useAnimatedStyle(() => ({
    borderColor: interpolateColor(flash.value, [0, 1], [DIVIDER, SUCCESS]),
  }));
  const BG_FFFFFF = toneBg('#ffffff'); // current mode's colour, read on the JS thread (worklets can't call toneBg)
  const inputBoxStyle = useAnimatedStyle(() => ({
    borderColor: interpolateColor(focus.value, [0, 1], [DIVIDER, BRAND_500]),
    backgroundColor: interpolateColor(focus.value, [0, 1], [HOVER, BG_FFFFFF]),
  }));
  const shake = useSharedValue(0);
  const shakeStyle = useAnimatedStyle(() => ({ transform: [{ translateX: shake.value }] }));

  useEffect(() => {
    if (!justSaved) return;
    const t = setTimeout(() => setJustSaved(false), 2200);
    return () => clearTimeout(t);
  }, [justSaved]);

  const open = (next: 'add' | 'edit') => {
    setError('');
    const text = next === 'add' ? '' : saved;
    editBaseRef.current = notes ?? null;
    setDraft(text);
    selection.current = { start: text.length, end: text.length };
    setMode(next);
    setTimeout(() => inputRef.current?.focus(), 250); // onFocus below keeps it above the keyboard
  };

  const cancel = () => {
    setDraft(saved);
    setMode(null);
    setError('');
  };

  const insert = (text: string) => {
    const { start, end } = selection.current;
    const before = draft.slice(0, start);
    const after = draft.slice(end);
    const needsBreak = before.length > 0 && !before.endsWith('\n') ? '\n' : '';
    const next = `${before}${needsBreak}${text}${after}`;
    if (next.length > NOTES_MAX) return;
    setDraft(next);
    const pos = before.length + needsBreak.length + text.length;
    selection.current = { start: pos, end: pos };
    inputRef.current?.focus();
  };

  const save = async () => {
    if (saving) return;
    if (!dirty) { setMode(null); return; }
    // A new note goes below the existing ones, stamped with when it was written.
    const entry = `${stamp()} — ${draft.trim()}`;
    const next = mode === 'add' ? (saved.trim() ? `${saved.trimEnd()}\n\n${entry}` : entry) : draft.trimEnd();
    if (next.length > NOTES_MAX) {
      setError(`Notes are too long (max ${NOTES_MAX.toLocaleString()} characters).`);
      shake.value = withSequence(withTiming(-6, { duration: 50 }), withTiming(6, { duration: 50 }), withTiming(-4, { duration: 50 }), withTiming(0, { duration: 50 }));
      return;
    }
    setSaving(true);
    setError('');
    // base: a new note is added to the notes shown now; "Edit all" started from the notes at open.
    const patch = { notes: next || null };
    const base = { notes: mode === 'add' ? (notes ?? null) : editBaseRef.current };
    const done = () => {
      setMode(null);
      setJustSaved(true);
      flash.value = withSequence(withTiming(1, { duration: 180 }), withTiming(0, { duration: 900 }));
    };
    try {
      const updated = await contactsFeature.updateContact(contactId, patch, { base });
      onSaved(updated);
      done();
    } catch (e: any) {
      // Changed elsewhere meanwhile: the sheet offers keep mine / use theirs / keep both.
      if (conflict?.handle(e, contactId, patch, {
        onSaved: (updated) => { onSaved(updated); done(); },
        onUseTheirs: () => { setMode(null); setError(''); },
      })) return;
      setError(errorMessage(e, NOTE_FAILED, NOTE_ERRORS));
      shake.value = withSequence(withTiming(-6, { duration: 50 }), withTiming(6, { duration: 50 }), withTiming(-4, { duration: 50 }), withTiming(0, { duration: 50 }));
    } finally {
      setSaving(false);
    }
  };

  const long = saved.length > 360 || saved.split('\n').length > 8;
  const words = draft.trim() ? draft.trim().split(/\s+/).length : 0;
  const nearLimit = draft.length > NOTES_MAX * 0.9;

  return (
    <Animated.View entering={entering} layout={LinearTransition.duration(260)} style={[styles.card, cardStyle]}>
      <View style={styles.header}>
        <View style={styles.icon}><StickyNote size={16} color={theme.colors.brand[700]} /></View>
        <Text style={styles.title}>NOTES</Text>
        {justSaved && (
          <Animated.View entering={ZoomIn.duration(260)} exiting={FadeOut.duration(250)} style={styles.savedPill}>
            <Check size={12} color={theme.colors.semantic.successText} />
            <Text style={styles.savedText}>Saved</Text>
          </Animated.View>
        )}
        {!editing && (
          <Animated.View entering={FadeIn} exiting={FadeOut.duration(120)} style={styles.headerActions}>
            {saved ? (
              <>
                <PressableScale onPress={() => open('edit')} scaleTo={0.9} style={styles.headerBtn} accessibilityLabel="Edit existing notes"
                  leadingIcon={<Pencil size={12} color={theme.colors.brand[700]} />}>
                  <Text style={styles.headerBtnText}>Edit all</Text>
                </PressableScale>
                <PressableScale onPress={() => open('add')} scaleTo={0.9} style={[styles.headerBtn, styles.headerBtnPrimary]} accessibilityLabel="Write a new note"
                  leadingIcon={<Plus size={13} color="white" />}>
                  <Text style={[styles.headerBtnText, { color: 'white' }]}>New note</Text>
                </PressableScale>
              </>
            ) : null}
          </Animated.View>
        )}
      </View>

      {editing ? (
        // Entering animation on the wrapper, shake on the inner view (both animate transform).
        <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(120)}>
        <Animated.View style={shakeStyle}>
          <Animated.View key={mode} entering={FadeInDown.duration(260)} style={[styles.modeBanner, mode === 'edit' && styles.modeBannerEdit]}>
            {mode === 'add' ? <Plus size={14} color={theme.colors.semantic.successText} /> : <Pencil size={13} color={toneFg('#b45309')} />}
            <View style={{ flex: 1 }}>
              <Text style={[styles.modeTitle, mode === 'edit' && { color: toneFg('#b45309') }]}>{mode === 'add' ? 'NEW NOTE' : 'EDITING ALL NOTES'}</Text>
              <Text style={styles.modeSub}>
                {mode === 'add'
                  ? (saved ? `Added below your notes, dated ${stamp()}` : `Saved with today's date (${stamp()})`)
                  : 'Change or delete anything you wrote before.'}
              </Text>
            </View>
          </Animated.View>
          {mode === 'add' && saved ? (
            <Animated.View entering={FadeIn.delay(80)} style={styles.soFar}>
              <Text style={styles.soFarLabel}>YOUR NOTES SO FAR</Text>
              <Text style={styles.soFarText} numberOfLines={3}>{saved}</Text>
            </Animated.View>
          ) : null}
          <Animated.View style={[styles.inputBox, inputBoxStyle]}>
            <TextInput
              ref={inputRef}
              value={draft}
              onChangeText={(t) => { setDraft(t); if (error) setError(''); }}
              onSelectionChange={(e) => { selection.current = e.nativeEvent.selection; }}
              placeholder={mode === 'add' ? 'Write a new note — what happened, next steps…' : 'What did you talk about? Needs, budget, next steps…'}
              placeholderTextColor={theme.colors.text.muted}
              multiline
              maxLength={NOTES_MAX}
              editable={!saving}
              textAlignVertical="top"
              style={styles.input}
              onFocus={() => { focus.value = withTiming(1, { duration: 180 }); onFocusEditor?.(footerRef.current); }}
              onBlur={() => { focus.value = withTiming(0, { duration: 180 }); }}
            />
          </Animated.View>

          <Animated.View entering={FadeInDown.delay(80).duration(260)} style={styles.chips}>
            {mode === 'edit' ? (
              <PressableScale onPress={() => insert(`${stamp()} — `)} scaleTo={0.92} style={styles.chip}
                leadingIcon={<Clock size={12} color={theme.colors.brand[700]} />}>
                <Text style={styles.chipText}>Timestamp</Text>
              </PressableScale>
            ) : null}
            <PressableScale onPress={() => insert('Next step: ')} scaleTo={0.92} style={styles.chip}
              leadingIcon={<ListChecks size={12} color={theme.colors.brand[700]} />}>
              <Text style={styles.chipText}>Next step</Text>
            </PressableScale>
            {eventName ? (
              <PressableScale onPress={() => insert(`Met at ${eventName}. `)} scaleTo={0.92} style={styles.chip}
                leadingIcon={<CalendarDays size={12} color={theme.colors.brand[700]} />}>
                <Text style={styles.chipText} numberOfLines={1}>Met at {eventName}</Text>
              </PressableScale>
            ) : null}
          </Animated.View>

          {error ? (
            <Animated.View entering={FadeInDown.duration(260)} style={styles.errorRow}>
              <AlertCircle size={14} color={theme.colors.semantic.danger} />
              <Text style={styles.errorText}>{error}</Text>
            </Animated.View>
          ) : null}

          <View ref={footerRef} collapsable={false} style={styles.footer}>
            <Text style={[styles.counter, nearLimit && { color: theme.colors.semantic.warning }]}>
              {nearLimit ? `${draft.length.toLocaleString()}/${NOTES_MAX.toLocaleString()}` : `${words} word${words === 1 ? '' : 's'}`}
            </Text>
            <View style={{ flex: 1 }} />
            <PressableScale onPress={cancel} disabled={saving} scaleTo={0.94} style={styles.cancelBtn}
              leadingIcon={<X size={14} color={theme.colors.text.body} />} accessibilityLabel="Cancel editing notes">
              <Text style={styles.cancelText}>Cancel</Text>
            </PressableScale>
            <PressableScale onPress={save} disabled={saving || !dirty} scaleTo={0.94}
              style={[styles.saveBtn, (!dirty || saving) && { opacity: 0.55 }]}
              leadingIcon={saving ? <ActivityIndicator size="small" color="white" /> : error ? <RotateCw size={14} color="white" /> : <Check size={14} color="white" />}
              accessibilityLabel="Save notes">
              <Text style={styles.saveText}>{saving ? 'Saving…' : error ? 'Retry' : mode === 'add' ? 'Add note' : 'Save changes'}</Text>
            </PressableScale>
          </View>
        </Animated.View>
        </Animated.View>
      ) : saved ? (
        <Animated.View entering={FadeIn.duration(220)}>
          {/* Read-only (long-press to copy). Use New note / Edit all to change them. */}
          <View style={[styles.entries, long && !expanded && styles.entriesCollapsed]}>
            {saved.split(/\n\s*\n/).filter((x) => x.trim()).map((entry, i) => {
              const m = /^((?=[^—\n]*\d:\d\d)[^—\n]{3,28}?)\s+—\s+([\s\S]*)$/.exec(entry.trim());
              return (
                <Animated.View key={i} entering={FadeInDown.delay(i * 50).duration(260)} style={styles.entry}>
                  <View style={styles.entryDot} />
                  <View style={{ flex: 1 }}>
                    {m ? <Text style={styles.entryStamp}>{m[1]}</Text> : null}
                    <Text style={styles.notes} selectable>{m ? m[2] : entry.trim()}</Text>
                  </View>
                </Animated.View>
              );
            })}
          </View>
          {long && (
            <Pressable onPress={() => setExpanded((x) => !x)} hitSlop={8} style={{ marginTop: 6 }}>
              <Text style={styles.link}>{expanded ? 'Show less' : 'Show more'}</Text>
            </Pressable>
          )}
        </Animated.View>
      ) : (
        <Animated.View entering={FadeIn.duration(220)} style={styles.empty}>
          <Text style={styles.muted}>No notes yet. Jot down what you talked about and the follow-ups.</Text>
          <PressableScale onPress={() => open('add')} scaleTo={0.94} style={styles.addBtn}
            leadingIcon={<Plus size={15} color="white" />} accessibilityLabel="Add a note">
            <Text style={styles.addText}>Add a note</Text>
          </PressableScale>
        </Animated.View>
      )}
    </Animated.View>
  );
}

const styles = makeStyles(() => ({
  card: {
    backgroundColor: theme.colors.bg.surface, borderRadius: 20, padding: 16, marginBottom: 12,
    borderWidth: 1, borderColor: theme.colors.bg.divider, ...theme.shadows.card,
  },
  header: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 10, minHeight: 30 },
  icon: { width: 30, height: 30, borderRadius: 10, backgroundColor: theme.colors.brand[50], alignItems: 'center', justifyContent: 'center' },
  title: { flex: 1, fontSize: 11, fontWeight: '800', color: theme.colors.text.display, letterSpacing: 1 },
  headerActions: { flexDirection: 'row', gap: 6 },
  headerBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 10, paddingVertical: 5,
    borderRadius: 999, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100],
  },
  headerBtnText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  headerBtnPrimary: { backgroundColor: theme.colors.brand[600], borderColor: theme.colors.fill.brand700 },
  modeBanner: {
    flexDirection: 'row', alignItems: 'center', gap: 8, padding: 10, borderRadius: 12, marginBottom: 10,
    backgroundColor: theme.colors.semantic.successBg, borderWidth: 1, borderColor: toneBg('#a7f3d0'),
  },
  modeBannerEdit: { backgroundColor: toneBg('#fffbeb'), borderColor: toneBg('#fde68a') },
  modeTitle: { fontSize: 11, fontWeight: '800', letterSpacing: 0.8, color: theme.colors.semantic.successText },
  modeSub: { fontSize: 12, color: theme.colors.text.body, marginTop: 1 },
  soFar: { padding: 10, borderRadius: 12, backgroundColor: theme.colors.bg.surfaceHover, marginBottom: 10 },
  soFarLabel: { fontSize: 10, fontWeight: '800', letterSpacing: 0.8, color: theme.colors.text.muted, marginBottom: 3 },
  soFarText: { fontSize: 13, lineHeight: 18, color: theme.colors.text.muted },
  entries: { gap: 10 },
  entriesCollapsed: { maxHeight: 220, overflow: 'hidden' },
  entry: { flexDirection: 'row', gap: 10 },
  entryDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: theme.colors.brand[400], marginTop: 7 },
  entryStamp: { fontSize: 11, fontWeight: '800', color: theme.colors.brand[600], marginBottom: 1 },
  savedPill: {
    flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 8, paddingVertical: 4,
    borderRadius: 999, backgroundColor: theme.colors.semantic.successBg,
  },
  savedText: { fontSize: 11, fontWeight: '800', color: theme.colors.semantic.successText },
  notes: { fontSize: 15, color: theme.colors.text.display, lineHeight: 22 },
  link: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[600] },
  muted: { fontSize: 14, color: theme.colors.text.muted, lineHeight: 20 },
  empty: { gap: 12 },
  addBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 6, alignSelf: 'flex-start',
    backgroundColor: theme.colors.brand[600], paddingHorizontal: 14, paddingVertical: 9, borderRadius: 999,
  },
  addText: { color: 'white', fontSize: 13, fontWeight: '700' },
  inputBox: { borderWidth: 1.5, borderRadius: 14, paddingHorizontal: 12, paddingVertical: 8 },
  input: {
    minHeight: 110, maxHeight: 280, fontSize: 15, lineHeight: 22, color: theme.colors.text.display,
    fontFamily: theme.typography.fontFamily.sans, paddingTop: 4, paddingBottom: 4,
  },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 10 },
  chip: {
    flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 10, paddingVertical: 6,
    borderRadius: 999, backgroundColor: theme.colors.brand[50], maxWidth: 220,
  },
  chipText: { fontSize: 12, fontWeight: '600', color: theme.colors.brand[700], flexShrink: 1 },
  errorRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 6, marginTop: 10 },
  errorText: { flex: 1, fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600', lineHeight: 18 },
  footer: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 12 },
  counter: { fontSize: 12, color: theme.colors.text.muted },
  cancelBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 12, paddingVertical: 8,
    borderRadius: 999, borderWidth: 1, borderColor: theme.colors.bg.divider, backgroundColor: toneBg('#ffffff'),
  },
  cancelText: { fontSize: 13, fontWeight: '700', color: theme.colors.text.body },
  saveBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 5, paddingHorizontal: 14, paddingVertical: 8,
    borderRadius: 999, backgroundColor: theme.colors.brand[600],
  },
  saveText: { fontSize: 13, fontWeight: '700', color: 'white' },
}));
