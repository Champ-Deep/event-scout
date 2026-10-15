import React, { useRef, useState } from 'react';
import { View, TextInput, Pressable } from 'react-native';
import Animated, { FadeIn, FadeInDown, FadeOut, LinearTransition, ZoomIn, ZoomOut } from 'react-native-reanimated';
import { Check, Plus, X } from 'lucide-react-native';
import { Text } from '../../../ui/Text';
import { theme, toneBg, makeStyles } from '../../../ui/theme';
import { PressableScale } from '../../onboarding/components';

/* Small editors for the Me page's profile sheets (products, sales approach, event goals). */

// ─── ChoiceChips ──────────────────────────────────────────────────────────────

export interface Choice { value: string; label: string; hint?: string; icon?: React.ReactNode }

export function ChoiceChips({ options, selected, onToggle, multi = false }: {
  options: Choice[]; selected: string[]; onToggle: (value: string) => void; multi?: boolean;
}) {
  return (
    <View style={styles.chips} accessibilityRole={multi ? undefined : 'radiogroup'}>
      {options.map((o, i) => {
        const on = selected.includes(o.value);
        return (
          <Animated.View key={o.value} entering={ZoomIn.delay(i * 35).duration(220)} layout={LinearTransition.duration(220)}>
            <PressableScale onPress={() => onToggle(o.value)} scaleTo={0.93}
              style={[styles.chip, on && styles.chipOn]}
              accessibilityRole={multi ? 'checkbox' : 'radio'} accessibilityState={multi ? { checked: on } : { selected: on }}
              accessibilityLabel={o.hint ? `${o.label}: ${o.hint}` : o.label}>
              {on ? (
                <Animated.View entering={ZoomIn.duration(200)}><Check size={13} color="white" strokeWidth={3} /></Animated.View>
              ) : o.icon ?? null}
              <Text style={[styles.chipText, on && { color: 'white' }]}>{o.label}</Text>
            </PressableScale>
          </Animated.View>
        );
      })}
    </View>
  );
}

// ─── ListEditor ───────────────────────────────────────────────────────────────
// Longer one-line items (value propositions, event goals): rows with ✕, inline add.

export function ListEditor({ items, setItems, placeholder, addLabel, max = 10, maxLength = 200 }: {
  items: string[]; setItems: (next: string[]) => void; placeholder: string; addLabel: string; max?: number; maxLength?: number;
}) {
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState('');
  const [notice, setNotice] = useState('');
  const draftRef = useRef('');

  const commit = (close: boolean) => {
    const v = draftRef.current.replace(/\s+/g, ' ').trim();
    draftRef.current = '';
    setDraft('');
    if (v) {
      if (items.some((x) => x.toLowerCase() === v.toLowerCase())) setNotice('Already in the list');
      else setItems([...items, v]);
    }
    if (close || items.length + (v ? 1 : 0) >= max) setAdding(false);
  };

  return (
    <View>
      {items.map((item, i) => (
        <Animated.View key={item} entering={FadeInDown.duration(220)} exiting={ZoomOut.duration(150)} layout={LinearTransition.duration(240)} style={styles.listRow}>
          <View style={styles.bullet}><Text style={styles.bulletText}>{i + 1}</Text></View>
          <Text style={styles.listText}>{item}</Text>
          <Pressable onPress={() => setItems(items.filter((x) => x !== item))} hitSlop={10} accessibilityLabel={`Remove ${item}`}
            style={({ pressed }) => [styles.removeBtn, pressed && { opacity: 0.5 }]}>
            <X size={14} color={theme.colors.text.muted} />
          </Pressable>
        </Animated.View>
      ))}
      {adding ? (
        <Animated.View entering={FadeInDown.duration(220)} exiting={FadeOut.duration(120)} style={styles.inputRow}>
          <TextInput
            autoFocus value={draft} maxLength={maxLength} placeholder={placeholder} placeholderTextColor={theme.colors.text.muted}
            onChangeText={(t) => { draftRef.current = t; setDraft(t); if (notice) setNotice(''); }}
            onSubmitEditing={() => commit(false)} submitBehavior="submit" onBlur={() => commit(true)} returnKeyType="done"
            style={styles.input}
          />
          <PressableScale onPress={() => commit(true)} scaleTo={0.9} style={styles.confirm} accessibilityLabel="Add">
            <Check size={16} color="white" strokeWidth={3} />
          </PressableScale>
        </Animated.View>
      ) : items.length < max ? (
        <Animated.View layout={LinearTransition.duration(240)} entering={FadeIn.duration(180)}>
          <PressableScale onPress={() => setAdding(true)} scaleTo={0.95} style={styles.addBtn}
            leadingIcon={<Plus size={14} color={theme.colors.brand[700]} strokeWidth={2.5} />}>
            <Text style={styles.addText}>{addLabel}</Text>
          </PressableScale>
        </Animated.View>
      ) : (
        <Text style={styles.notice}>That's the maximum of {max}.</Text>
      )}
      {notice ? <Animated.Text entering={FadeIn} style={styles.notice}>{notice}</Animated.Text> : null}
    </View>
  );
}

// ─── PairListEditor ───────────────────────────────────────────────────────────
// Two-part items: objection → response, case study → result.

export type Pair = Record<string, string>;

export function PairListEditor({ items, setItems, keys, labels, placeholders, addLabel, max = 10 }: {
  items: Pair[]; setItems: (next: Pair[]) => void;
  keys: [string, string]; labels: [string, string]; placeholders: [string, string]; addLabel: string; max?: number;
}) {
  // Stable keys so rows animate correctly while their text changes.
  const ids = useRef<string[]>([]);
  while (ids.current.length < items.length) ids.current.push(Math.random().toString(36).slice(2));
  const update = (i: number, k: string, v: string) => setItems(items.map((it, j) => (j === i ? { ...it, [k]: v } : it)));
  const remove = (i: number) => { ids.current.splice(i, 1); setItems(items.filter((_, j) => j !== i)); };

  return (
    <View>
      {items.map((it, i) => (
        <Animated.View key={ids.current[i]} entering={FadeInDown.duration(220)} exiting={ZoomOut.duration(150)} layout={LinearTransition.duration(240)} style={styles.pairCard}>
          <View style={styles.pairHead}>
            <Text style={styles.pairLabel}>{labels[0]}</Text>
            <Pressable onPress={() => remove(i)} hitSlop={10} accessibilityLabel="Remove" style={({ pressed }) => [styles.removeBtn, pressed && { opacity: 0.5 }]}>
              <X size={14} color={theme.colors.text.muted} />
            </Pressable>
          </View>
          <TextInput value={it[keys[0]] || ''} onChangeText={(v) => update(i, keys[0], v)} placeholder={placeholders[0]}
            placeholderTextColor={theme.colors.text.muted} maxLength={400} multiline style={styles.pairInput} />
          <Text style={[styles.pairLabel, { marginTop: 8 }]}>{labels[1]}</Text>
          <TextInput value={it[keys[1]] || ''} onChangeText={(v) => update(i, keys[1], v)} placeholder={placeholders[1]}
            placeholderTextColor={theme.colors.text.muted} maxLength={400} multiline style={styles.pairInput} />
        </Animated.View>
      ))}
      {items.length < max ? (
        <Animated.View layout={LinearTransition.duration(240)}>
          <PressableScale onPress={() => setItems([...items, { [keys[0]]: '', [keys[1]]: '' }])} scaleTo={0.95} style={styles.addBtn}
            leadingIcon={<Plus size={14} color={theme.colors.brand[700]} strokeWidth={2.5} />}>
            <Text style={styles.addText}>{addLabel}</Text>
          </PressableScale>
        </Animated.View>
      ) : null}
    </View>
  );
}

const styles = makeStyles(() => ({
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: {
    flexDirection: 'row', alignItems: 'center', gap: 5, paddingHorizontal: 12, paddingVertical: 8, borderRadius: 999,
    backgroundColor: toneBg('#ffffff'), borderWidth: 1.5, borderColor: theme.colors.brand[200],
  },
  chipOn: { backgroundColor: theme.colors.brand[600], borderColor: theme.colors.fill.brand700 },
  chipText: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[700] },

  listRow: {
    flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: toneBg('#ffffff'), borderRadius: 12, paddingHorizontal: 10,
    paddingVertical: 9, marginBottom: 6, borderWidth: 1, borderColor: theme.colors.bg.divider,
  },
  bullet: { width: 22, height: 22, borderRadius: 11, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  bulletText: { fontSize: 11, fontWeight: '800', color: theme.colors.brand[700] },
  listText: { flex: 1, fontSize: 14, lineHeight: 19, color: theme.colors.text.display },
  removeBtn: { width: 26, height: 26, borderRadius: 13, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.bg.app },

  inputRow: { flexDirection: 'row', gap: 8, marginBottom: 6 },
  input: {
    flex: 1, minHeight: 44, borderRadius: 12, borderWidth: 1.5, borderColor: theme.colors.brand[300], paddingHorizontal: 12,
    fontSize: 14, color: theme.colors.text.display, backgroundColor: toneBg('#ffffff'),
  },
  confirm: { width: 44, height: 44, borderRadius: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600] },
  addBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 5, alignSelf: 'flex-start', paddingHorizontal: 12, paddingVertical: 8,
    borderRadius: 999, borderWidth: 1.5, borderStyle: 'dashed', borderColor: theme.colors.brand[300], marginTop: 2,
  },
  addText: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[700] },
  notice: { fontSize: 12, color: theme.colors.text.muted, marginTop: 4 },

  pairCard: { backgroundColor: toneBg('#ffffff'), borderRadius: 14, padding: 12, marginBottom: 8, borderWidth: 1, borderColor: theme.colors.bg.divider },
  pairHead: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  pairLabel: { fontSize: 10, fontWeight: '800', letterSpacing: 1, color: theme.colors.text.muted },
  pairInput: {
    marginTop: 4, minHeight: 40, borderRadius: 10, borderWidth: 1, borderColor: theme.colors.bg.divider, paddingHorizontal: 10,
    paddingVertical: 8, fontSize: 14, color: theme.colors.text.display, backgroundColor: theme.colors.bg.app, textAlignVertical: 'top',
  },
}));
