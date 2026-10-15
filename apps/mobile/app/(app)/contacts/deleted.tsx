import React, { useCallback, useState } from 'react';
import { View, FlatList, Pressable, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useLocalSearchParams } from 'expo-router';
import { useReturnTo } from '../../../src/features/contacts/open';
import Animated, { FadeInDown, FadeOut, FadeInUp, LinearTransition } from 'react-native-reanimated';
import { ArrowLeft, RotateCcw, Trash2, Clock, Trash, CheckCircle2 } from 'lucide-react-native';
import { Text } from '../../../src/ui/Text';
import { theme, makeStyles } from '../../../src/ui/theme';
import { api, errorMessage, type DeletedContact } from '../../../src/lib/api';
import { contactsFeature } from '../../../src/features/contacts';
import { ContactAvatar } from '../../../src/features/contacts/components/ContactAvatar';
import { OptionSheet } from '../../../src/features/contacts/components/OptionSheet';
import { PressableScale } from '../../../src/features/onboarding/components';

function splitName(name: string) {
  const parts = name.trim().split(/\s+/);
  return { first: parts[0] || '', last: parts.slice(1).join(' ') };
}

export default function RecentlyDeletedScreen() {
  // Opened from Scout / Me (`from`): Back returns there, not to the Contacts list.
  const { from } = useLocalSearchParams<{ from?: string }>();
  const leave = useReturnTo(from);
  const [items, setItems] = useState<DeletedContact[]>([]);
  const [retention, setRetention] = useState(30);
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [confirm, setConfirm] = useState<{ kind: 'one'; item: DeletedContact } | { kind: 'all' } | null>(null);

  const load = useCallback(async () => {
    setError('');
    try {
      const res = await api.listDeletedContacts();
      setItems(res?.contacts ?? []);
      setRetention(res?.retentionDays ?? 30);
    } catch (e: any) {
      setError(errorMessage(e, "Couldn't load Recently deleted."));
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  const restore = async (item: DeletedContact) => {
    setBusyId(item.id);
    setError('');
    try {
      await contactsFeature.restoreContact(item.id);
      setItems((prev) => prev.filter((i) => i.id !== item.id));
      setNotice(`${item.name} restored`);
    } catch (e: any) {
      setError(errorMessage(e, "Couldn't restore this contact."));
    } finally {
      setBusyId(null);
    }
  };

  const purge = async (item: DeletedContact) => {
    setBusyId(item.id);
    setError('');
    try {
      await api.purgeDeletedContact(item.id);
      setItems((prev) => prev.filter((i) => i.id !== item.id));
      setNotice(`${item.name} permanently deleted`);
    } catch (e: any) {
      setError(errorMessage(e, "Couldn't delete this contact."));
    } finally {
      setBusyId(null);
    }
  };

  const emptyAll = async () => {
    setBusyId('all');
    setError('');
    try {
      const res = await api.emptyDeletedContacts();
      setItems([]);
      setNotice(`${res?.deleted ?? 0} contact${res?.deleted === 1 ? '' : 's'} permanently deleted`);
    } catch (e: any) {
      setError(errorMessage(e, "Couldn't empty Recently deleted."));
    } finally {
      setBusyId(null);
    }
  };

  const renderItem = ({ item, index }: { item: DeletedContact; index: number }) => {
    const { first, last } = splitName(item.name);
    const busy = busyId === item.id || busyId === 'all';
    const urgent = item.daysLeft <= 3;
    return (
      <Animated.View
        entering={FadeInDown.delay(Math.min(index, 10) * 45).duration(260)}
        exiting={FadeOut.duration(180)}
        layout={LinearTransition.duration(260)}
        style={styles.card}
      >
        <View style={{ opacity: 0.75 }}>
          <ContactAvatar firstName={first} lastName={last} photoUrl={item.photoUrl} size={44} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={styles.name} numberOfLines={1}>{item.name}</Text>
          <Text style={styles.sub} numberOfLines={1}>{[item.title, item.company].filter(Boolean).join(' · ') || 'No company'}</Text>
          {item.reason ? <Text style={styles.reason} numberOfLines={1}>{item.reason}</Text> : null}
          <View style={[styles.daysChip, urgent && styles.daysChipUrgent]}>
            <Clock size={11} color={urgent ? theme.colors.semantic.danger : theme.colors.text.muted} />
            <Text style={[styles.daysText, urgent && styles.daysTextUrgent]}>
              {item.daysLeft <= 0 ? 'Deleting today' : `${item.daysLeft} day${item.daysLeft === 1 ? '' : 's'} left`}
            </Text>
          </View>
        </View>
        <View style={styles.actions}>
          <PressableScale
            style={styles.restoreBtn}
            onPress={() => restore(item)}
            disabled={busy}
            scaleTo={0.92}
            leadingIcon={busyId === item.id ? <ActivityIndicator size="small" color="white" /> : <RotateCcw size={14} color="white" strokeWidth={2.5} />}
            accessibilityRole="button"
            accessibilityLabel={`Restore ${item.name}`}
          >
            <Text style={styles.restoreText}>Restore</Text>
          </PressableScale>
          <Pressable
            onPress={() => setConfirm({ kind: 'one', item })}
            disabled={busy}
            hitSlop={8}
            style={styles.purgeBtn}
            accessibilityRole="button"
            accessibilityLabel={`Delete ${item.name} forever`}
          >
            <Trash2 size={15} color={theme.colors.semantic.danger} />
          </Pressable>
        </View>
      </Animated.View>
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.header}>
        <PressableScale style={styles.iconBtn} onPress={leave} scaleTo={0.9} accessibilityRole="button" accessibilityLabel="Back">
          <ArrowLeft size={20} color={theme.colors.text.display} />
        </PressableScale>
        <Text style={styles.headerTitle}>Recently deleted</Text>
        {items.length > 0 ? (
          <Pressable onPress={() => setConfirm({ kind: 'all' })} disabled={busyId === 'all'} hitSlop={8} accessibilityRole="button">
            <Text style={styles.emptyAll}>{busyId === 'all' ? 'Emptying…' : 'Empty'}</Text>
          </Pressable>
        ) : <View style={{ width: 48 }} />}
      </View>

      <Animated.View entering={FadeInDown.duration(260)} style={styles.info}>
        <Clock size={16} color={theme.colors.brand[700]} />
        <Text style={styles.infoText}>
          Deleted contacts stay here for {retention} days, then they're removed for good. Restore brings back the photo, notes and details.
        </Text>
      </Animated.View>

      {notice ? (
        <Animated.View key={notice} entering={FadeInDown.duration(260)} style={styles.notice}>
          <CheckCircle2 size={15} color={theme.colors.semantic.successText} />
          <Text style={styles.noticeText}>{notice}</Text>
        </Animated.View>
      ) : null}
      {error ? <Text style={styles.error}>{error}</Text> : null}

      {loading ? (
        <View style={styles.center}><ActivityIndicator color={theme.colors.brand[600]} /></View>
      ) : (
        <FlatList
          data={items}
          keyExtractor={(i) => i.id}
          renderItem={renderItem}
          contentContainerStyle={styles.list}
          refreshing={false}
          onRefresh={load}
          ListEmptyComponent={
            <Animated.View entering={FadeInUp.duration(260)} style={styles.center}>
              <View style={styles.emptyIcon}><Trash size={26} color={theme.colors.brand[600]} /></View>
              <Text style={styles.emptyTitle}>Nothing here</Text>
              <Text style={styles.emptySub}>Contacts you delete show up here for {retention} days.</Text>
            </Animated.View>
          }
        />
      )}

      <OptionSheet
        visible={confirm !== null}
        title={confirm?.kind === 'all' ? 'Empty Recently deleted?' : `Delete ${confirm?.kind === 'one' ? confirm.item.name : ''} forever?`}
        message={confirm?.kind === 'all'
          ? `All ${items.length} contact${items.length === 1 ? '' : 's'} will be permanently deleted. This can't be undone.`
          : "It will be permanently deleted. This can't be undone."}
        options={[{
          label: confirm?.kind === 'all' ? 'Delete all forever' : 'Delete forever',
          icon: <Trash2 size={20} color={theme.colors.semantic.danger} />,
          destructive: true,
          onPress: () => { if (confirm?.kind === 'all') emptyAll(); else if (confirm?.kind === 'one') purge(confirm.item); },
        }]}
        onClose={() => setConfirm(null)}
      />
    </SafeAreaView>
  );
}

const styles = makeStyles(() => ({
  safeArea: { flex: 1, backgroundColor: theme.colors.bg.app },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 10 },
  iconBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: theme.colors.bg.surface, borderWidth: 1, borderColor: theme.colors.bg.divider, alignItems: 'center', justifyContent: 'center' },
  headerTitle: { fontSize: 18, fontWeight: '800', color: theme.colors.text.display },
  emptyAll: { fontSize: 14, fontWeight: '700', color: theme.colors.semantic.danger, width: 64, textAlign: 'right' },
  info: { flexDirection: 'row', gap: 10, alignItems: 'flex-start', marginHorizontal: 16, marginBottom: 10, backgroundColor: theme.colors.brand[50], borderRadius: 14, padding: 12 },
  infoText: { flex: 1, fontSize: 13, color: theme.colors.brand[800], lineHeight: 18 },
  notice: { flexDirection: 'row', alignItems: 'center', gap: 6, marginHorizontal: 16, marginBottom: 8 },
  noticeText: { fontSize: 13, color: theme.colors.semantic.successText, fontWeight: '600' },
  error: { fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600', marginHorizontal: 16, marginBottom: 8 },
  list: { paddingHorizontal: 16, paddingBottom: 60, flexGrow: 1 },
  card: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: theme.colors.bg.surface, borderRadius: 16, padding: 12, marginBottom: 8, borderWidth: 1, borderColor: theme.colors.bg.divider },
  name: { fontSize: 15, fontWeight: '700', color: theme.colors.text.display },
  sub: { fontSize: 12, color: theme.colors.text.body, marginTop: 1 },
  reason: { fontSize: 12, color: theme.colors.brand[700], fontWeight: '600', marginTop: 2 },
  daysChip: { flexDirection: 'row', alignItems: 'center', gap: 4, alignSelf: 'flex-start', marginTop: 6, backgroundColor: theme.colors.bg.surfaceHover, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 999 },
  daysChipUrgent: { backgroundColor: theme.colors.semantic.dangerBg },
  daysText: { fontSize: 11, fontWeight: '700', color: theme.colors.text.muted },
  daysTextUrgent: { color: theme.colors.semantic.danger },
  actions: { alignItems: 'flex-end', gap: 8 },
  restoreBtn: { flexDirection: 'row', alignItems: 'center', gap: 5, backgroundColor: theme.colors.brand[600], paddingHorizontal: 12, paddingVertical: 7, borderRadius: 999 },
  restoreText: { color: 'white', fontSize: 12, fontWeight: '800' },
  purgeBtn: { width: 30, height: 30, borderRadius: 15, backgroundColor: theme.colors.semantic.dangerBg, alignItems: 'center', justifyContent: 'center' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingTop: 60, gap: 8, paddingHorizontal: 24 },
  emptyIcon: { width: 60, height: 60, borderRadius: 20, backgroundColor: theme.colors.brand[50], alignItems: 'center', justifyContent: 'center' },
  emptyTitle: { fontSize: 17, fontWeight: '800', color: theme.colors.text.display },
  emptySub: { fontSize: 14, color: theme.colors.text.muted, textAlign: 'center' },
}));
