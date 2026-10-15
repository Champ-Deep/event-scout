import React, { useEffect, useState } from 'react';
import { View, StyleSheet, ActivityIndicator, Pressable } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import Animated, { FadeIn, FadeOut, ZoomIn, useSharedValue, useAnimatedStyle, interpolate, Extrapolation } from 'react-native-reanimated';
import type { Contact } from '@es/types';
import { Text } from '../../../../src/ui/Text';
import { theme, toneBg, makeStyles } from '../../../../src/ui/theme';
import { contactsFeature } from '../../../../src/features/contacts';
import { ShareContactPanel } from '../../../../src/features/contacts/components/ShareContactSheet';

/**
 * Share dialog, presented over the contact as a transparent modal *route* (see the contacts
 * Stack layout). It's a real screen, not an RN <Modal> — inside a Modal on Android the Share /
 * Copy link taps never reached the API — but it looks and behaves like a dialog: dimmed
 * backdrop (tap to close), sheet sliding up from the bottom.
 */
export default function ShareContactScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const [contact, setContact] = useState<Contact | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!id) { setError('No contact was chosen to share.'); return; }
    contactsFeature.getContact(id)
      .then((c) => (c ? setContact(c) : setError('This contact no longer exists.')))
      .catch(() => setError("Couldn't load this contact. Check your connection."));
  }, [id]);

  // Back to the contact; if there's nothing to go back to, open the contacts list instead.
  const close = () => (router.canGoBack() ? router.back() : router.replace('/(app)/contacts' as any));
  // The backdrop lightens as the dialog is dragged down (swipe to close).
  const dragY = useSharedValue(0);
  const dimStyle = useAnimatedStyle(() => ({ opacity: interpolate(dragY.value, [0, 400], [1, 0.15], Extrapolation.CLAMP) }));

  return (
    <View style={styles.root}>
      <Animated.View entering={FadeIn.duration(220)} exiting={FadeOut.duration(160)} style={StyleSheet.absoluteFill}>
        <Animated.View style={[StyleSheet.absoluteFill, dimStyle]}>
          <Pressable style={styles.backdrop} onPress={close} accessibilityLabel="Close share" />
        </Animated.View>
      </Animated.View>
      {contact ? (
        <ShareContactPanel contact={contact} onClose={close} dragY={dragY} />
      ) : (
        <Animated.View entering={ZoomIn.duration(260)} style={styles.loadingCard}>
          {error ? (
            <>
              <Text style={styles.error}>{error}</Text>
              <Pressable onPress={close} hitSlop={8}><Text style={styles.link}>Close</Text></Pressable>
            </>
          ) : (
            <ActivityIndicator color={theme.colors.brand[600]} />
          )}
        </Animated.View>
      )}
    </View>
  );
}

const styles = makeStyles(() => ({
  root: { flex: 1, justifyContent: 'flex-end', backgroundColor: 'transparent' },
  backdrop: { flex: 1, backgroundColor: 'rgba(15,23,42,0.5)' },
  loadingCard: { margin: 24, marginBottom: 60, padding: 24, borderRadius: 24, backgroundColor: toneBg('#ffffff'), alignItems: 'center', gap: 10 },
  error: { fontSize: 15, color: theme.colors.text.body, textAlign: 'center' },
  link: { fontSize: 14, fontWeight: '700', color: theme.colors.brand[600] },
}));
