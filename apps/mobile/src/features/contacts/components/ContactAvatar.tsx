import React from 'react';
import { View, Image, Pressable, ActivityIndicator } from 'react-native';
import Animated, { FadeIn, ZoomIn } from 'react-native-reanimated';
import { Camera } from 'lucide-react-native';
import { Text } from '../../../ui/Text';
import { theme, toneBg, makeStyles } from '../../../ui/theme';

interface ContactAvatarProps {
  firstName?: string | null | undefined;
  lastName?: string | null | undefined;
  // A real picture (scanned card thumbnail or one the user set), from contact.photoUrl.
  photoUrl?: string | null | undefined;
  size?: number | undefined;
  onPress?: (() => void) | undefined;
  /** Shows a camera badge to signal the photo can be changed. */
  editable?: boolean | undefined;
  busy?: boolean | undefined;
}

export function initialsFor(firstName?: string | null, lastName?: string | null) {
  const a = (firstName || '').trim()[0] || '';
  const b = (lastName || '').trim()[0] || '';
  return (a + b).toUpperCase() || '?';
}

export function ContactAvatar({ firstName, lastName, photoUrl, size = 64, onPress, editable, busy }: ContactAvatarProps) {
  const radius = size / 2;
  const body = (
    <View style={{ width: size, height: size }}>
      {photoUrl ? (
        <Animated.View key={photoUrl.slice(-24)} entering={FadeIn.duration(250)}>
          <Image
            source={{ uri: photoUrl }}
            style={{ width: size, height: size, borderRadius: radius, backgroundColor: theme.colors.brand[50] }}
            accessibilityIgnoresInvertColors
          />
        </Animated.View>
      ) : (
        <View style={[styles.initials, { width: size, height: size, borderRadius: radius }]}>
          <Text style={[styles.initialsText, { fontSize: Math.round(size * 0.36) }]}>{initialsFor(firstName, lastName)}</Text>
        </View>
      )}
      {busy ? (
        <View style={[styles.busy, { borderRadius: radius }]}>
          <ActivityIndicator color="white" />
        </View>
      ) : null}
      {editable && !busy ? (
        <Animated.View entering={ZoomIn.duration(260)} style={styles.badge}>
          <Camera size={13} color="white" strokeWidth={2.5} />
        </Animated.View>
      ) : null}
    </View>
  );

  if (!onPress) return body;
  return (
    <Pressable
      onPress={onPress}
      disabled={busy}
      accessibilityRole="button"
      accessibilityLabel={photoUrl ? 'Change contact photo' : 'Add contact photo'}
      hitSlop={6}
    >
      {body}
    </Pressable>
  );
}

const styles = makeStyles(() => ({
  initials: { backgroundColor: theme.colors.brand[100], alignItems: 'center', justifyContent: 'center' },
  initialsText: { fontWeight: '800', color: theme.colors.brand[800] },
  busy: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(15,23,42,0.45)', alignItems: 'center', justifyContent: 'center' },
  badge: {
    position: 'absolute',
    right: -2,
    bottom: -2,
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: theme.colors.brand[600],
    borderWidth: 2,
    borderColor: toneBg('#ffffff'),
    alignItems: 'center',
    justifyContent: 'center',
  },
}));
