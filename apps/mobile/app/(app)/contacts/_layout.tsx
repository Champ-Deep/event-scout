import React from 'react';
import { Stack } from 'expo-router';
import { theme } from '../../../src/ui/theme';

// The list is always where the Contacts tab starts. Without this, the first <Stack.Screen>
// declared below ("[id]/share") became the initial route, so opening Contacts showed an empty
// share dialog that never loaded.
export const unstable_settings = { initialRouteName: 'index' };

/**
 * Contacts is a stack inside its tab: list → detail → edit / Recently deleted.
 * As separate tab routes, "Back" jumped to the first tab (Scout) instead of the list,
 * and the Contacts tab lost its highlight on the detail/edit screens.
 */
export default function ContactsLayout() {
  return (
    <Stack
      screenOptions={{
        headerShown: false,
        animation: 'slide_from_right',
        contentStyle: { backgroundColor: theme.colors.bg.app },
      }}
    >
      <Stack.Screen name="index" />
      {/* Share opens as a dialog over the contact (transparent modal route, not a full page). */}
      <Stack.Screen
        name="[id]/share"
        options={{
          presentation: 'transparentModal',
          animation: 'fade',
          contentStyle: { backgroundColor: 'transparent' },
        }}
      />
    </Stack>
  );
}
