import React from 'react';
import { View } from 'react-native';
import { Tabs, Redirect } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { theme, toneFg, makeStyles } from '../../src/ui/theme';
import { Radar, Contact, Scan, Sparkles, SlidersHorizontal } from 'lucide-react-native';
import { useSession } from '../../src/providers/SessionProvider';
import { AutoSync } from '../../src/sync/AutoSync';
import { AnnouncementsLayer } from '../../src/features/broadcasts/AnnouncementsMenu';
import { ReminderLayer } from '../../src/features/me/components/ReminderLayer';
import { OverlayProvider } from '../../src/ui/OverlayHost';

export default function AppLayout() {
  const { isAuthenticated } = useSession();
  const insets = useSafeAreaInsets();
  const bottomPad = Math.max(insets.bottom, 8);

  if (!isAuthenticated) {
    return <Redirect href="/welcome" />;
  }

  return (
    // Window layer for overlays that must cover the tab bar too (action sheets, duplicate review,
    // LinkedIn disconnect): its host is drawn after everything below, and is empty when none is open.
    <OverlayProvider>
    {/* Uploads changes and scans saved offline as soon as the connection is back. */}
    <AutoSync />
    <Tabs
      // Back from a hidden screen (sync, scan review…) returns to where you came from,
      // not always to the first tab.
      backBehavior="history"
      screenOptions={{
        headerShown: false,
        // While typing, the keyboard covers the tab bar anyway (Android edge-to-edge): hide it so
        // screens get that space and their inputs / Save buttons sit right above the keyboard.
        tabBarHideOnKeyboard: true,
        // Sized from the phone's real bottom inset (gesture bar / 3-button nav), not a fixed 85px
        // that's too tall on some phones and sits under the system nav bar on others.
        tabBarStyle: {
          backgroundColor: theme.colors.bg.surface,
          borderTopColor: theme.colors.bg.divider,
          height: 60 + bottomPad,
          paddingBottom: bottomPad,
          paddingTop: 8,
        },
        tabBarActiveTintColor: theme.colors.brand[600],
        tabBarInactiveTintColor: theme.colors.text.muted,
        tabBarLabelStyle: {
          ...theme.typography.sizes.caption,
          marginTop: 4,
        }
      }}
    >
      <Tabs.Screen
        name="feed"
        options={{
          title: 'Scout',
          tabBarIcon: ({ color, size }) => <Radar color={color} size={size} />,
        }}
      />
      <Tabs.Screen
        name="contacts"
        options={{
          title: 'Contacts',
          tabBarIcon: ({ color, size }) => <Contact color={color} size={size} />,
        }}
      />
      <Tabs.Screen
        name="scan"
        options={{
          title: '',
          tabBarLabel: () => null,
          tabBarStyle: { display: 'none' },
          tabBarIcon: ({ focused }) => (
            <View style={styles.fabContainer}>
              <Scan color="white" size={28} />
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="ask"
        options={{
          title: 'Ask AI',
          tabBarIcon: ({ color, size }) => <Sparkles color={color} size={size} />,
        }}
      />
      <Tabs.Screen
        name="me"
        options={{
          title: 'Me',
          tabBarIcon: ({ color, size }) => <SlidersHorizontal color={color} size={size} />,
        }}
      />
      <Tabs.Screen name="me/sync" options={{ href: null }} />
      <Tabs.Screen name="scan/review" options={{ href: null }} />
      {/* contacts/* (detail, edit, Recently deleted) live in the Contacts tab's own stack
          (contacts/_layout.tsx), so they aren't listed as hidden tabs here. */}
      <Tabs.Screen name="events/[eventId]" options={{ href: null }} />
    </Tabs>
    {/* Team announcements live on every tab: pop-up when one is sent + the dropdown with all of them. */}
    <AnnouncementsLayer />
    {/* Calendar reminders pop up on any tab when due (Done · Snooze · Open). */}
    <ReminderLayer />
    </OverlayProvider>
  );
}

const styles = makeStyles(() => ({
  container: {
    flex: 1,
    backgroundColor: theme.colors.bg.app,
  },
  fabContainer: {
    position: 'absolute',
    top: -20,
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: theme.colors.brand[600],
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: theme.colors.brand[600],
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  syncBanner: {
    backgroundColor: theme.colors.semantic.successText,
    paddingVertical: 4,
    paddingHorizontal: theme.spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  syncText: {
    color: toneFg('#86efac'),
    fontWeight: '600',
  },
}));
