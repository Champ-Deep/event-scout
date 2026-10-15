import { useCallback, useEffect, useRef } from 'react';
import { BackHandler } from 'react-native';
import { useFocusEffect, useNavigation, useRoute, useRouter } from 'expo-router';

/**
 * Opening Contacts-tab pages (a contact, Add contact, Recently deleted) from another tab —
 * Scout, an event, Scan, Ask AI, Me.
 *
 * Those pages live in the Contacts tab's stack, so a plain push landed you there with "Back"
 * going to the Contacts list instead of where you came from, and the Contacts tab kept showing
 * that page afterwards. `from` tells the page where Back returns to (see useReturnTo).
 */
export type ContactOrigin = '/feed' | '/scan' | '/ask' | '/me' | `/events/${string}`;

export function contactHref(id: string, from?: ContactOrigin | string) {
  return { pathname: '/contacts/[id]', params: from ? { id, from } : { id } } as const;
}

/**
 * `withAnchor`: build the Contacts stack as [list, page] rather than [page] alone when the Contacts
 * tab hasn't been opened yet. Without it, a contact opened from Scout / Ask / an event was the
 * ONLY screen in the Contacts tab — tapping Contacts kept showing that contact (there was no list
 * to go back to) and leaving it couldn't reset the tab. (Expo Router only applies the layout's
 * initialRouteName when the navigation asks for the anchor.)
 */
const ANCHOR = { withAnchor: true } as const;

type Nav = { push: (href: any, options?: any) => void };

export function openContact(router: Nav, id: string, from: ContactOrigin) {
  router.push(contactHref(id, from), ANCHOR);
}

/** Add contact (`/contacts/new`) or Recently deleted (`/contacts/deleted`) from another tab. */
export function openContactsPage(router: Nav, page: 'new' | 'deleted', from: ContactOrigin) {
  router.push({ pathname: `/contacts/${page}`, params: { from } }, ANCHOR);
}

/** Replace the current (other-tab) screen with a contact — e.g. Scan review → the saved contact. */
export function replaceWithContact(router: { replace: (href: any, options?: any) => void }, id: string, from: ContactOrigin) {
  router.replace(contactHref(id, from), ANCHOR);
}

/** What the Contacts list can be opened showing (contacts/index.tsx reads `show`). */
export type ContactsListView = 'all' | 'hot' | 'unscored' | 'today' | 'week' | 'scored';

/**
 * Open the Contacts list on a view from another tab (Scout "Met today", Me's stat tiles…).
 * If a contact (or Add / Recently deleted) is still open on the Contacts tab, it's popped first —
 * otherwise you'd land on that old page instead of the list. Use from a tab screen (feed, me).
 */
export function useOpenContactsList() {
  const router = useRouter();
  const navigation = useNavigation();
  return useCallback((show: ContactsListView) => {
    const tabs: any = navigation.getState();
    const contacts = tabs?.routes?.find((r: any) => r.name === 'contacts');
    if (contacts?.state?.key && contacts.state.index > 0) {
      navigation.dispatch({ type: 'POP_TO_TOP', target: contacts.state.key } as any);
    }
    // `t` makes tapping the same tile again re-apply the view.
    router.navigate({ pathname: '/contacts', params: { show, t: String(Date.now()) } } as any);
  }, [navigation, router]);
}

/**
 * For a Contacts-stack page that may have been opened from another tab (`from` param).
 * Returns `leave()`: back to `from` when set, otherwise a normal stack back. When `from` is set:
 *  - the Android back button runs `onBack` (the page's own Back, so unsaved-change checks still
 *    apply), falling back to `leave`;
 *  - iOS swipe-back is off (it would drop you into the Contacts list);
 *  - leaving the page while it's still the top of the stack (Back, or tapping another tab)
 *    resets the Contacts tab to its list. Pushing Edit / Share on top keeps it.
 */
export function useReturnTo(from: string | undefined, onBack?: () => void) {
  const router = useRouter();
  const navigation = useNavigation();
  const route = useRoute();
  const onBackRef = useRef(onBack);
  onBackRef.current = onBack;

  const leave = useCallback(() => {
    if (from) router.navigate(from as any);
    else router.back();
  }, [from, router]);

  useEffect(() => {
    if (!from) return undefined;
    navigation.setOptions({ gestureEnabled: false } as any);
    return navigation.addListener('blur' as any, () => {
      const s: any = navigation.getState();
      if (s?.index > 0 && s.routes?.[s.index]?.key === route.key) navigation.dispatch({ type: 'POP_TO_TOP' } as any);
    });
  }, [from, navigation, route.key]);

  useFocusEffect(useCallback(() => {
    if (!from) return undefined;
    const sub = BackHandler.addEventListener('hardwareBackPress', () => {
      (onBackRef.current ?? leave)();
      return true;
    });
    return () => sub.remove();
  }, [from, leave]));

  return leave;
}
