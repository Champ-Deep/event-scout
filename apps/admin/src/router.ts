import React from 'react';
import { createRouter, createRootRouteWithContext, createRoute, redirect, Outlet } from '@tanstack/react-router';
import type { QueryClient } from '@tanstack/react-query';
import { RootComponent } from './routes/__root';
import { EventBoardComponent } from './routes/_auth.index';
import { ActivityComponent } from './routes/_auth.activity';
import { ContactsGridComponent } from './routes/_auth.contacts';
import { ContactDrawerRoute } from './routes/_auth.contacts.$id';
import { DuplicatesComponent } from './routes/_auth.duplicates';
import { DeletedComponent } from './routes/_auth.deleted';
import { ContactListsComponent } from './routes/_auth.lists';
import { BroadcastsCenterComponent } from './routes/_auth.broadcasts';
import { PipelinesComponent } from './routes/_auth.pipelines';
import { ExhibitorsComponent } from './routes/_auth.exhibitors';
import { FilesComponent } from './routes/_auth.files';
import { AssistantComponent } from './routes/_auth.assistant';
import { SystemComponent } from './routes/_auth.system';
import { AccountComponent } from './routes/_auth.account';
import { SalesProfileComponent } from './routes/_auth.sales-profile';
import { SeatsComponent } from './routes/_auth.seats';
import { ExportsComponent } from './routes/_auth.exports';
import { LoginComponent } from './routes/login';
import { queryClient } from './lib/queryClient';
import { isSignedIn } from './lib/auth';
import { q } from './lib/queries';
import { contactQuery, contactCountsQuery, contactsSearch, dashboardSearch, activitySearch, pipelinesSearch, seatsSearch, listsSearch, eventsSearch, deletedSearch, broadcastsSearch } from './lib/search';
import { PageSkeleton, RouteError } from './ui/route-states';

// Wiring (route → cache → server): the URL holds every filter / sort; each route's loader
// ensureQueryData()s exactly what its component reads; components useSuspenseQuery() the same
// options, so they read the cache and never fetch on their own.
const rootRoute = createRootRouteWithContext<{ queryClient: QueryClient }>()({ component: Outlet });

const authRoute = createRoute({
  getParentRoute: () => rootRoute,
  id: '_auth',
  beforeLoad: ({ location }) => {
    if (!isSignedIn()) throw redirect({ to: '/login', search: { next: location.href } });
  },
  component: RootComponent,
});

const skel = (kind: 'table' | 'dashboard' | 'cards' | 'split') => () => React.createElement(PageSkeleton, { kind });
const base = { getParentRoute: () => authRoute, errorComponent: RouteError } as const;

const indexRoute = createRoute({
  ...base, path: '/', validateSearch: dashboardSearch, pendingComponent: skel('dashboard'),
  loaderDeps: ({ search }) => ({ days: search.days }),
  loader: ({ context: { queryClient: qc }, deps }) => {
    // first visit waits for the board; a range switch only prefetches (the page keeps the old range up)
    const warm = qc.getQueriesData({ queryKey: ['admin', 'dashboard'] }).some(([, d]) => d);
    // activity(200) is the header Activity menu's query too, so the board adds no extra request
    if (warm) { void qc.prefetchQuery(q.dashboard(deps.days)); return qc.ensureQueryData(q.activity(200)); }
    return Promise.all([qc.ensureQueryData(q.dashboard(deps.days)), qc.ensureQueryData(q.activity(200))]);
  },
  component: EventBoardComponent,
});

const activityRoute = createRoute({
  ...base, path: '/activity', validateSearch: activitySearch, pendingComponent: skel('table'),
  loaderDeps: ({ search }) => ({ date: search.date }),
  loader: ({ context: { queryClient: qc }, deps }) => (deps.date ? qc.ensureQueryData(q.activityDay(deps.date)) : qc.ensureQueryData(q.activity(200))),
  component: ActivityComponent,
});

const contactsRoute = createRoute({
  ...base, path: '/contacts', validateSearch: contactsSearch, pendingComponent: skel('table'),
  // only what the list queries use: opening / closing the drawer or switching its tab (tab, full, add)
  // must not re-run the loader and reload the whole grid
  loaderDeps: ({ search: { q: text, band, event, owner, since, sort } }) => ({ q: text, band, event, owner, since, sort }),
  loader: ({ context: { queryClient: qc }, deps, cause }) => {
    const lookups = [qc.ensureQueryData(q.users()), qc.ensureQueryData(q.events()), qc.ensureQueryData(q.dashboard(7))];
    // Arriving on the page waits for the list (unless these exact rows are cached); a filter / level /
    // sort change while on it only prefetches, so the grid keeps showing the current rows until the new
    // ones arrive. Other contact queries (palette, pickers) don't count as "already loaded".
    const cached = qc.getQueryData(q.contacts(contactQuery(deps)).queryKey) !== undefined;
    if (cause === 'stay' || cached) {
      void qc.prefetchQuery(q.contacts(contactQuery(deps)));
      void qc.prefetchQuery(q.contactStats(contactCountsQuery(deps)));
      return Promise.all(lookups);
    }
    return Promise.all([qc.ensureQueryData(q.contacts(contactQuery(deps))), qc.ensureQueryData(q.contactStats(contactCountsQuery(deps))), ...lookups]);
  },
  component: ContactsGridComponent,
});
const contactsIndexRoute = createRoute({ getParentRoute: () => contactsRoute, path: '/', component: () => null });
const contactDrawerRoute = createRoute({
  getParentRoute: () => contactsRoute, path: '$id',
  loader: ({ context: { queryClient: qc }, params }) => qc.ensureQueryData(q.contact(params.id)),
  pendingComponent: () => null,
  errorComponent: RouteError,
  component: ContactDrawerRoute,
});

const duplicatesRoute = createRoute({
  ...base, path: '/duplicates', pendingComponent: skel('cards'),
  loader: ({ context: { queryClient: qc } }) => qc.ensureQueryData(q.duplicates()),
  component: DuplicatesComponent,
});

const deletedRoute = createRoute({
  ...base, path: '/deleted', validateSearch: deletedSearch, pendingComponent: skel('table'),
  loader: ({ context: { queryClient: qc } }) => qc.ensureQueryData(q.deleted()),
  component: DeletedComponent,
});

const listsRoute = createRoute({
  ...base, path: '/lists', validateSearch: listsSearch, pendingComponent: skel('split'),
  loaderDeps: ({ search }) => ({ list: search.list }),
  loader: async ({ context: { queryClient: qc }, deps }) => {
    const [lists] = await Promise.all([qc.ensureQueryData(q.lists()), qc.ensureQueryData(q.users())]);
    const id = deps.list && lists.some((l) => l.id === deps.list) ? deps.list : lists[0]?.id;
    if (id) await qc.ensureQueryData(q.listMembers(id));
  },
  component: ContactListsComponent,
});

const pipelinesRoute = createRoute({
  ...base, path: '/pipelines', validateSearch: pipelinesSearch, pendingComponent: skel('table'),
  loaderDeps: ({ search }) => ({ status: search.status }),
  // the first page of the chosen filter; the page fetches later pages itself (Prev / Next)
  loader: ({ context: { queryClient: qc }, deps }) => qc.ensureQueryData(q.pipelines(deps.status)),
  component: PipelinesComponent,
});

const exhibitorsRoute = createRoute({
  ...base, path: '/exhibitors', validateSearch: eventsSearch, pendingComponent: skel('table'),
  loaderDeps: ({ search }) => ({ event: search.event }),
  loader: async ({ context: { queryClient: qc }, deps }) => {
    // every event (directory ∪ contacts ∪ reps' lists ∪ files); no event in the URL → the most recent
    const events = await qc.ensureQueryData(q.adminEvents());
    const event = deps.event || events[0]?.name || '';
    if (event) await qc.ensureQueryData(q.eventDetail(event));
  },
  component: ExhibitorsComponent,
});

const filesRoute = createRoute({
  ...base, path: '/files', pendingComponent: skel('table'),
  // the upload form's event suggestions (admin events ∪ planner) load in the background
  loader: ({ context: { queryClient: qc } }) => { void qc.prefetchQuery(q.adminEvents()); void qc.prefetchQuery(q.plannerEvents()); return qc.ensureQueryData(q.files()); },
  component: FilesComponent,
});

const broadcastsRoute = createRoute({
  ...base, path: '/broadcasts', validateSearch: broadcastsSearch, pendingComponent: skel('split'),
  loader: ({ context: { queryClient: qc } }) => Promise.all([qc.ensureQueryData(q.broadcasts()), qc.ensureQueryData(q.users()), qc.ensureQueryData(q.plannerEvents())]),
  component: BroadcastsCenterComponent,
});

const seatsRoute = createRoute({
  ...base, path: '/seats', validateSearch: seatsSearch, pendingComponent: skel('table'),
  loader: ({ context: { queryClient: qc } }) => qc.ensureQueryData(q.users()),
  component: SeatsComponent,
});

const assistantRoute = createRoute({
  ...base, path: '/assistant', pendingComponent: skel('split'),
  loader: ({ context: { queryClient: qc } }) => qc.prefetchQuery(q.models()),
  component: AssistantComponent,
});

const exportsRoute = createRoute({
  ...base, path: '/exports', pendingComponent: skel('cards'),
  loader: ({ context: { queryClient: qc } }) => Promise.all([qc.ensureQueryData(q.dashboard(7)), qc.ensureQueryData(q.users()), qc.ensureQueryData(q.lists())]),
  component: ExportsComponent,
});

const systemRoute = createRoute({
  ...base, path: '/system', pendingComponent: skel('cards'),
  // prefetch only: one failing check (e.g. the server is down) shows on its own tile instead of replacing the page
  loader: ({ context: { queryClient: qc } }) => Promise.allSettled([qc.prefetchQuery(q.health()), qc.prefetchQuery(q.backup()), qc.prefetchQuery(q.models())]),
  component: SystemComponent,
});

const accountRoute = createRoute({
  ...base, path: '/account', pendingComponent: skel('split'),
  loader: ({ context: { queryClient: qc } }) => qc.prefetchQuery(q.me()),
  component: AccountComponent,
});

const salesProfileRoute = createRoute({
  ...base, path: '/sales-profile', pendingComponent: skel('cards'),
  loader: ({ context: { queryClient: qc } }) => qc.ensureQueryData(q.salesProfile()),
  component: SalesProfileComponent,
});

const loginRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/login',
  validateSearch: (search: Record<string, unknown>): { next?: string } => (typeof search.next === 'string' ? { next: search.next } : {}),
  component: LoginComponent,
});

const routeTree = rootRoute.addChildren([
  authRoute.addChildren([
    indexRoute, activityRoute,
    contactsRoute.addChildren([contactsIndexRoute, contactDrawerRoute]),
    duplicatesRoute, deletedRoute, listsRoute, pipelinesRoute, exhibitorsRoute, filesRoute, broadcastsRoute, seatsRoute,
    assistantRoute, exportsRoute, systemRoute, accountRoute, salesProfileRoute,
  ]),
  loginRoute,
]);

export const router = createRouter({
  routeTree,
  context: { queryClient },
  defaultPreload: 'intent',
  // loaders own freshness via the query cache
  defaultPreloadStaleTime: 0,
  defaultPendingMs: 150,
  defaultPendingMinMs: 250,
});

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}
