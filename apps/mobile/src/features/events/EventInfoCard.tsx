import React, { useState } from 'react';
import { View, Pressable, Linking } from 'react-native';
import Animated, { FadeInDown, LinearTransition } from 'react-native-reanimated';
import {
  CalendarDays, MapPin, Globe, Building2, Tag, Clock, Ticket, ClipboardList, Shirt, UserRound, Mail, Phone, StickyNote,
  Store, UsersRound, Map as MapIcon, Info, ChevronRight,
} from 'lucide-react-native';
import { Text } from '../../ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../ui/theme';
import type { PlannedEvent } from '../../lib/api';
import { formatRange, relativeToToday } from './dates';

const open = (url: string) => { Linking.openURL(url).catch(() => {}); };
const mapsUrl = (q: string) => `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(q)}`;

function Row({ icon, label, value, onPress, lines = 2 }: {
  icon: React.ReactNode; label: string; value: string; onPress?: () => void; lines?: number;
}) {
  const body = (
    <>
      <View style={styles.rowIcon}>{icon}</View>
      <View style={{ flex: 1 }}>
        <Text style={styles.rowLabel}>{label}</Text>
        <Text style={[styles.rowValue, onPress && styles.rowLink]} numberOfLines={lines}>{value}</Text>
      </View>
      {onPress ? <ChevronRight size={15} color={theme.colors.text.muted} /> : null}
    </>
  );
  return onPress ? (
    <Pressable onPress={onPress} style={({ pressed }) => [styles.row, pressed && { opacity: 0.7 }]} accessibilityRole="link" accessibilityLabel={`${label}: ${value}`}>
      {body}
    </Pressable>
  ) : <View style={styles.row}>{body}</View>;
}

/**
 * What the admins planned for this event (Admin console → Events): dates, where it is, booth,
 * hours, registration, who to contact, the team. Read-only here.
 */
export function EventInfoCard({ event, onOpenFloorMap }: { event: PlannedEvent; onOpenFloorMap?: () => void }) {
  const [more, setMore] = useState(false);
  const d = event.details || {};
  const when = formatRange(event.startDate, event.endDate);
  const rel = relativeToToday(event.startDate, event.endDate);
  const place = [event.venue, event.address, event.city, event.country].filter(Boolean).join(', ');
  const site = event.website ? event.website.replace(/^https?:\/\//, '').replace(/\/$/, '') : '';
  const contact = [d.contactName, d.contactEmail, d.contactPhone].some(Boolean);
  const relBg = rel?.tone === 'live' ? styles.relLive : rel?.tone === 'soon' ? styles.relSoon : rel?.tone === 'past' ? styles.relPast : null;
  const relFg = rel?.tone === 'live' ? styles.relLiveText : rel?.tone === 'soon' ? styles.relSoonText : rel?.tone === 'past' ? styles.relPastText : null;

  return (
    <Animated.View entering={FadeInDown.delay(100).duration(260)} layout={LinearTransition.duration(260)} style={styles.card}>
      <View style={styles.head}>
        <Info size={13} color={theme.colors.brand[700]} />
        <Text style={styles.headLabel}>EVENT DETAILS</Text>
        <View style={{ flex: 1 }} />
        {event.myRole ? (
          <View style={[styles.pill, event.myRole === 'lead' && styles.pillLead]}>
            <Text style={[styles.pillText, event.myRole === 'lead' && { color: theme.colors.semantic.warning }]}>
              {event.myRole === 'lead' ? 'You lead' : 'On the team'}
            </Text>
          </View>
        ) : null}
      </View>

      {when ? (
        <Row icon={<CalendarDays size={15} color={theme.colors.brand[600]} />} label="When" value={when} />
      ) : (
        <Row icon={<CalendarDays size={15} color={theme.colors.text.muted} />} label="When" value="Dates not set yet" />
      )}
      {rel ? (
        <View style={[styles.rel, relBg]}><Text style={[styles.relText, relFg]}>{rel.label}</Text></View>
      ) : null}
      {d.hours ? <Row icon={<Clock size={15} color={theme.colors.brand[600]} />} label="Hours" value={d.hours} /> : null}
      {place ? (
        <Row icon={<MapPin size={15} color={theme.colors.brand[600]} />} label="Location" value={place} lines={3} onPress={() => open(mapsUrl(place))} />
      ) : null}
      {d.booth ? <Row icon={<Store size={15} color={theme.colors.brand[600]} />} label="Our booth" value={d.booth} /> : null}
      {event.floorPlans > 0 && onOpenFloorMap ? (
        <Row icon={<MapIcon size={15} color={theme.colors.brand[600]} />} label="Floor map"
          value={`${event.floorPlans} floor plan${event.floorPlans === 1 ? '' : 's'} from your admins`} onPress={onOpenFloorMap} />
      ) : null}
      {site ? <Row icon={<Globe size={15} color={theme.colors.brand[600]} />} label="Website" value={site} lines={1}
        onPress={() => open(/^https?:\/\//.test(event.website) ? event.website : `https://${event.website}`)} /> : null}
      {event.team.length ? (
        <Row icon={<UsersRound size={15} color={theme.colors.brand[600]} />} label="Team"
          value={`${event.team.length} going${event.lead ? ` · led by ${event.lead}` : ''}`} />
      ) : null}

      {more ? (
        <>
          {event.organizer ? <Row icon={<Building2 size={15} color={theme.colors.brand[600]} />} label="Organizer" value={event.organizer} /> : null}
          {event.industry ? <Row icon={<Tag size={15} color={theme.colors.brand[600]} />} label="Industry" value={event.industry} /> : null}
          {d.registration ? <Row icon={<ClipboardList size={15} color={theme.colors.brand[600]} />} label="Registration" value={d.registration} lines={4} /> : null}
          {d.tickets ? <Row icon={<Ticket size={15} color={theme.colors.brand[600]} />} label="Tickets" value={d.tickets} lines={4} /> : null}
          {d.dressCode ? <Row icon={<Shirt size={15} color={theme.colors.brand[600]} />} label="Dress code" value={d.dressCode} /> : null}
          {contact ? (
            <>
              {d.contactName ? <Row icon={<UserRound size={15} color={theme.colors.brand[600]} />} label="Contact" value={d.contactName} /> : null}
              {d.contactEmail ? <Row icon={<Mail size={15} color={theme.colors.brand[600]} />} label="Email" value={d.contactEmail} lines={1} onPress={() => open(`mailto:${d.contactEmail}`)} /> : null}
              {d.contactPhone ? <Row icon={<Phone size={15} color={theme.colors.brand[600]} />} label="Phone" value={d.contactPhone} lines={1} onPress={() => open(`tel:${(d.contactPhone || '').replace(/[^\d+]/g, '')}`)} /> : null}
            </>
          ) : null}
          {d.notes ? <Row icon={<StickyNote size={15} color={theme.colors.brand[600]} />} label="Notes from your admins" value={d.notes} lines={8} /> : null}
          {event.description ? <Text style={styles.desc}>{event.description}</Text> : null}
        </>
      ) : null}
      {event.organizer || event.industry || d.registration || d.tickets || d.dressCode || contact || d.notes || event.description ? (
        <Pressable onPress={() => setMore((x) => !x)} hitSlop={8} style={styles.moreBtn} accessibilityRole="button">
          <Text style={styles.moreText}>{more ? 'Show less' : 'More details'}</Text>
        </Pressable>
      ) : null}
    </Animated.View>
  );
}

const styles = makeStyles(() => ({
  card: { backgroundColor: toneBg('#ffffff'), borderRadius: 20, paddingHorizontal: 14, paddingTop: 12, paddingBottom: 8, marginBottom: 12, borderWidth: 1, borderColor: theme.colors.bg.divider, ...theme.shadows.card },
  head: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 4 },
  headLabel: { fontSize: 11, fontWeight: '800', letterSpacing: 1, color: theme.colors.brand[700] },
  pill: { paddingHorizontal: 9, paddingVertical: 3, borderRadius: 999, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  pillLead: { backgroundColor: theme.colors.semantic.warningBg, borderColor: theme.colors.semantic.warningBorder },
  pillText: { fontSize: 11, fontWeight: '800', color: theme.colors.brand[700] },
  row: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 8, borderTopWidth: 1, borderTopColor: theme.colors.bg.divider },
  rowIcon: { width: 30, height: 30, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  rowLabel: { fontSize: 10, fontWeight: '800', letterSpacing: 0.6, color: theme.colors.text.muted, textTransform: 'uppercase' },
  rowValue: { fontSize: 14, lineHeight: 19, fontWeight: '600', color: theme.colors.text.display, marginTop: 1 },
  rowLink: { color: theme.colors.brand[700] },
  rel: { alignSelf: 'flex-start', marginLeft: 40, marginTop: -4, marginBottom: 6, paddingHorizontal: 8, paddingVertical: 2, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  relText: { fontSize: 11, fontWeight: '800', color: theme.colors.brand[700] },
  relLive: { backgroundColor: theme.colors.semantic.successBg },
  relSoon: { backgroundColor: theme.colors.semantic.warningBg },
  relPast: { backgroundColor: toneBg('#f1f5f9') },
  relLiveText: { color: theme.colors.semantic.successText },
  relSoonText: { color: theme.colors.semantic.warning },
  relPastText: { color: toneFg('#64748b') },
  desc: { fontSize: 13, lineHeight: 19, color: theme.colors.text.body, paddingTop: 8, borderTopWidth: 1, borderTopColor: theme.colors.bg.divider },
  moreBtn: { alignSelf: 'center', paddingVertical: 8 },
  moreText: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[600] },
}));
