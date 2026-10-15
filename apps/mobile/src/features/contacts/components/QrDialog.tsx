import React, { useEffect, useState } from 'react';
import { View, Pressable, Image, ActivityIndicator, Share, ScrollView, useWindowDimensions, Platform } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import * as Clipboard from 'expo-clipboard';
import * as Sharing from 'expo-sharing';
import { File, Paths } from 'expo-file-system';
import * as LegacyFS from 'expo-file-system/legacy';
import Constants, { ExecutionEnvironment } from 'expo-constants';
import { saveFileToDownloads } from '../../me/export';
import { GestureDetector } from 'react-native-gesture-handler';
import Animated, {
  FadeIn, ZoomIn,
  useSharedValue, useAnimatedStyle, withTiming, withRepeat, withSequence, Easing,
} from 'react-native-reanimated';
import { X, Share2, Link2, Check, QrCode, Globe, AlertCircle, RotateCcw, ScanLine, CreditCard, Download } from 'lucide-react-native';
import { Text } from '../../../ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../../ui/theme';
import { useSheetMotion, SheetOverlay } from '../../../ui/SheetOverlay';
import { PressableScale } from '../../onboarding/components';
import { errorMessage } from '../../../lib/api';
import { ContactAvatar } from './ContactAvatar';

export interface QrDialogData {
  title: string;                 // e.g. the contact's name, or "My card"
  subtitle?: string | null;      // title · company
  firstName?: string | null;
  lastName?: string | null;
  photoUrl?: string | null;
  vcard: string;                 // what the contact QR contains
  qrDataUrl: string;             // PNG data URI of the vCard QR
  link?: string | null;          // optional public page (My card)
  linkQrDataUrl?: string | null; // PNG data URI of the link QR
}

interface Props {
  visible: boolean;
  loading?: boolean;
  error?: string | null;
  data: QrDialogData | null;
  onClose: () => void;
  onRetry?: () => void;
  onResetLink?: (() => Promise<void>) | undefined;
  /** My card only: renders the premium business-card PNG (GET /api/me/business-card). */
  onLoadBusinessCard?: (() => Promise<{ pngDataUrl: string; filename: string }>) | undefined;
}

type Mode = 'card' | 'link' | 'business';

/** Soft scan-line sweep over the QR while it's shown. */
function ScanSweep() {
  const t = useSharedValue(0);
  useEffect(() => {
    t.value = withRepeat(withSequence(withTiming(1, { duration: 1800, easing: Easing.inOut(Easing.quad) }), withTiming(0, { duration: 1800, easing: Easing.inOut(Easing.quad) })), -1);
  }, []);
  const style = useAnimatedStyle(() => ({ top: `${6 + t.value * 84}%`, opacity: 0.55 }));
  return <Animated.View pointerEvents="none" style={[styles.sweep, style]} />;
}

/**
 * QR code dialog (in-screen overlay, not an RN <Modal> — taps inside a Modal were unreliable
 * on Android). Shows a vCard QR any phone camera can scan to add the contact, and optionally
 * the public card link. Share sends the QR as an image; copy/share-link for the link.
 */
export function QrDialog({ visible, loading, error, data, onClose, onRetry, onResetLink, onLoadBusinessCard }: Props) {
  const insets = useSafeAreaInsets();
  const { width: winW, height: winH } = useWindowDimensions();
  // The QR must always fit whole: never wider than the sheet, never taller than ~38% of the screen.
  const qrSize = Math.round(Math.max(180, Math.min(280, winW - 96, winH * 0.38)));
  const [mode, setMode] = useState<Mode>('card');
  const [biz, setBiz] = useState<{ pngDataUrl: string; filename: string } | null>(null);
  const [bizLoading, setBizLoading] = useState(false);
  const [bizError, setBizError] = useState<string | null>(null);
  const [busy, setBusy] = useState<'image' | 'copy' | 'reset' | 'download' | null>(null);
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null);
  // Slide-in, drag, back button and slide-out: the shared sheet motion (one shared value).
  const motion = useSheetMotion({ open: visible, onClose });

  useEffect(() => {
    if (!visible) return;
    setMode('card');
    setNote(null);
    setBiz(null);
    setBizError(null);
  }, [visible]);

  useEffect(() => {
    if (!note) return;
    const t = setTimeout(() => setNote(null), note.ok ? 2800 : 7000);
    return () => clearTimeout(t);
  }, [note]);

  if (!motion.mounted) return null;
  const showLink = mode === 'link' && !!data?.link && !!data?.linkQrDataUrl;
  const qr = showLink ? data!.linkQrDataUrl! : data?.qrDataUrl;

  /** Write the QR PNG into the app cache and return its file:// URI. Tries the current
   *  expo-file-system API first, then the legacy one (Expo Go builds differ). */
  const saveQrPng = async (dataUrl: string, name: string): Promise<string> => {
    const base64 = dataUrl.includes(',') ? dataUrl.split(',', 2)[1]! : dataUrl;
    if (!base64) throw new Error('The QR image is empty.');
    try {
      const file = new File(Paths.cache, name);
      if (file.exists) file.delete();
      file.create();
      file.write(base64, { encoding: 'base64' });
      if (file.exists && (file.size ?? 1) > 0) return file.uri;
      throw new Error('file not written');
    } catch (err: any) {
      console.warn('[qr] new file API failed, using legacy', err?.message);
      const dir = LegacyFS.cacheDirectory;
      if (!dir) throw new Error("This phone didn't give the app a place to save the image.");
      const uri = `${dir}${name}`;
      await LegacyFS.writeAsStringAsync(uri, base64, { encoding: LegacyFS.EncodingType.Base64 });
      return uri;
    }
  };

  const loadBusinessCard = async () => {
    if (!onLoadBusinessCard || bizLoading) return;
    setBizLoading(true);
    setBizError(null);
    try {
      setBiz(await onLoadBusinessCard());
    } catch (e: any) {
      setBizError(errorMessage(e, "Couldn't make your business card."));
    } finally {
      setBizLoading(false);
    }
  };
  const selectMode = (m: Mode) => {
    setMode(m);
    setNote(null);
    if (m === 'business' && !biz) loadBusinessCard();
  };

  /** Save a PNG (the QR or the business card) and hand it to the system share sheet —
   *  WhatsApp, Gmail, or "Save to Files / Drive / Photos" to download it. */
  const shareImage = async (asBusinessCard = false) => {
    if (!data || busy) return;
    const png = asBusinessCard ? biz?.pngDataUrl : qr;
    if (!png) return;
    setBusy('image');
    setNote(null);
    const safe = (data.title || 'contact').replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '').toLowerCase() || 'contact';
    let uri: string | null = null;
    try {
      uri = await saveQrPng(png, asBusinessCard ? (biz?.filename || `${safe}-business-card.png`) : `qr-${safe}${showLink ? '-link' : ''}-${Date.now()}.png`);
      if (__DEV__) console.log('[qr] image saved', uri);
    } catch (e: any) {
      console.warn('[qr] saving the image failed', e?.message);
    }
    try {
      if (uri && (await Sharing.isAvailableAsync())) {
        await Sharing.shareAsync(uri, { mimeType: 'image/png', dialogTitle: asBusinessCard ? `Business card — ${data.title}` : `QR code — ${data.title}`, UTI: 'public.png' });
        setNote({ ok: true, text: asBusinessCard ? 'Pick an app to send your business card' : 'Pick an app to send the QR image' });
        return;
      }
      throw new Error(uri ? 'Sharing files is not available on this phone' : 'The image could not be saved');
    } catch (e: any) {
      console.warn('[qr] image share failed, sharing as text', e?.message);
      // Still share the card: the vCard text (contacts apps can import it) or the link.
      try {
        await Share.share({ message: showLink ? `${data.title}: ${data.link}` : data.vcard, title: data.title });
        setNote({ ok: false, text: `Couldn't attach the QR image (${e?.message || 'unknown error'}) — shared the card as text instead.` });
      } catch (e2: any) {
        setNote({ ok: false, text: `Couldn't open sharing: ${e2?.message || e?.message || 'unknown error'}` });
      }
    } finally {
      setBusy(null);
    }
  };

  /** Download the business card to the phone.
   *  App build: Photos / Gallery. Expo Go on Android can't use the media library any more (it only
   *  warns), so there — and whenever gallery access is refused — the PNG goes to the same
   *  remembered Downloads folder as the contacts export. */
  const downloadCard = async () => {
    if (!biz || busy) return;
    setBusy('download');
    setNote(null);
    const name = biz.filename || 'business-card.png';
    const inExpoGo = Constants.executionEnvironment === ExecutionEnvironment.StoreClient;
    try {
      if (!(inExpoGo && Platform.OS === 'android')) {
        try {
          // Loaded only here, so Expo Go never touches the media library.
          const MediaLibrary = require('expo-media-library/legacy');
          const uri = await saveQrPng(biz.pngDataUrl, name);
          const perm = await MediaLibrary.requestPermissionsAsync(true); // write-only: just saving
          if (perm.granted) {
            await MediaLibrary.saveToLibraryAsync(uri);
            setNote({ ok: true, text: Platform.OS === 'ios' ? 'Saved to Photos' : 'Saved to your gallery (Pictures)' });
            return;
          }
        } catch (err: any) {
          console.warn('[card] save to gallery failed', err?.message);
        }
      }
      if (Platform.OS === 'android') {
        const r = await saveFileToDownloads(name.replace(/\.png$/i, ''), 'image/png', biz.pngDataUrl.split(',', 2)[1] || '', 'base64');
        setNote({ ok: true, text: `Saved to ${r.where}${r.askedForFolder ? ' · folder remembered' : ''}` });
        return;
      }
      setNote({ ok: false, text: 'Allow photo access in Settings to download the card, or use Share → Save Image.' });
    } catch (e: any) {
      console.warn('[card] download failed', e?.message);
      setNote({ ok: false, text: `Couldn't download the card: ${e?.message || 'unknown error'}` });
    } finally {
      setBusy(null);
    }
  };

  const copyLink = async () => {
    if (!data?.link || busy) return;
    setBusy('copy');
    try {
      await Clipboard.setStringAsync(data.link);
      setNote({ ok: true, text: 'Link copied — paste it anywhere' });
    } catch {
      try { await Share.share({ message: data.link }); } catch { setNote({ ok: false, text: 'Couldn’t copy the link.' }); }
    } finally {
      setBusy(null);
    }
  };

  const resetLink = async () => {
    if (!onResetLink || busy) return;
    setBusy('reset');
    try {
      await onResetLink();
      setNote({ ok: true, text: 'New link made — the old one no longer works' });
    } catch {
      setNote({ ok: false, text: "Couldn't make a new link." });
    } finally {
      setBusy(null);
    }
  };

  return (
    <SheetOverlay motion={motion} backdropColor="rgba(15,23,42,0.55)" zIndex={60} backdropLabel="Close QR code">
      <View style={[styles.wrap, { maxHeight: winH - Math.max(insets.top, 12) - 24 }]}>
        <Animated.View style={[styles.sheet, { paddingBottom: Math.max(insets.bottom, 16) }, motion.sheetStyle]}>
          <View style={styles.topRow}>
            {/* Drag area (handle + title). The ✕ sits OUTSIDE it so the swipe gesture can't swallow its tap. */}
            <GestureDetector gesture={motion.pan}>
              <View collapsable={false} style={{ flex: 1 }}>
                <View style={styles.handleArea}><View style={styles.handle} /></View>
                <View style={styles.header}>
                  <View style={styles.headerIcon}><QrCode size={16} color="white" /></View>
                  <Text style={styles.headerTitle}>QR code</Text>
                </View>
              </View>
            </GestureDetector>
            <Pressable onPress={motion.dismiss} hitSlop={12} style={({ pressed }) => [styles.closeBtn, pressed && { opacity: 0.6, transform: [{ scale: 0.92 }] }]}
              accessibilityRole="button" accessibilityLabel="Close">
              <X size={18} color={theme.colors.text.body} />
            </Pressable>
          </View>

          <ScrollView style={{ flexShrink: 1 }} contentContainerStyle={{ paddingBottom: 4 }} showsVerticalScrollIndicator={false} bounces={false}>
          {loading ? (
            <View style={styles.center}><ActivityIndicator color={theme.colors.brand[600]} /><Text style={styles.muted}>Making the QR code…</Text></View>
          ) : error || !data ? (
            <View style={styles.center}>
              <AlertCircle size={22} color={theme.colors.semantic.danger} />
              <Text style={styles.muted}>{error || "Couldn't load the QR code."}</Text>
              {onRetry ? (
                <PressableScale onPress={onRetry} style={styles.retryBtn} leadingIcon={<RotateCcw size={14} color={theme.colors.brand[700]} />}>
                  <Text style={styles.retryText}>Try again</Text>
                </PressableScale>
              ) : null}
            </View>
          ) : (
            <>
              <Animated.View entering={FadeIn.delay(80)} style={styles.who}>
                <ContactAvatar firstName={data.firstName ?? data.title} lastName={data.lastName ?? null} photoUrl={data.photoUrl ?? null} size={42} />
                <View style={{ flex: 1 }}>
                  <Text style={styles.name} numberOfLines={1}>{data.title}</Text>
                  {data.subtitle ? <Text style={styles.sub} numberOfLines={1}>{data.subtitle}</Text> : null}
                </View>
              </Animated.View>

              {data.link || onLoadBusinessCard ? (
                <View style={styles.tabs}>
                  {([
                    { m: 'card' as Mode, label: 'Add to contacts', Icon: ScanLine, show: true },
                    { m: 'link' as Mode, label: 'Card page', Icon: Globe, show: !!data.link },
                    { m: 'business' as Mode, label: 'Business card', Icon: CreditCard, show: !!onLoadBusinessCard },
                  ]).filter((x) => x.show).map(({ m, label, Icon }) => {
                    const on = mode === m;
                    return (
                      <Pressable key={m} onPress={() => selectMode(m)} style={[styles.tab, on && styles.tabOn]} accessibilityRole="tab" accessibilityState={{ selected: on }}>
                        <Icon size={13} color={on ? 'white' : theme.colors.text.body} />
                        <Text style={[styles.tabText, on && { color: 'white' }]} numberOfLines={1}>{label}</Text>
                      </Pressable>
                    );
                  })}
                </View>
              ) : null}

              {mode === 'business' ? (
                <Animated.View key="business" entering={FadeIn.duration(220)}>
                  <View style={[styles.bizFrame, { height: Math.round((winW - 36) / 1.75) }]}>
                    {biz ? (
                      <Animated.View entering={ZoomIn.duration(260)} style={{ width: '100%', height: '100%' }}>
                        <Image source={{ uri: biz.pngDataUrl }} style={styles.bizImg} resizeMode="contain" />
                      </Animated.View>
                    ) : bizLoading ? (
                      <View style={styles.center}><ActivityIndicator color={toneFg('#b08d48')} /><Text style={styles.muted}>Designing your card…</Text></View>
                    ) : (
                      <View style={styles.center}>
                        <Text style={[styles.muted, { color: theme.colors.semantic.danger }]}>{bizError || "Couldn't make your business card."}</Text>
                        <Pressable onPress={loadBusinessCard} hitSlop={8}><Text style={styles.retryText}>Try again</Text></Pressable>
                      </View>
                    )}
                  </View>
                  <Text style={styles.caption}>
                    A print-ready card (3.5″ × 2″, 600 dpi). The QR on it adds you to their contacts. Details come from your profile — edit them with ✏️ on the Me page.
                  </Text>
                </Animated.View>
              ) : (
              <>
              <Animated.View key={mode} entering={ZoomIn.duration(260)} style={[styles.qrCard, { width: qrSize, height: qrSize }]}>
                <Image source={{ uri: qr }} style={{ width: qrSize - 24, height: qrSize - 24 }} resizeMode="contain" />
                <ScanSweep />
              </Animated.View>
              <Text style={styles.caption}>
                {showLink
                  ? 'Card page: opens a web page with these details and a “Save contact” button — best for sending in chats or email.'
                  : 'Add to contacts: point any phone camera here and it offers “Add contact” with these details. Works offline — no app needed.'}
              </Text>

              {showLink ? (
                <Text style={styles.linkText} numberOfLines={1} selectable>{data.link}</Text>
              ) : null}
              </>
              )}

              {note ? (
                <Animated.View key={note.text} entering={FadeIn} style={styles.note}>
                  {note.ok ? <Check size={14} color={theme.colors.semantic.successText} /> : <AlertCircle size={14} color={theme.colors.semantic.danger} />}
                  <Text style={[styles.noteText, !note.ok && { color: theme.colors.semantic.danger }]}>{note.text}</Text>
                </Animated.View>
              ) : null}

              <View style={styles.actions}>
                {mode === 'business' ? (
                  <>
                    <PressableScale onPress={downloadCard} disabled={!!busy || !biz} containerStyle={{ flex: 1 }} style={[styles.primaryBtn, !biz && { opacity: 0.6 }]} scaleTo={0.96}
                      leadingIcon={busy === 'download' ? <ActivityIndicator size="small" color="white" /> : <Download size={16} color="white" />}
                      accessibilityLabel="Download business card">
                      <Text style={styles.primaryText}>Download</Text>
                    </PressableScale>
                    <PressableScale onPress={() => shareImage(true)} disabled={!!busy || !biz} containerStyle={{ flex: 1 }} style={[styles.ghostBtn, styles.ghostWide, !biz && { opacity: 0.6 }]} scaleTo={0.96}
                      leadingIcon={busy === 'image' ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <Share2 size={16} color={theme.colors.brand[700]} />}
                      accessibilityLabel="Share business card">
                      <Text style={styles.ghostText}>Share</Text>
                    </PressableScale>
                  </>
                ) : null}
                {mode !== 'business' && data.link ? (
                  <PressableScale onPress={copyLink} disabled={!!busy} style={styles.ghostBtn} scaleTo={0.95}
                    leadingIcon={busy === 'copy' ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <Link2 size={16} color={theme.colors.brand[700]} />}>
                    <Text style={styles.ghostText}>Copy link</Text>
                  </PressableScale>
                ) : null}
                {mode !== 'business' ? (
                  <PressableScale onPress={() => shareImage(false)} disabled={!!busy} containerStyle={{ flex: 1 }} style={styles.primaryBtn} scaleTo={0.96}
                    leadingIcon={busy === 'image' ? <ActivityIndicator size="small" color="white" /> : <Share2 size={16} color="white" />}>
                    <Text style={styles.primaryText}>Share QR</Text>
                  </PressableScale>
                ) : null}
              </View>
              {onResetLink && data.link && showLink ? (
                <Pressable onPress={resetLink} disabled={!!busy} style={styles.resetRow} hitSlop={6}>
                  {busy === 'reset' ? <ActivityIndicator size="small" color={theme.colors.text.muted} /> : <RotateCcw size={12} color={theme.colors.text.muted} />}
                  <Text style={styles.resetText}>Make a new link (turns the old one off)</Text>
                </Pressable>
              ) : null}
            </>
          )}
          </ScrollView>
        </Animated.View>
      </View>
    </SheetOverlay>
  );
}

const styles = makeStyles(() => ({
  wrap: { width: '100%' },
  topRow: { flexDirection: 'row', alignItems: 'flex-end', gap: 8 },
  sheet: { backgroundColor: toneBg('#ffffff'), borderTopLeftRadius: 28, borderTopRightRadius: 28, paddingHorizontal: 18, flexShrink: 1 },
  handleArea: { alignItems: 'center', paddingTop: 10, paddingBottom: 4 },
  handle: { width: 42, height: 5, borderRadius: 3, backgroundColor: toneBg('#cbd5e1') },
  header: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 8 },
  headerIcon: { width: 32, height: 32, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600] },
  headerTitle: { flex: 1, fontSize: 17, fontWeight: '800', color: theme.colors.text.display },
  closeBtn: { width: 38, height: 38, borderRadius: 19, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.bg.surfaceHover, marginBottom: 8 },
  center: { alignItems: 'center', gap: 10, paddingVertical: 40 },
  muted: { fontSize: 13, color: theme.colors.text.muted, textAlign: 'center' },
  retryBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 14, paddingVertical: 8, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  retryText: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[700] },
  who: { flexDirection: 'row', alignItems: 'center', gap: 12, marginTop: 4, marginBottom: 12 },
  name: { fontSize: 17, lineHeight: 22, fontWeight: '800', color: theme.colors.text.display },
  sub: { fontSize: 13, color: theme.colors.text.body },
  tabs: { flexDirection: 'row', gap: 6, padding: 4, borderRadius: 999, backgroundColor: theme.colors.bg.surfaceHover, marginBottom: 12 },
  tab: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, paddingVertical: 8, paddingHorizontal: 4, borderRadius: 999 },
  bizFrame: { width: '100%', borderRadius: 14, overflow: 'hidden', backgroundColor: toneBg('#fcfbf7'), borderWidth: 1, borderColor: toneBg('#e7dcc3'), alignItems: 'center', justifyContent: 'center', ...theme.shadows.active },
  bizImg: { width: '100%', height: '100%' },
  tabOn: { backgroundColor: theme.colors.brand[600] },
  tabText: { fontSize: 12, fontWeight: '700', color: theme.colors.text.body, flexShrink: 1 },
  qrCard: {
    alignSelf: 'center', padding: 12, borderRadius: 24, backgroundColor: '#ffffff' /* white in both modes: QR codes must scan */, overflow: 'hidden', alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: theme.colors.brand[100], ...theme.shadows.active,
  },
  sweep: { position: 'absolute', left: 14, right: 14, height: 2, borderRadius: 1, backgroundColor: theme.colors.brand[500] },
  caption: { fontSize: 12, color: theme.colors.text.muted, textAlign: 'center', marginTop: 12, lineHeight: 17 },
  linkText: { fontSize: 12, color: theme.colors.text.body, textAlign: 'center', marginTop: 6, fontFamily: theme.typography.fontFamily.mono },
  note: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, marginTop: 10 },
  noteText: { fontSize: 12, fontWeight: '700', color: theme.colors.semantic.successText },
  actions: { flexDirection: 'row', gap: 10, marginTop: 14 },
  ghostBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 16, height: 48, borderRadius: 999, borderWidth: 1.5, borderColor: theme.colors.brand[200], backgroundColor: toneBg('#ffffff') },
  ghostText: { fontSize: 14, fontWeight: '800', color: theme.colors.brand[700] },
  ghostWide: { justifyContent: 'center' },
  primaryBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, height: 48, borderRadius: 999, backgroundColor: theme.colors.brand[600], ...theme.shadows.active },
  primaryText: { fontSize: 15, fontWeight: '800', color: 'white' },
  resetRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, marginTop: 12 },
  resetText: { fontSize: 12, color: theme.colors.text.muted },
}));
