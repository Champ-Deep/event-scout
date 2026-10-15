import React, { useEffect, useRef, useState } from 'react';
import { View, StyleSheet, Pressable, Image, Linking, ActivityIndicator, LayoutChangeEvent } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect, useRouter } from 'expo-router';
import { CameraView, useCameraPermissions, type BarcodeScanningResult } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated, {
  FadeIn, FadeInDown, FadeInUp, FadeOut, ZoomIn,
  useSharedValue, useAnimatedStyle, withRepeat, withSequence, withSpring, withTiming, Easing, interpolateColor,
} from 'react-native-reanimated';
import {
  X, Zap, ZapOff, SwitchCamera, ImageIcon, CreditCard, QrCode, Camera, AlertCircle, RotateCcw, Copy, ChevronRight, Settings, CloudOff, Plus, Minus,
} from 'lucide-react-native';
import { useScanSession, type ScanOutcome } from '../../../src/features/scan';
import { prepareOnDeviceScanning } from '../../../src/features/scan/ocr/saveOnDevice';
import { useSession } from '../../../src/features/auth';
import { Text } from '../../../src/ui/Text';
import { openContact } from '../../../src/features/contacts/open';
import { theme, toneBg, toneFg, makeStyles } from '../../../src/ui/theme';
import { PressableScale } from '../../../src/features/onboarding/components';


type Mode = 'card' | 'qr';
type Phase = 'idle' | 'processing' | 'error' | 'duplicate' | 'queued';

const GREEN = '#34d399';
const MODE_META: Record<Mode, { label: string; title: string }> = {
  card: { label: 'Card', title: 'Scan a business card' },
  qr: { label: 'QR badge', title: 'Scan a badge QR code' },
};

/**
 * Zoom: − / + buttons, a level bar and a readout. expo-camera's zoom is 0–1 across the phone's own
 * range (no real magnification is reported), so the readout is "1×" at no zoom, else a percentage.
 * Tapping the readout resets to 1×.
 */
const ZOOM_STEP = 0.1;
function ZoomControl({ zoom, onChange }: { zoom: number; onChange: (z: number) => void }) {
  const fill = useSharedValue(zoom);
  useEffect(() => { fill.value = withTiming(zoom, { duration: 120 }); }, [zoom]);
  const barStyle = useAnimatedStyle(() => ({ width: `${Math.round(fill.value * 100)}%` }));
  const set = (z: number) => onChange(Math.round(Math.max(0, Math.min(1, z)) * 100) / 100);
  return (
    <View style={styles.zoomPill}>
      <PressableScale onPress={() => set(zoom - ZOOM_STEP)} disabled={zoom <= 0} scaleTo={0.85}
        style={[styles.zoomBtn, zoom <= 0 && { opacity: 0.4 }]} accessibilityRole="button" accessibilityLabel="Zoom out">
        <Minus size={16} color="white" />
      </PressableScale>
      <Pressable onPress={() => set(0)} hitSlop={8} style={styles.zoomMid} accessibilityRole="adjustable"
        accessibilityLabel={zoom <= 0 ? 'Zoom: none' : `Zoom ${Math.round(zoom * 100)} percent. Tap to reset`}>
        <Text style={styles.zoomText}>{zoom <= 0 ? '1×' : `Zoom ${Math.round(zoom * 100)}%`}</Text>
        <View style={styles.zoomTrack}><Animated.View style={[styles.zoomFill, barStyle]} /></View>
      </Pressable>
      <PressableScale onPress={() => set(zoom + ZOOM_STEP)} disabled={zoom >= 1} scaleTo={0.85}
        style={[styles.zoomBtn, zoom >= 1 && { opacity: 0.4 }]} accessibilityRole="button" accessibilityLabel="Zoom in">
        <Plus size={16} color="white" />
      </PressableScale>
    </View>
  );
}

/** Viewfinder: breathing corner brackets + a sweeping scan line; reshapes per mode. */
function Viewfinder({ mode, found, busy, maxWidth }: { mode: Mode; found: boolean; busy: boolean; maxWidth: number }) {
  const w = Math.min(maxWidth * 0.86, 360);
  const target = mode === 'card' ? { width: w, height: Math.round(w / 1.6) } : { width: Math.min(w, 250), height: Math.min(w, 250) };
  const width = useSharedValue(target.width);
  const height = useSharedValue(target.height);
  const breathe = useSharedValue(0);
  const sweep = useSharedValue(0);
  const tint = useSharedValue(0);

  useEffect(() => {
    width.value = withSpring(target.width, { damping: 22, stiffness: 140, overshootClamping: true });
    height.value = withSpring(target.height, { damping: 22, stiffness: 140, overshootClamping: true });
  }, [mode, maxWidth]);
  useEffect(() => {
    breathe.value = withRepeat(withSequence(withTiming(1, { duration: 1100 }), withTiming(0, { duration: 1100 })), -1);
    sweep.value = withRepeat(withTiming(1, { duration: 2200, easing: Easing.inOut(Easing.quad) }), -1, true);
  }, []);
  useEffect(() => {
    tint.value = withTiming(found ? 1 : busy ? 2 : 0, { duration: 220 });
  }, [found, busy]);

  const frameStyle = useAnimatedStyle(() => ({
    width: width.value,
    height: height.value,
    transform: [{ scale: 1 + breathe.value * 0.015 }],
  }));
  const BRAND_300 = theme.colors.brand[300]; // plain strings for the animation (a worklet keeps its first copy of the theme)
  const cornerColor = useAnimatedStyle(() => ({
    borderColor: interpolateColor(tint.value, [0, 1, 2], ['#ffffff', GREEN, BRAND_300]),
  }));
  const lineStyle = useAnimatedStyle(() => ({
    opacity: busy ? 0 : 0.85,
    transform: [{ translateY: sweep.value * (height.value - 4) }],
    backgroundColor: interpolateColor(tint.value, [0, 1], [BRAND_300, GREEN]),
  }));

  return (
    <Animated.View style={[styles.frame, frameStyle]} pointerEvents="none">
      <Animated.View style={[styles.corner, styles.cTL, cornerColor]} />
      <Animated.View style={[styles.corner, styles.cTR, cornerColor]} />
      <Animated.View style={[styles.corner, styles.cBL, cornerColor]} />
      <Animated.View style={[styles.corner, styles.cBR, cornerColor]} />
      <Animated.View style={[styles.scanLine, lineStyle]} />
    </Animated.View>
  );
}

/** Dark two-option switch with a sliding pill. */
function ModeSwitch({ mode, onChange, disabled }: { mode: Mode; onChange: (m: Mode) => void; disabled: boolean }) {
  const [w, setW] = useState(0);
  const x = useSharedValue(0);
  const seg = w ? (w - 8) / 2 : 0;
  useEffect(() => {
    x.value = withSpring(mode === 'card' ? 0 : seg, { damping: 22, stiffness: 200, overshootClamping: true });
  }, [mode, seg]);
  const pill = useAnimatedStyle(() => ({ width: seg, transform: [{ translateX: x.value }] }));
  return (
    <View style={[styles.modeSwitch, disabled && { opacity: 0.5 }]} onLayout={(e: LayoutChangeEvent) => setW(e.nativeEvent.layout.width)}>
      <Animated.View style={[styles.modePill, pill]} />
      {(['card', 'qr'] as Mode[]).map((m) => {
        const Icon = m === 'card' ? CreditCard : QrCode;
        const active = mode === m;
        return (
          <Pressable key={m} style={styles.modeSeg} onPress={() => !disabled && onChange(m)} accessibilityRole="radio" accessibilityState={{ selected: active }}>
            <Icon size={15} color={active ? '#0f172a' : 'rgba(255,255,255,0.75)'} />
            <Text style={[styles.modeText, active && styles.modeTextActive]}>{MODE_META[m].label}</Text>
          </Pressable>
        );
      })}
    </View>
  );
}

export default function ScanScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const [permission, requestPermission] = useCameraPermissions();
  const { userId } = useSession();
  const { processImageScan, processQRScan } = useScanSession();

  // Load the on-device OCR models in the background so an offline card scan is quick.
  useEffect(() => { prepareOnDeviceScanning(userId ?? null); }, [userId]);
  const cameraRef = useRef<CameraView>(null);

  const [mode, setMode] = useState<Mode>('card');
  const [phase, setPhase] = useState<Phase>('idle');
  const [torch, setTorch] = useState(false);
  const [facing, setFacing] = useState<'back' | 'front'>('back');
  // Camera zoom 0–1: pinch on the preview, or the − / + control above the mode switch.
  const [zoom, setZoom] = useState(0);
  const pinchStart = useRef(0);
  const pinch = Gesture.Pinch()
    .runOnJS(true)
    .onStart(() => { pinchStart.current = zoom; })
    .onUpdate((e) => {
      // spread / pinch of the fingers → zoom; scaled so a comfortable pinch covers the range
      const next = Math.round(Math.max(0, Math.min(1, pinchStart.current + (e.scale - 1) * 0.45)) * 100) / 100;
      setZoom((z) => (z === next ? z : next));
    });
  useEffect(() => { setZoom(0); }, [facing]); // each camera starts unzoomed
  const [captured, setCaptured] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [dup, setDup] = useState<ScanOutcome | null>(null);
  const [qrState, setQrState] = useState<'none' | 'contact' | 'other'>('none');
  const [elapsed, setElapsed] = useState(0);
  const [screenW, setScreenW] = useState(360);
  const lastQrAt = useRef(0);

  const flash = useSharedValue(0);
  const flashStyle = useAnimatedStyle(() => ({ opacity: flash.value }));
  const shutterScale = useSharedValue(1);
  const shutterStyle = useAnimatedStyle(() => ({ transform: [{ scale: shutterScale.value }] }));

  // Reset when the user comes back to this tab (e.g. after reviewing a scan).
  useFocusEffect(React.useCallback(() => {
    setPhase('idle');
    setCaptured(null);
    setErrorMsg('');
    setDup(null);
    setQrState('none');
    return () => setTorch(false);
  }, []));

  // Real elapsed time while the server reads the card.
  useEffect(() => {
    if (phase !== 'processing') return;
    setElapsed(0);
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [phase]);

  const busy = phase === 'processing';

  const process = async (uri: string, kind: Mode) => {
    setCaptured(uri);
    setPhase('processing');
    setErrorMsg('');
    try {
      const outcome = kind === 'card' ? await processImageScan(uri) : await processQRScan(uri);
      if (outcome.status === 'duplicate') {
        setDup(outcome);
        setPhase('duplicate');
        return;
      }
      if (outcome.status === 'queued') {
        setPhase('queued'); // offline: the photo is saved and read automatically later
        return;
      }
      router.push({
        pathname: '/scan/review',
        params: {
          id: outcome.contactId, autoScore: outcome.autoScoreStarted ? '1' : '0', source: kind,
          ...(outcome.readOnDevice ? { onDevice: outcome.pendingSync ? 'offline' : '1' } : {}),
        },
      });
    } catch (e: any) {
      setErrorMsg(e?.message || "Couldn't read that. Please try again.");
      setPhase('error');
    }
  };

  const capture = async (kind: Mode) => {
    if (!cameraRef.current || phase !== 'idle') return;
    flash.value = withSequence(withTiming(0.9, { duration: 60 }), withTiming(0, { duration: 260 }));
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
      if (photo?.uri) await process(photo.uri, kind);
    } catch {
      setErrorMsg("The camera couldn't take a photo. Please try again.");
      setPhase('error');
    }
  };

  const pickImage = async () => {
    if (busy) return;
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], allowsEditing: true, quality: 0.9 });
    if (!result.canceled && result.assets?.[0]?.uri) await process(result.assets[0].uri, mode);
  };

  // QR mode: the camera spots codes live. Contact cards (vCards) are captured automatically;
  // anything else just gets a hint and is never uploaded.
  const onBarcode = (res: BarcodeScanningResult) => {
    if (mode !== 'qr' || phase !== 'idle') return;
    const now = Date.now();
    if (now - lastQrAt.current < 1500) return;
    lastQrAt.current = now;
    const isContact = /BEGIN:VCARD/i.test(res.data || '');
    setQrState(isContact ? 'contact' : 'other');
    if (isContact) capture('qr');
    else setTimeout(() => setQrState('none'), 2500);
  };

  const reset = () => {
    setPhase('idle');
    setCaptured(null);
    setErrorMsg('');
    setDup(null);
    setQrState('none');
  };

  // ── Permission states ────────────────────────────────────────────────────
  if (!permission) {
    return <View style={styles.container}><ActivityIndicator color="white" style={{ marginTop: 120 }} /></View>;
  }
  if (!permission.granted) {
    const blocked = permission.canAskAgain === false;
    return (
      <View style={[styles.permissionContainer, { paddingTop: insets.top + 12 }]}>
        <PressableScale style={[styles.roundBtn, styles.permissionClose]} onPress={() => router.back()} scaleTo={0.9} accessibilityLabel="Close">
          <X size={20} color="white" />
        </PressableScale>
        <Animated.View entering={ZoomIn.duration(260)} style={styles.permissionIcon}>
          <Camera size={34} color="white" />
        </Animated.View>
        <Animated.Text entering={FadeInDown.delay(80).duration(260)} style={styles.permissionTitle}>Camera access needed</Animated.Text>
        <Animated.Text entering={FadeInDown.delay(140).duration(260)} style={styles.permissionText}>
          EventScout uses the camera to read business cards and badge QR codes. Photos are only used to create the contact.
        </Animated.Text>
        <Animated.View entering={FadeInDown.delay(200).duration(260)} style={{ gap: 12, alignItems: 'center' }}>
          <PressableScale
            style={styles.permissionBtn}
            onPress={blocked ? () => Linking.openSettings() : requestPermission}
            leadingIcon={blocked ? <Settings size={16} color="white" /> : <Camera size={16} color="white" />}
            accessibilityRole="button"
          >
            <Text style={styles.permissionBtnText}>{blocked ? 'Open Settings' : 'Allow camera'}</Text>
          </PressableScale>
          <Pressable onPress={pickImage} hitSlop={8}>
            <Text style={styles.permissionLink}>Or choose a photo from your gallery</Text>
          </Pressable>
        </Animated.View>
      </View>
    );
  }

  const hint = mode === 'card'
    ? 'Fit the whole card inside the frame, then tap the shutter'
    : qrState === 'contact' ? 'Contact QR code found — capturing…'
    : qrState === 'other' ? "That QR code isn't a contact card"
    : 'Point at the QR code on the badge — it captures automatically';

  return (
    <View style={styles.container} onLayout={(e) => setScreenW(e.nativeEvent.layout.width)}>
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        facing={facing}
        enableTorch={torch}
        zoom={zoom}
        autofocus="on"
        barcodeScannerSettings={{ barcodeTypes: ['qr'] }}
        {...(mode === 'qr' && phase === 'idle' ? { onBarcodeScanned: onBarcode } : {})}
      />

      {/* Pinch anywhere on the preview to zoom (buttons and panels above still get their taps) */}
      {phase === 'idle' ? (
        <GestureDetector gesture={pinch}>
          <View collapsable={false} style={StyleSheet.absoluteFill} />
        </GestureDetector>
      ) : null}

      {/* Frozen capture while the server reads it */}
      {captured && phase !== 'idle' ? (
        <Animated.View entering={FadeIn.duration(150)} style={StyleSheet.absoluteFill}>
          <Image source={{ uri: captured }} style={StyleSheet.absoluteFill} resizeMode="cover" />
          <View style={styles.dim} />
        </Animated.View>
      ) : null}

      {/* Shutter flash */}
      <Animated.View pointerEvents="none" style={[StyleSheet.absoluteFill, styles.flash, flashStyle]} />

      {/* Top bar */}
      <View style={[styles.topBar, { paddingTop: insets.top + 10 }]}>
        <PressableScale style={styles.roundBtn} onPress={() => router.back()} scaleTo={0.9} accessibilityRole="button" accessibilityLabel="Close scanner">
          <X size={20} color="white" />
        </PressableScale>
        <Animated.View key={mode} entering={FadeIn.duration(200)} style={styles.titleChip}>
          {mode === 'card' ? <CreditCard size={14} color="white" /> : <QrCode size={14} color="white" />}
          <Text style={styles.titleChipText}>{MODE_META[mode].title}</Text>
        </Animated.View>
        <PressableScale
          style={[styles.roundBtn, torch && styles.roundBtnOn]}
          onPress={() => setTorch((t) => !t)}
          scaleTo={0.9}
          accessibilityRole="switch"
          accessibilityState={{ checked: torch }}
          accessibilityLabel="Flashlight"
        >
          {torch ? <Zap size={18} color="#0f172a" /> : <ZapOff size={18} color="white" />}
        </PressableScale>
      </View>

      {/* Viewfinder + honest hint */}
      <View style={styles.middle} pointerEvents="none">
        <Viewfinder mode={mode} found={qrState === 'contact'} busy={busy} maxWidth={screenW} />
        {phase === 'idle' ? (
          <Animated.View key={hint} entering={FadeInUp.duration(220)} exiting={FadeOut.duration(120)}
            style={[styles.hintPill, qrState === 'contact' && styles.hintPillFound, qrState === 'other' && styles.hintPillWarn]}>
            <View style={[styles.hintDot, qrState === 'contact' && { backgroundColor: GREEN }, qrState === 'other' && { backgroundColor: toneBg('#fbbf24') }]} />
            <Text style={styles.hintText}>{hint}</Text>
          </Animated.View>
        ) : null}
      </View>

      {/* Processing / result panels */}
      {phase === 'processing' ? (
        <Animated.View entering={FadeInUp.duration(260)} style={[styles.panel, { bottom: insets.bottom + 24 }]}>
          <ActivityIndicator color={theme.colors.brand[600]} />
          <View style={{ flex: 1 }}>
            <Text style={styles.panelTitle}>{mode === 'card' ? 'Reading the card with AI…' : 'Reading the QR code…'}</Text>
            <Text style={styles.panelSub}>
              {mode === 'card' ? `This can take a few seconds · ${elapsed}s` : `Creating the contact · ${elapsed}s`}
            </Text>
          </View>
        </Animated.View>
      ) : null}

      {phase === 'error' ? (
        <Animated.View entering={FadeInUp.duration(260)} style={[styles.panel, styles.panelCol, { bottom: insets.bottom + 24 }]}>
          <View style={styles.panelRow}>
            <View style={[styles.panelIcon, { backgroundColor: theme.colors.semantic.dangerBg }]}>
              <AlertCircle size={18} color={theme.colors.semantic.danger} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.panelTitle}>Couldn't read that</Text>
              <Text style={styles.panelSub}>{errorMsg}</Text>
            </View>
          </View>
          <View style={styles.panelActions}>
            <PressableScale containerStyle={{ flex: 1 }} style={styles.panelBtnGhost} onPress={pickImage} scaleTo={0.95}
              leadingIcon={<ImageIcon size={15} color={theme.colors.text.display} />} accessibilityRole="button">
              <Text style={styles.panelBtnGhostText}>From gallery</Text>
            </PressableScale>
            <PressableScale containerStyle={{ flex: 1 }} style={styles.panelBtn} onPress={reset} scaleTo={0.95}
              leadingIcon={<RotateCcw size={15} color="white" />} accessibilityRole="button">
              <Text style={styles.panelBtnText}>Try again</Text>
            </PressableScale>
          </View>
        </Animated.View>
      ) : null}

      {phase === 'duplicate' && dup ? (
        <Animated.View entering={FadeInUp.duration(260)} style={[styles.panel, styles.panelCol, { bottom: insets.bottom + 24 }]}>
          <View style={styles.panelRow}>
            <View style={[styles.panelIcon, { backgroundColor: theme.colors.semantic.warningBg }]}>
              <Copy size={18} color={theme.colors.semantic.warning} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.panelTitle}>Already in your contacts</Text>
              <Text style={styles.panelSub}>
                {dup.existingName}{dup.matchedOn ? ` · same ${dup.matchedOn}` : ''} — nothing new was added.
              </Text>
            </View>
          </View>
          <View style={styles.panelActions}>
            <PressableScale containerStyle={{ flex: 1 }} style={styles.panelBtnGhost} onPress={reset} scaleTo={0.95}
              leadingIcon={<RotateCcw size={15} color={theme.colors.text.display} />} accessibilityRole="button">
              <Text style={styles.panelBtnGhostText}>Scan another</Text>
            </PressableScale>
            <PressableScale containerStyle={{ flex: 1 }} style={styles.panelBtn}
              onPress={() => openContact(router, dup.contactId, '/scan')} scaleTo={0.95}
              trailingIcon={<ChevronRight size={15} color="white" />} accessibilityRole="button">
              <Text style={styles.panelBtnText}>Open contact</Text>
            </PressableScale>
          </View>
        </Animated.View>
      ) : null}

      {phase === 'queued' ? (
        <Animated.View entering={FadeInUp.duration(260)} style={[styles.panel, styles.panelCol, { bottom: insets.bottom + 24 }]}>
          <View style={styles.panelRow}>
            <View style={[styles.panelIcon, { backgroundColor: theme.colors.semantic.infoBg }]}>
              <CloudOff size={18} color={theme.colors.brand[600]} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.panelTitle}>Saved offline</Text>
              <Text style={styles.panelSub}>
                You're not connected. The photo is kept on this phone and the contact will be added automatically when you're back online.
              </Text>
            </View>
          </View>
          <View style={styles.panelActions}>
            <PressableScale containerStyle={{ flex: 1 }} style={styles.panelBtnGhost}
              onPress={() => router.push('/me/sync' as any)} scaleTo={0.95} accessibilityRole="button">
              <Text style={styles.panelBtnGhostText}>View queue</Text>
            </PressableScale>
            <PressableScale containerStyle={{ flex: 1 }} style={styles.panelBtn} onPress={reset} scaleTo={0.95}
              leadingIcon={<RotateCcw size={15} color="white" />} accessibilityRole="button">
              <Text style={styles.panelBtnText}>Scan another</Text>
            </PressableScale>
          </View>
        </Animated.View>
      ) : null}

      {/* Controls */}
      {phase === 'idle' ? (
        <Animated.View entering={FadeInUp.duration(260)} style={[styles.bottomArea, { paddingBottom: insets.bottom + 28 }]}>
          <Animated.View entering={FadeInUp.delay(60).duration(260)}>
            <ZoomControl zoom={zoom} onChange={setZoom} />
          </Animated.View>
          <ModeSwitch mode={mode} onChange={(m) => { setMode(m); setQrState('none'); }} disabled={busy} />
          <View style={styles.controlsRow}>
            <PressableScale style={styles.sideBtn} onPress={pickImage} scaleTo={0.9} accessibilityRole="button" accessibilityLabel="Choose a photo from the gallery">
              <ImageIcon size={20} color="white" />
            </PressableScale>

            <Animated.View style={shutterStyle}>
              <Pressable
                style={styles.shutterOuter}
                onPress={() => capture(mode)}
                onPressIn={() => { shutterScale.value = withSpring(0.9, { damping: 22, stiffness: 400, overshootClamping: true }); }}
                onPressOut={() => { shutterScale.value = withSpring(1, { damping: 22, stiffness: 220, overshootClamping: true }); }}
                accessibilityRole="button"
                accessibilityLabel={mode === 'card' ? 'Take photo of the card' : 'Capture the QR code'}
              >
                <View style={styles.shutterInner}>
                  {mode === 'qr' ? <QrCode size={26} color="#0f172a" /> : null}
                </View>
              </Pressable>
            </Animated.View>

            <PressableScale style={styles.sideBtn} onPress={() => setFacing((f) => (f === 'back' ? 'front' : 'back'))} scaleTo={0.9}
              accessibilityRole="button" accessibilityLabel="Switch camera">
              <SwitchCamera size={20} color="white" />
            </PressableScale>
          </View>
        </Animated.View>
      ) : null}
    </View>
  );
}

const CORNER = 28;
const styles = makeStyles(() => ({
  container: { flex: 1, backgroundColor: toneBg('#090a10') },
  dim: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(9,10,16,0.55)' },
  flash: { backgroundColor: '#ffffff' },

  topBar: { position: 'absolute', top: 0, left: 0, right: 0, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, zIndex: 5 },
  roundBtn: { width: 44, height: 44, borderRadius: 22, backgroundColor: 'rgba(0,0,0,0.45)', borderWidth: 1, borderColor: 'rgba(255,255,255,0.18)', alignItems: 'center', justifyContent: 'center' },
  roundBtnOn: { backgroundColor: '#ffffff' },
  titleChip: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: 'rgba(0,0,0,0.45)', borderWidth: 1, borderColor: 'rgba(255,255,255,0.18)', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 999 },
  titleChipText: { color: 'white', fontSize: 13, fontWeight: '700' },

  middle: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 22 },
  frame: { borderRadius: 18 },
  corner: { position: 'absolute', width: CORNER, height: CORNER, borderColor: '#ffffff' },
  cTL: { top: 0, left: 0, borderTopWidth: 4, borderLeftWidth: 4, borderTopLeftRadius: 18 },
  cTR: { top: 0, right: 0, borderTopWidth: 4, borderRightWidth: 4, borderTopRightRadius: 18 },
  cBL: { bottom: 0, left: 0, borderBottomWidth: 4, borderLeftWidth: 4, borderBottomLeftRadius: 18 },
  cBR: { bottom: 0, right: 0, borderBottomWidth: 4, borderRightWidth: 4, borderBottomRightRadius: 18 },
  scanLine: { position: 'absolute', left: 14, right: 14, top: 2, height: 2, borderRadius: 1, shadowColor: theme.colors.brand[300], shadowOpacity: 0.9, shadowRadius: 6, shadowOffset: { width: 0, height: 0 } },
  hintPill: { flexDirection: 'row', alignItems: 'center', gap: 8, maxWidth: '86%', backgroundColor: 'rgba(0,0,0,0.65)', borderWidth: 1, borderColor: 'rgba(255,255,255,0.15)', paddingHorizontal: 14, paddingVertical: 9, borderRadius: 999 },
  hintPillFound: { borderColor: GREEN },
  hintPillWarn: { borderColor: toneBg('#fbbf24') },
  hintDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: theme.colors.brand[300] },
  hintText: { color: toneFg('#e2e8f0'), fontSize: 13, fontWeight: '600', flexShrink: 1 },

  bottomArea: { position: 'absolute', left: 0, right: 0, bottom: 0, alignItems: 'center', gap: 24 },
  // zoom control: same glassy dark-on-camera look as the other camera controls (both modes)
  zoomPill: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: 'rgba(0,0,0,0.5)', borderWidth: 1, borderColor: 'rgba(255,255,255,0.18)', borderRadius: 999, padding: 4, marginBottom: -12 },
  zoomBtn: { width: 34, height: 34, borderRadius: 17, alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(255,255,255,0.12)' },
  zoomMid: { width: 104, alignItems: 'center', gap: 4, paddingVertical: 2 },
  zoomText: { color: 'white', fontSize: 12, fontWeight: '800' },
  zoomTrack: { width: 84, height: 4, borderRadius: 2, backgroundColor: 'rgba(255,255,255,0.25)', overflow: 'hidden' },
  zoomFill: { height: 4, borderRadius: 2, backgroundColor: '#ffffff' },
  modeSwitch: { flexDirection: 'row', width: 240, backgroundColor: 'rgba(0,0,0,0.55)', borderRadius: 999, padding: 4, borderWidth: 1, borderColor: 'rgba(255,255,255,0.15)' },
  modePill: { position: 'absolute', top: 4, bottom: 4, left: 4, borderRadius: 999, backgroundColor: '#ffffff' },
  modeSeg: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, paddingVertical: 9 },
  modeText: { color: 'rgba(255,255,255,0.75)', fontSize: 13, fontWeight: '800' },
  modeTextActive: { color: '#0f172a' }, // on the white pill in both modes
  controlsRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-evenly', width: '100%', paddingHorizontal: 24 },
  sideBtn: { width: 54, height: 54, borderRadius: 27, backgroundColor: 'rgba(0,0,0,0.45)', borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' },
  shutterOuter: { width: 82, height: 82, borderRadius: 41, borderWidth: 4, borderColor: '#ffffff', backgroundColor: 'rgba(255,255,255,0.12)', alignItems: 'center', justifyContent: 'center' },
  shutterInner: { width: 64, height: 64, borderRadius: 32, backgroundColor: '#ffffff', alignItems: 'center', justifyContent: 'center' },

  panel: { position: 'absolute', left: 16, right: 16, flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: toneBg('#ffffff'), borderRadius: 20, padding: 16, ...theme.shadows.active },
  panelCol: { flexDirection: 'column', alignItems: 'stretch' },
  panelRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 12 },
  panelIcon: { width: 36, height: 36, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  panelTitle: { fontSize: 15, fontWeight: '800', color: theme.colors.text.display },
  panelSub: { fontSize: 13, color: theme.colors.text.body, marginTop: 2, lineHeight: 18 },
  panelActions: { flexDirection: 'row', gap: 10, marginTop: 14 },
  panelBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, backgroundColor: theme.colors.brand[600], paddingVertical: 12, borderRadius: 999 },
  panelBtnText: { color: 'white', fontSize: 14, fontWeight: '800' },
  panelBtnGhost: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, backgroundColor: theme.colors.bg.surfaceHover, borderWidth: 1, borderColor: theme.colors.bg.divider, paddingVertical: 12, borderRadius: 999 },
  panelBtnGhostText: { color: theme.colors.text.display, fontSize: 14, fontWeight: '700' },

  permissionContainer: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 16, backgroundColor: toneBg('#090a10'), paddingHorizontal: 32 },
  permissionClose: { position: 'absolute', top: 56, left: 16 },
  permissionIcon: { width: 76, height: 76, borderRadius: 26, backgroundColor: theme.colors.brand[600], alignItems: 'center', justifyContent: 'center' },
  permissionTitle: { color: 'white', fontSize: 22, fontWeight: '900', textAlign: 'center' },
  permissionText: { color: 'rgba(255,255,255,0.75)', fontSize: 15, lineHeight: 22, textAlign: 'center' },
  permissionBtn: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: theme.colors.brand[600], paddingHorizontal: 22, paddingVertical: 13, borderRadius: 999 },
  permissionBtnText: { color: 'white', fontWeight: '800', fontSize: 15 },
  permissionLink: { color: theme.colors.fg.brand300, fontSize: 14, fontWeight: '700' },
}));
