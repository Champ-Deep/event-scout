import { useCallback, useEffect, useRef, useState } from 'react';
import { Platform } from 'react-native';
import {
  useAudioRecorder,
  useAudioRecorderState,
  RecordingPresets,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
} from 'expo-audio';

export type VoiceStatus = 'idle' | 'recording' | 'denied' | 'error';

const MAX_SECONDS = 120; // the server accepts about 2 minutes of audio
const MIN_SECONDS = 0.8;

/**
 * Records a voice question as .m4a (AAC) with live metering for the waveform.
 * `stop()` resolves to the file URI, or null when the recording was too short / failed.
 */
export function useVoiceRecorder() {
  const recorder = useAudioRecorder({ ...RecordingPresets.HIGH_QUALITY, isMeteringEnabled: true });
  const state = useAudioRecorderState(recorder, 80);
  const [status, setStatus] = useState<VoiceStatus>('idle');
  const [error, setError] = useState('');
  const busy = useRef(false);

  const seconds = (state.durationMillis || 0) / 1000;
  // Metering is dBFS (about -160 … 0). Map speech range (-50 … -5 dB) to 0 … 1.
  const level = state.metering == null ? 0 : Math.max(0, Math.min(1, (state.metering + 50) / 45));

  const start = useCallback(async () => {
    if (busy.current) return false;
    busy.current = true;
    setError('');
    try {
      const perm = await requestRecordingPermissionsAsync();
      if (!perm.granted) {
        setStatus('denied');
        setError('Microphone access is off. Allow it in Settings to ask by voice.');
        return false;
      }
      await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
      await recorder.prepareToRecordAsync();
      recorder.record();
      setStatus('recording');
      return true;
    } catch (e: any) {
      console.warn('[voice] start failed', e);
      setStatus('error');
      setError(Platform.OS === 'web'
        ? "Voice input isn't supported in this browser. Type your question instead."
        : "Couldn't start the microphone. Please try again.");
      return false;
    } finally {
      busy.current = false;
    }
  }, [recorder]);

  const finish = useCallback(async (keep: boolean): Promise<string | null> => {
    if (status !== 'recording') return null;
    const duration = (recorder.currentTime || seconds);
    try {
      await recorder.stop();
    } catch (e) {
      console.warn('[voice] stop failed', e);
    }
    setStatus('idle');
    try {
      await setAudioModeAsync({ allowsRecording: false, playsInSilentMode: true });
    } catch {}
    if (!keep) return null;
    if (duration < MIN_SECONDS) {
      setError('That was too short — hold on and speak your question.');
      return null;
    }
    return recorder.uri;
  }, [recorder, status, seconds]);

  const stop = useCallback(() => finish(true), [finish]);
  const cancel = useCallback(() => finish(false), [finish]);

  // Auto-send at the limit instead of cutting the user off silently.
  const autoStopped = useRef(false);
  const [autoUri, setAutoUri] = useState<string | null>(null);
  useEffect(() => {
    if (status === 'recording' && seconds >= MAX_SECONDS && !autoStopped.current) {
      autoStopped.current = true;
      stop().then((uri) => setAutoUri(uri));
    }
    if (status !== 'recording') autoStopped.current = false;
  }, [seconds, status, stop]);

  return {
    status,
    isRecording: status === 'recording',
    seconds,
    level,
    maxSeconds: MAX_SECONDS,
    error,
    clearError: () => setError(''),
    start,
    stop,
    cancel,
    /** Set when the recording hit the time limit and stopped itself; consume then clear. */
    autoUri,
    clearAutoUri: () => setAutoUri(null),
  };
}
