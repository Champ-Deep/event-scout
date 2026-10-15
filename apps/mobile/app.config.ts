import { ExpoConfig, ConfigContext } from 'expo/config';

// Store builds talk to the deployed backend; release Android blocks plain http, and a missing URL
// would silently fall back to localhost. Fail the build instead of shipping an app that can't sign in.
if (['production', 'preview'].includes(process.env.EAS_BUILD_PROFILE ?? '')) {
  const apiBase = process.env.EXPO_PUBLIC_API_BASE_URL ?? '';
  if (!/^https:\/\/[^/]+/.test(apiBase)) {
    throw new Error('EXPO_PUBLIC_API_BASE_URL must be the https URL of the deployed backend for release builds.');
  }
  if (!process.env.EXPO_PUBLIC_API_KEY) {
    throw new Error('EXPO_PUBLIC_API_KEY must be set (same value as the backend APP_API_KEY) for release builds.');
  }
}

export default ({ config }: ConfigContext): ExpoConfig => ({
  ...config,
  name: 'EventScout',
  slug: 'event-scout',
  version: '1.0.0',
  orientation: 'portrait',
  scheme: 'eventscout',
  userInterfaceStyle: 'automatic',
  // Brand assets: generated (violet brand gradient + white radar mark) from src/ui/theme.ts colours.
  icon: './assets/icon.png',
  ios: {
    supportsTablet: false,
    bundleIdentifier: 'com.eventscout.app',
    infoPlist: {
      NSCameraUsageDescription: 'EventScout uses the camera to scan attendee badges and business cards.',
      NSMicrophoneUsageDescription: 'EventScout uses the microphone so you can ask the AI assistant questions by voice.',
    },
  },
  android: {
    package: 'com.eventscout.app',
    permissions: ['CAMERA', 'RECORD_AUDIO'],
    adaptiveIcon: {
      foregroundImage: './assets/adaptive-icon.png', // mark within the central 66% safe zone
      backgroundColor: '#6d28d9', // brand 700
    },
  },
  plugins: [
    'expo-router',
    'expo-web-browser',
    'expo-dev-client',
    'expo-font',
    'expo-asset',
    'expo-sqlite',
    ['expo-splash-screen', {
      image: './assets/splash-icon.png',
      imageWidth: 200,
      resizeMode: 'contain',
      backgroundColor: '#fcfaff', // bg.app (light)
      dark: { image: './assets/splash-icon-dark.png', backgroundColor: '#0d0c14' }, // bg.app (dark)
    }],
    ['expo-camera', { cameraPermission: 'EventScout uses the camera to scan attendee badges and business cards.', microphonePermission: false, recordAudioAndroid: false }],
    ['expo-image-picker', { photosPermission: 'EventScout lets you pick a photo of a business card or a contact picture.', cameraPermission: 'EventScout uses the camera to scan attendee badges and business cards.' }],
    ['expo-audio', { microphonePermission: 'EventScout uses the microphone so you can ask the AI assistant questions by voice.' }],
    'expo-document-picker',
    'expo-sharing',
    ['expo-media-library', { photosPermission: 'EventScout saves your business card to your photos.', savePhotosPermission: 'EventScout saves your business card to your photos.', isAccessMediaLocationEnabled: false }],
  ],
  experiments: {
    typedRoutes: true,
  },
});
