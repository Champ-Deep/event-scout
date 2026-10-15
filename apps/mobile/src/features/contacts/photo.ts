import { Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export type PhotoSource = 'library' | 'camera';

export class PhotoPermissionError extends Error {}

/**
 * Let the user pick (or take) a square contact photo.
 * Resolves with a data URI ready for PATCH /api/contacts/:id { photo }, or null if cancelled.
 * The backend re-encodes it to a ≤512px JPEG, so a moderate quality here keeps uploads small.
 */
export async function pickContactPhoto(source: PhotoSource): Promise<string | null> {
  if (source === 'camera') {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      throw new PhotoPermissionError('Camera access is off. Enable it in Settings to take a contact photo.');
    }
  } else if (Platform.OS !== 'web') {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      throw new PhotoPermissionError('Photo library access is off. Enable it in Settings to choose a contact photo.');
    }
  }

  const options: ImagePicker.ImagePickerOptions = {
    mediaTypes: ['images'],
    allowsEditing: true,
    aspect: [1, 1],
    quality: 0.7,
    base64: true,
  };
  const result = source === 'camera'
    ? await ImagePicker.launchCameraAsync(options)
    : await ImagePicker.launchImageLibraryAsync(options);

  if (result.canceled || !result.assets?.[0]) return null;
  const asset = result.assets[0];
  if (asset.base64) {
    return `data:${asset.mimeType || 'image/jpeg'};base64,${asset.base64}`;
  }
  // On web the picker hands back a data URI directly.
  if (asset.uri?.startsWith('data:')) return asset.uri;
  throw new Error("Couldn't read that image. Please try another one.");
}
