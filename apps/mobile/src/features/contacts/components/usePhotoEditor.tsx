import React, { useState } from 'react';
import { Platform } from 'react-native';
import { ImageIcon, Camera, Trash2 } from 'lucide-react-native';
import type { Contact } from '@es/types';
import { theme } from '../../../ui/theme';
import { errorMessage, type ErrorMessageOptions } from '../../../lib/api';
import { contactsFeature } from '..';
import { pickContactPhoto, PhotoSource } from '../photo';
import { OptionSheet, SheetOption } from './OptionSheet';

// How a failed photo upload is worded (see errorMessage).
const PHOTO_ERRORS: ErrorMessageOptions = {
  offline: "You're offline. Try again when connected.",
  tooLarge: 'That photo is too large.',
};
const PHOTO_FAILED = "Couldn't update the photo. Please try again.";

/**
 * Photo actions for one contact: choose from library, take a photo, or remove.
 * Saves through PATCH /api/contacts/:id { photo } and hands back the updated contact.
 */
export function usePhotoEditor(contact: Contact | null, onSaved: (updated: Contact) => void) {
  const [sheetOpen, setSheetOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const save = async (photo: string | null) => {
    if (!contact) return;
    setBusy(true);
    setError('');
    try {
      onSaved(await contactsFeature.updateContact(contact.id, { photo }));
    } catch (e: any) {
      setError(errorMessage(e, PHOTO_FAILED, PHOTO_ERRORS));
    } finally {
      setBusy(false);
    }
  };

  const choose = async (source: PhotoSource) => {
    setError('');
    try {
      const photo = await pickContactPhoto(source);
      if (photo) await save(photo);
    } catch (e: any) {
      setError(errorMessage(e, PHOTO_FAILED, PHOTO_ERRORS));
    }
  };

  const hasPhoto = !!contact?.photoUrl;
  const options: SheetOption[] = [
    { label: 'Choose from library', icon: <ImageIcon size={20} color={theme.colors.brand[600]} />, onPress: () => choose('library') },
    // Web has no camera capture through the picker.
    ...(Platform.OS !== 'web'
      ? [{ label: 'Take a photo', icon: <Camera size={20} color={theme.colors.brand[600]} />, onPress: () => choose('camera') }]
      : []),
    ...(hasPhoto
      ? [{ label: 'Remove photo', icon: <Trash2 size={20} color={theme.colors.semantic.danger} />, destructive: true, onPress: () => save(null) }]
      : []),
  ];

  const sheet = (
    <OptionSheet
      visible={sheetOpen}
      title={hasPhoto ? 'Contact photo' : 'Add a photo'}
      message="Shown on this contact's card and in your contacts list."
      options={options}
      onClose={() => setSheetOpen(false)}
    />
  );

  return { open: () => setSheetOpen(true), busy, error, sheet };
}
