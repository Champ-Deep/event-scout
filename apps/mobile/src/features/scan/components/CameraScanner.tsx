import React, { useState, useRef, useEffect } from 'react';
import { View, TouchableOpacity, Alert, ActivityIndicator, Text } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { CardOverlay } from './CardOverlay';
import { Ionicons } from '@expo/vector-icons';
import { tokens } from '../../../ui/tokens';

import { makeStyles } from '../../../ui/theme';
interface CameraScannerProps {
  onScanComplete: (imageUri: string) => Promise<void>;
  onCancel: () => void;
  scanning?: boolean;
}

export const CameraScanner: React.FC<CameraScannerProps> = ({
  onScanComplete,
  onCancel,
  scanning = false,
}) => {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);

  useEffect(() => {
    if (!permission) {
      requestPermission();
    }
  }, [permission]);

  const handleTakePicture = async () => {
    if (!cameraRef.current || !isCameraReady || scanning) return;

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
        skipProcessing: false,
      });

      if (photo?.uri) {
        await onScanComplete(photo.uri);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to capture image. Please try again.');
      console.error('Camera error:', error);
    }
  };

  if (!permission) {
    return (
      <View style={styles.container}>
        <View style={styles.centered}>
          <Ionicons name="camera" size={64} color={tokens.colors.text.secondary} style={styles.cameraIcon} />
          <ActivityIndicator size="large" color={tokens.colors.accent.emerald} />
        </View>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <View style={styles.centered}>
          <Ionicons name="videocam-off" size={64} color={tokens.colors.accent.rose} style={styles.cameraIcon} />
          <Text style={styles.permissionTitle}>Camera access needed</Text>
          <Text style={styles.permissionText}>Allow camera access to scan business cards and badges.</Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Ionicons name="camera" size={24} color={tokens.colors.bg.primary} />
            <Text style={styles.permissionButtonText}>Allow camera</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="back"
        onCameraReady={() => setIsCameraReady(true)}
      >
        <View style={styles.overlay}>
          <CardOverlay active={isCameraReady && !scanning} />
          
          {/* Camera controls */}
          <View style={styles.controls}>
            <TouchableOpacity style={styles.cancelButton} onPress={onCancel}>
              <Ionicons name="close" size={24} color={tokens.colors.text.primary} />
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[styles.captureButton, scanning && styles.captureButtonDisabled]}
              onPress={handleTakePicture}
              disabled={scanning || !isCameraReady}
            >
              {scanning ? (
                <ActivityIndicator size="large" color={tokens.colors.text.primary} />
              ) : (
                <View style={styles.captureButtonInner} />
              )}
            </TouchableOpacity>
            
            <View style={styles.spacer} />
          </View>
        </View>
      </CameraView>
    </View>
  );
};

const styles = makeStyles(() => ({
  container: {
    flex: 1,
    backgroundColor: tokens.colors.bg.primary,
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'transparent',
  },
  controls: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  cancelButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: tokens.colors.accent.emerald,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: tokens.colors.text.primary,
  },
  captureButtonDisabled: {
    backgroundColor: tokens.colors.text.muted,
  },
  captureButtonInner: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: tokens.colors.text.primary,
  },
  spacer: {
    width: 50,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: tokens.colors.bg.primary,
  },
  cameraIcon: {
    marginVertical: 20,
  },
  permissionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: tokens.colors.text.primary,
    borderRadius: tokens.radii.md,
    marginTop: 20,
  },
  permissionIcon: {
    marginLeft: 8,
  },
  permissionTitle: {
    color: tokens.colors.text.primary,
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 8,
  },
  permissionText: {
    color: tokens.colors.text.secondary,
    fontSize: 15,
    textAlign: 'center',
    paddingHorizontal: 32,
  },
  permissionButtonText: {
    color: tokens.colors.bg.primary,
    fontSize: 16,
    fontWeight: '700',
    marginLeft: 8,
  },
}));