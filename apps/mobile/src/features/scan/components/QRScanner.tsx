import React, { useState, useRef, useEffect } from 'react';
import { View, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';
import { theme, makeStyles } from '../../../ui/theme';
import { tokens } from '../../../ui/tokens';

interface QRScannerProps {
  onQRScanned: (data: string) => Promise<void>;
  onCancel: () => void;
  scanning?: boolean;
}

export const QRScanner: React.FC<QRScannerProps> = ({
  onQRScanned,
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

  useEffect(() => {
    // We removed the mock timeout logic here.
    // The actual scanning will be handled by the CameraView's onBarcodeScanned prop.
  }, [isCameraReady, scanning]);

  const handleQRDetected = async (data: string) => {
    try {
      await onQRScanned(data);
    } catch (error) {
      Alert.alert('Error', 'Failed to process QR code. Please try again.');
      console.error('QR scan error:', error);
    }
  };

  const handleManualScan = () => {
    Alert.prompt(
      'Enter QR Data',
      'Paste the QR code data or vCard text:',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Scan', 
          onPress: (data?: string) => data && handleQRDetected(data)
        }
      ]
    );
  };

  if (!permission) {
    return (
      <View style={styles.container}>
        <View style={styles.centered}>
          <ActivityIndicator size="large" color={theme.colors.semantic.success} />
          <Ionicons name="qr-code" size={64} color={theme.colors.text.muted} style={styles.qrIcon} />
          <ActivityIndicator size="large" color={theme.colors.semantic.success} />
        </View>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <View style={styles.centered}>
          <Ionicons name="qr-code-outline" size={64} color={theme.colors.semantic.danger} />
          <Ionicons name="qr-code-outline" size={64} color={theme.colors.semantic.danger} style={styles.qrIcon} />
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Ionicons name="camera" size={24} color={theme.colors.brand[600]} />
            <Ionicons name="camera" size={24} color={theme.colors.brand[600]} style={styles.permissionIcon} />
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
        barcodeScannerSettings={{
          barcodeTypes: ['qr'],
        }}
        onBarcodeScanned={({ data }) => {
          if (!scanning) {
            handleQRDetected(data);
          }
        }}
      >
        <View style={styles.overlay}>
          {/* QR Scanner overlay with frame */}
          <View style={styles.qrFrame}>
            <View style={[styles.corner, styles.topLeft]} />
            <View style={[styles.corner, styles.topRight]} />
            <View style={[styles.corner, styles.bottomLeft]} />
            <View style={[styles.corner, styles.bottomRight]} />
          </View>
          
          <View style={styles.instructions}>
            <Ionicons name="qr-code" size={32} color={theme.colors.text.inverse} />
            <Ionicons name="qr-code" size={32} color={theme.colors.text.inverse} style={styles.instructionsIcon} />
          </View>

          {/* Controls */}
          <View style={styles.controls}>
            <TouchableOpacity style={styles.cancelButton} onPress={onCancel}>
              <Ionicons name="close" size={24} color={theme.colors.text.inverse} />
            </TouchableOpacity>
            
            <TouchableOpacity
              style={styles.manualButton}
              onPress={handleManualScan}
              disabled={scanning}
            >
              <Ionicons name="text" size={20} color={theme.colors.brand[600]} />
              <Ionicons name="text" size={20} color={theme.colors.brand[600]} style={styles.manualIcon} />
            </TouchableOpacity>
            
            <View style={styles.scanIndicator}>
              {scanning ? (
                <ActivityIndicator size="large" color={theme.colors.semantic.success} />
              ) : (
                <Ionicons name="scan" size={32} color={theme.colors.semantic.success} />
              )}
            </View>
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
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
  },
  qrFrame: {
    position: 'absolute',
    top: '30%',
    left: '20%',
    width: '60%',
    height: '40%',
    borderWidth: 2,
    borderColor: tokens.colors.accent.emerald,
    borderRadius: tokens.radii.md,
  },
  corner: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderColor: tokens.colors.accent.emerald,
  },
  topLeft: {
    top: -2,
    left: -2,
    borderTopWidth: 4,
    borderLeftWidth: 4,
    borderTopLeftRadius: tokens.radii.md,
  },
  topRight: {
    top: -2,
    right: -2,
    borderTopWidth: 4,
    borderRightWidth: 4,
    borderTopRightRadius: tokens.radii.md,
  },
  bottomLeft: {
    bottom: -2,
    left: -2,
    borderBottomWidth: 4,
    borderLeftWidth: 4,
    borderBottomLeftRadius: tokens.radii.md,
  },
  bottomRight: {
    bottom: -2,
    right: -2,
    borderBottomWidth: 4,
    borderRightWidth: 4,
    borderBottomRightRadius: tokens.radii.md,
  },
  instructions: {
    position: 'absolute',
    top: '20%',
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  instructionsIcon: {
    marginTop: 8,
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
  manualButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: theme.colors.bg.surface,
    borderRadius: tokens.radii.md,
  },
  manualIcon: {
    marginLeft: 4,
  },
  scanIndicator: {
    width: 60,
    height: 60,
    justifyContent: 'center',
    alignItems: 'center',
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: theme.colors.bg.app,
  },
  qrIcon: {
    marginVertical: 20,
  },
  permissionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: theme.colors.bg.surface,
    borderRadius: tokens.radii.md,
    marginTop: 20,
  },
  permissionIcon: {
    marginLeft: 8,
  },
}));