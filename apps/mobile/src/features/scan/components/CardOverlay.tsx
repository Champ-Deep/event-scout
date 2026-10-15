import React from 'react';
import { View, Dimensions } from 'react-native';
import { tokens } from '../../../ui/tokens';

import { makeStyles } from '../../../ui/theme';
const { width } = Dimensions.get('window');
const FRAME_WIDTH = width * 0.85;
const FRAME_HEIGHT = FRAME_WIDTH * 0.63; // Standard business card / badge aspect ratio

export const CardOverlay: React.FC<{ active?: boolean }> = ({ active = true }) => {
  return (
    <View style={styles.overlay}>
      <View style={[styles.frame, active && styles.frameActive]}>
        <View style={[styles.corner, styles.topLeft]} />
        <View style={[styles.corner, styles.topRight]} />
        <View style={[styles.corner, styles.bottomLeft]} />
        <View style={[styles.corner, styles.bottomRight]} />
      </View>
    </View>
  );
};

const styles = makeStyles(() => ({
  overlay: {
    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, // absoluteFillObject was removed from RN
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0,0,0,0.3)',
  },
  frame: {
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    borderRadius: tokens.radii.lg,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.4)',
    position: 'relative',
  },
  frameActive: {
    borderColor: tokens.colors.accent.emerald,
  },
  corner: {
    position: 'absolute',
    width: 24,
    height: 24,
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
}));
