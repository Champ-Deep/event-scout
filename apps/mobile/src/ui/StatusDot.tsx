import React from 'react';
import { View, ViewStyle } from 'react-native';
import { theme, makeStyles } from './theme';

interface StatusDotProps {
  status?: 'success' | 'warning' | 'danger' | 'info' | 'primary' | 'muted';
  style?: ViewStyle;
}

export function StatusDot({ status = 'primary', style }: StatusDotProps) {
  const getBackgroundColor = () => {
    switch (status) {
      case 'success': return theme.colors.semantic.success;
      case 'warning': return theme.colors.semantic.warning;
      case 'danger': return theme.colors.semantic.danger;
      case 'info': return theme.colors.semantic.info;
      case 'muted': return theme.colors.text.muted;
      case 'primary':
      default:
        return theme.colors.brand[600];
    }
  };

  return (
    <View style={[styles.dot, { backgroundColor: getBackgroundColor() }, style]} />
  );
}

const styles = makeStyles(() => ({
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
}));
