import React from 'react';
import { View, ViewStyle } from 'react-native';
import { theme, makeStyles } from './theme';

interface DividerProps {
  variant?: 'solid' | 'dashed';
  style?: ViewStyle;
}

export function Divider({ variant = 'solid', style }: DividerProps) {
  return (
    <View style={[styles.base, styles[variant], style]} />
  );
}

const styles = makeStyles(() => ({
  base: {
    height: 1,
    width: '100%',
    marginVertical: theme.spacing.md,
  },
  solid: {
    backgroundColor: theme.colors.bg.divider,
  },
  dashed: {
    borderBottomWidth: 1,
    borderColor: theme.colors.bg.divider,
    borderStyle: 'dashed',
  },
}));
