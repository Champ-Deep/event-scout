import React from 'react';
import { View, ViewProps } from 'react-native';
import { theme, makeStyles } from './theme';

interface CardProps extends ViewProps {
  variant?: 'elevated' | 'dashed' | 'flat';
}

export function Card({ variant = 'elevated', style, children, ...props }: CardProps) {
  return (
    <View style={[styles.base, styles[variant], style]} {...props}>
      {children}
    </View>
  );
}

const styles = makeStyles(() => ({
  base: {
    borderRadius: theme.radii.lg,
    padding: theme.spacing.lg,
    marginBottom: theme.spacing.md,
  },
  elevated: {
    backgroundColor: theme.colors.bg.surface,
    borderWidth: 1,
    borderColor: theme.colors.bg.divider,
    ...theme.shadows.card,
  },
  dashed: {
    backgroundColor: theme.colors.bg.app,
    borderWidth: 1,
    borderColor: theme.colors.bg.dashed,
    borderStyle: 'dashed',
  },
  flat: {
    backgroundColor: theme.colors.bg.surface,
    borderWidth: 1,
    borderColor: theme.colors.bg.divider,
  },
}));
