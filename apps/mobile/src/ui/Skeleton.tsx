import React from 'react';
import { View, DimensionValue } from 'react-native';
import { tokens } from './tokens';

import { makeStyles } from './theme';
export interface SkeletonProps {
  width?: DimensionValue;
  height?: DimensionValue;
  borderRadius?: number;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  width = '100%',
  height = 20,
  borderRadius = tokens.radii.sm,
}) => {
  return (
    <View
      style={[
        styles.skeleton,
        { width, height, borderRadius },
      ]}
    />
  );
};

const styles = makeStyles(() => ({
  skeleton: {
    backgroundColor: tokens.colors.bg.card,
    opacity: 0.6,
  },
}));
