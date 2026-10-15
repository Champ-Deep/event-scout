import React from 'react';
import { View } from 'react-native';
import { Text } from './Text';
import { tokens } from './tokens';

import { makeStyles } from './theme';
export interface ChipProps {
  label: string;
  band?: 'A' | 'B' | 'C' | 'D' | 'F';
  color?: string;
}

export const Chip: React.FC<ChipProps> = ({ label, band, color }) => {
  const chipColor = band ? tokens.colors.bands[band] : (color || tokens.colors.accent.blue);

  return (
    <View style={[styles.container, { borderColor: chipColor, backgroundColor: `${chipColor}20` }]}>
      <Text variant="caption" style={{ color: chipColor, fontWeight: '700' }}>
        {label}
      </Text>
    </View>
  );
};

const styles = makeStyles(() => ({
  container: {
    paddingHorizontal: tokens.spacing.sm,
    paddingVertical: 2,
    borderRadius: tokens.radii.sm,
    borderWidth: 1,
    alignSelf: 'flex-start',
  },
}));
