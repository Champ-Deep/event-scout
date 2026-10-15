import React from 'react';
import { View } from 'react-native';
import Animated from 'react-native-reanimated';
import { Text } from '../../ui/Text';
import { theme, makeStyles } from '../../ui/theme';
import { LucideIcon } from 'lucide-react-native';

interface HeaderPillProps {
  icon: LucideIcon;
  text: string;
  animStyle?: any;
  iconColor?: string;
  iconBackground?: string;
}

export const HeaderPill: React.FC<HeaderPillProps> = ({
  icon: Icon,
  text,
  animStyle,
  iconColor = theme.colors.text.display,
  iconBackground = theme.colors.brand[900],
}) => {
  return (
    <Animated.View style={[styles.headerPill, animStyle]}>
      <View style={[styles.headerIconContainer, { backgroundColor: iconBackground }]}>
        <Icon size={12} color={iconColor} />
      </View>
      <Text style={styles.headerPillText}>{text}</Text>
    </Animated.View>
  );
};

const styles = makeStyles(() => ({
  headerPill: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    backgroundColor: theme.colors.bg.surface,
    borderWidth: 1,
    borderColor: theme.colors.bg.divider,
    borderRadius: theme.radii.full,
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginBottom: theme.spacing.xl,
  },
  headerIconContainer: {
    marginRight: 6,
    borderRadius: theme.radii.xs,
    padding: 2,
  },
  headerPillText: {
    fontSize: theme.typography.sizes.subtext.fontSize,
    color: theme.colors.text.body,
    fontWeight: '500',
    fontFamily: theme.typography.fontFamily.mono,
  },
}));
