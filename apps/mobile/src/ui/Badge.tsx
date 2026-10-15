import React from 'react';
import { View, ViewStyle, StyleProp } from 'react-native';
import { Text } from './Text';
import { theme, makeStyles } from './theme';
import { StatusDot } from './StatusDot';
import { Icon, type IconName } from './Icon';

interface BadgeProps {
  label: string;
  variant?: 'success' | 'warning' | 'danger' | 'info' | 'primary' | 'muted' | 'outline';
  icon?: IconName;
  showDot?: boolean;
  style?: StyleProp<ViewStyle>;
}

export function Badge({ label, variant = 'primary', icon, showDot, style }: BadgeProps) {
  const getStyles = () => {
    switch (variant) {
      case 'success': return { bg: theme.colors.semantic.successBg, text: theme.colors.semantic.successText };
      case 'warning': return { bg: theme.colors.semantic.warningBg, text: theme.colors.semantic.warning };
      case 'danger': return { bg: theme.colors.semantic.dangerBg, text: theme.colors.semantic.danger };
      case 'info': return { bg: theme.colors.semantic.infoBg, text: theme.colors.brand[600] };
      case 'muted': return { bg: theme.colors.bg.surfaceHover, text: theme.colors.text.body };
      case 'outline': return { bg: 'transparent', text: theme.colors.semantic.danger, border: theme.colors.semantic.dangerBorder };
      case 'primary':
      default:
        return { bg: theme.colors.brand[50], text: theme.colors.brand[700] };
    }
  };

  const vStyles = getStyles();

  return (
    <View style={[
      styles.base, 
      { backgroundColor: vStyles.bg },
      vStyles.border && { borderWidth: 1, borderColor: vStyles.border },
      style
    ]}>
      {showDot && <StatusDot status={variant !== 'outline' ? variant : 'danger'} style={styles.dot} />}
      {icon && <Icon name={icon} size="xs" color={vStyles.text} style={styles.icon} />}
      <Text style={[styles.text, { color: vStyles.text }]}>{label}</Text>
    </View>
  );
}

const styles = makeStyles(() => ({
  base: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: theme.radii.sm,
  },
  text: {
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  dot: {
    marginRight: 6,
  },
  icon: {
    marginRight: 4,
  },
}));
