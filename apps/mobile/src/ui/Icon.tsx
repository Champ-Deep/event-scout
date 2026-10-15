import React from 'react';
import { LucideIcon as LucideIconType } from 'lucide-react-native';
import { StyleProp, ViewStyle } from 'react-native';
import * as LucideIcons from 'lucide-react-native';
import { theme } from './theme';

/**
 * Icon component following microkit.io specification
 * - Uses Lucide as the base glyph family
 * - Stroke-based icons with consistent stroke width
 * - Size and strokeWidth configured through theme/tokens
 * - Identical names across mobile and admin apps
 */
export interface IconProps {
  /** Lucide icon name (must match LucideIcons export) */
  name: keyof typeof LucideIcons;
  /** Size in pixels - uses theme icon sizing */
  size?: keyof typeof theme.icon.sizes;
  /** Color - uses theme colors */
  color?: keyof typeof theme.colors.text | string;
  /** Stroke width - uses theme icon stroke widths */
  strokeWidth?: keyof typeof theme.icon.strokeWidths | number;
  /** Additional styles */
  style?: StyleProp<ViewStyle>;
  /** Whether to animate on press (micro-motion) */
  animated?: boolean;
}

/**
 * Unified Icon component for EventScout
 * Follows microkit.io specification for consistent icon system
 */
export const Icon: React.FC<IconProps> = ({
  name,
  size = 'md',
  color = 'display',
  strokeWidth = 'normal',
  style,
  animated = false,
}) => {
  // Handle case where name is not a string (e.g., an object or React component)
  if (typeof name !== 'string') {
    console.warn(`Icon name must be a string, got ${typeof name}`);
    return null;
  }
  
  const IconComponent = LucideIcons[name] as LucideIconType;
  
  if (!IconComponent) {
    console.warn(`Icon "${name}" not found in Lucide icons`);
    return null;
  }

  const iconSize = theme.icon.sizes[size];
  const iconColor = typeof color === 'string' && color in theme.colors.text 
    ? theme.colors.text[color as keyof typeof theme.colors.text]
    : color;
  const iconStrokeWidth = typeof strokeWidth === 'string'
    ? theme.icon.strokeWidths[strokeWidth]
    : strokeWidth;

  return (
    <IconComponent
      size={iconSize}
      color={iconColor as string}
      strokeWidth={iconStrokeWidth}
      style={style}
    />
  );
};

/**
 * Helper to get Lucide icon by name
 */
export function getLucideIcon(name: keyof typeof LucideIcons): LucideIconType | null {
  return LucideIcons[name] as LucideIconType || null;
}

/**
 * Type for Lucide icon names (for better TypeScript support)
 */
export type IconName = keyof typeof LucideIcons;