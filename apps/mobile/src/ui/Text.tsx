import React from 'react';
import { Text as RNText, TextProps as RNTextProps, StyleSheet, TextStyle } from 'react-native';
import { tokens } from './tokens';

export interface TextProps extends RNTextProps {
  variant?: 'title' | 'subtitle' | 'body' | 'caption';
  color?: 'primary' | 'secondary' | 'muted' | 'emerald' | 'rose';
}

export const Text: React.FC<TextProps> = ({
  variant = 'body',
  color = 'primary',
  style,
  children,
  maxFontSizeMultiplier = 1.3,
  ...rest
}) => {
  const variantStyle = tokens.typography[variant] as TextStyle;
  const own = (StyleSheet.flatten(style) || {}) as TextStyle;

  // A screen that sets a larger fontSize without a lineHeight used to inherit the variant's
  // lineHeight (20 for body), which clips big text — e.g. a 32px title drawn in a 20px line.
  // Only in that case, use a line height that fits the font; everything else is unchanged.
  let lineHeightFix: TextStyle | null = null;
  if (own.fontSize && own.lineHeight == null && variantStyle.lineHeight != null
      && own.fontSize * 1.2 > variantStyle.lineHeight) {
    lineHeightFix = { lineHeight: Math.round(own.fontSize * 1.25) };
  }

  return (
    <RNText
      style={[
        variantStyle,
        { color: color === 'primary' ? tokens.colors.text.primary :
                 color === 'secondary' ? tokens.colors.text.secondary :
                 color === 'muted' ? tokens.colors.text.muted :
                 color === 'emerald' ? tokens.colors.accent.emerald :
                 tokens.colors.accent.rose },
        style,
        lineHeightFix,
      ]}
      // A phone set to a very large system font used to blow tiles and headers past the screen
      // edge; text still grows with the setting, up to 1.3×. Pass a value to override.
      maxFontSizeMultiplier={maxFontSizeMultiplier}
      {...rest}
    >
      {children}
    </RNText>
  );
};
