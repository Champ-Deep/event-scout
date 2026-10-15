import React, { useState } from 'react';
import { Pressable, ActivityIndicator, PressableProps, ViewStyle, StyleProp } from 'react-native';
import Animated, { useSharedValue, useAnimatedStyle, withSpring } from 'react-native-reanimated';
import { Icon, type IconName } from './Icon';
import { Text } from './Text';
import { theme, makeStyles } from './theme';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable as any);

export interface ButtonProps extends PressableProps {
  title: string;
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost' | 'outline';
  icon?: IconName;
  loading?: boolean;
  style?: StyleProp<ViewStyle>;
  fullWidth?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  title,
  variant = 'primary',
  icon: iconName,
  loading = false,
  disabled,
  fullWidth = false,
  style,
  onPressIn,
  onPressOut,
  ...rest
}) => {
  const scale = useSharedValue(1);
  const [isPressed, setIsPressed] = useState(false);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const handlePressIn = (e: any) => {
    if (!disabled && !loading) {
      setIsPressed(true);
      scale.value = withSpring(0.96, { stiffness: 400, damping: 25, overshootClamping: true });
    }
    onPressIn?.(e);
  };

  const handlePressOut = (e: any) => {
    if (!disabled && !loading) {
      setIsPressed(false);
      scale.value = withSpring(1, { stiffness: 400, damping: 25, overshootClamping: true });
    }
    onPressOut?.(e);
  };

  const getStyles = (pressed: boolean) => {
    let bg: string = 'transparent';
    let text: string = theme.colors.text.display as string;
    let border: string = 'transparent';

    switch (variant) {
      case 'primary':
        bg = pressed ? theme.colors.brand[900] : theme.colors.brand[800];
        text = theme.colors.text.inverse;
        break;
      case 'secondary':
        bg = pressed ? theme.colors.bg.divider : theme.colors.bg.surfaceHover;
        text = theme.colors.brand[800];
        break;
      case 'danger':
        bg = pressed ? theme.colors.semantic.dangerBorder : theme.colors.semantic.danger;
        text = theme.colors.text.inverse;
        break;
      case 'outline':
        bg = pressed ? theme.colors.bg.surfaceHover : theme.colors.bg.surface;
        text = theme.colors.text.display;
        border = theme.colors.bg.divider;
        break;
      case 'ghost':
        bg = pressed ? theme.colors.bg.surfaceHover : 'transparent';
        text = theme.colors.text.body;
        break;
    }

    return { bg, text, border };
  };

  return (
    <AnimatedPressable
      disabled={disabled || loading}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      style={[
        styles.base,
        { backgroundColor: getStyles(isPressed).bg },
        getStyles(isPressed).border !== 'transparent' && { borderWidth: 1, borderColor: getStyles(isPressed).border },
        (disabled || loading) && styles.disabled,
        fullWidth && { width: '100%' },
        animatedStyle,
        style,
      ]}
      {...rest}
    >
      {() => {
        const vStyles = getStyles(isPressed);
        return loading ? (
          <ActivityIndicator color={vStyles.text} />
        ) : (
          <>
             {iconName && <Icon name={iconName} size="md" color={vStyles.text} style={styles.icon} />}
            <Text
              style={[
                styles.text,
                { color: vStyles.text },
              ]}
            >
              {title}
            </Text>
          </>
        );
      }}
    </AnimatedPressable>
  );
};

const styles = makeStyles(() => ({
  base: {
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: theme.radii.full,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  disabled: {
    opacity: 0.5,
  },
  text: {
    fontSize: 15,
    fontWeight: '700',
  },
  icon: {
    marginRight: 8,
  },
}));
