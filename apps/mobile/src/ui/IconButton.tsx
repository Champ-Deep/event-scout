import React, { useState } from 'react';
import { Pressable, PressableProps, ViewStyle, StyleProp } from 'react-native';
import Animated, { useSharedValue, useAnimatedStyle, withSpring } from 'react-native-reanimated';
import { Icon, type IconName } from './Icon';
import { theme, makeStyles } from './theme';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable as any);

export interface IconButtonProps extends PressableProps {
  icon: IconName;
  variant?: 'primary' | 'secondary' | 'ghost' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  color?: string;
  style?: StyleProp<ViewStyle>;
}

export const IconButton: React.FC<IconButtonProps> = ({
  icon,
  variant = 'secondary',
  size = 'md',
  color,
  style,
  onPressIn,
  onPressOut,
  disabled,
  ...rest
}) => {
  const getSizing = () => {
    switch (size) {
      case 'sm': return { box: 24, icon: 14 };
      case 'lg': return { box: 40, icon: 20 };
      case 'md':
      default: return { box: 32, icon: 16 };
    }
  };

  const { box } = getSizing();
  const scale = useSharedValue(1);
  const [isPressed, setIsPressed] = useState(false);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const handlePressIn = (e: any) => {
    if (!disabled) {
      setIsPressed(true);
      scale.value = withSpring(0.92, { stiffness: 400, damping: 25, overshootClamping: true });
    }
    onPressIn?.(e);
  };

  const handlePressOut = (e: any) => {
    if (!disabled) {
      setIsPressed(false);
      scale.value = withSpring(1, { stiffness: 400, damping: 25, overshootClamping: true });
    }
    onPressOut?.(e);
  };

  const getStyles = (pressed: boolean) => {
    let bg: string = 'transparent';
    let iconColor: string = (color || theme.colors.text.display) as string;
    let border: string = 'transparent';

    switch (variant) {
      case 'primary':
        bg = pressed ? theme.colors.brand[900] : theme.colors.brand[800];
        iconColor = color || theme.colors.text.inverse;
        break;
      case 'secondary':
        bg = pressed ? theme.colors.bg.divider : theme.colors.brand[50];
        iconColor = color || theme.colors.brand[700];
        break;
      case 'outline':
        bg = pressed ? theme.colors.bg.surfaceHover : theme.colors.bg.surface;
        iconColor = color || theme.colors.text.display;
        border = theme.colors.bg.divider;
        break;
      case 'ghost':
        bg = pressed ? theme.colors.bg.surfaceHover : 'transparent';
        iconColor = color || theme.colors.text.body;
        break;
    }

    return { bg, iconColor, border };
  };

  return (
    <AnimatedPressable
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      style={[
        styles.base,
        { 
          backgroundColor: getStyles(isPressed).bg,
          width: box,
          height: box,
          borderRadius: box / 2,
        },
        getStyles(isPressed).border !== 'transparent' && { borderWidth: 1, borderColor: getStyles(isPressed).border },
        disabled && styles.disabled,
        animatedStyle,
        style,
      ]}
      {...rest}
    >
      {() => (
        <Icon 
          name={icon}
          size={size} 
          color={getStyles(isPressed).iconColor}
          strokeWidth={2.5} 
        />
      )}
    </AnimatedPressable>
  );
};

const styles = makeStyles(() => ({
  base: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  disabled: {
    opacity: 0.5,
  },
}));
