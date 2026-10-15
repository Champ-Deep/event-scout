import React, { useEffect } from 'react';
import { Pressable, StyleProp, ViewStyle } from 'react-native';
import Animated, { 
  useSharedValue, 
  useAnimatedStyle, 
  withSpring, 
  withTiming,
  Easing,
  withSequence,
  withDelay
} from 'react-native-reanimated';
import * as LucideIcons from 'lucide-react-native';
import { Icon, IconProps } from './Icon';

/**
 * Animated Icon component with micro-motions following microkit.io spec
 * Supports: press-scale, stroke draw-on, path morph animations
 */
export interface AnimatedIconProps extends IconProps {
  /** Animation type */
  animation?: 'press' | 'draw' | 'morph' | 'none';
  /** Whether to animate on press */
  animateOnPress?: boolean;
  /** Animation duration in milliseconds */
  duration?: number;
  /** Callback when pressed */
  onPress?: () => void;
  /** Additional container styles */
  containerStyle?: StyleProp<ViewStyle>;
}

/**
 * Animated Icon component for micro-motions
 * Following microkit.io specification for interactive icons
 */
export const AnimatedIcon: React.FC<AnimatedIconProps> = ({
  name,
  size = 'md',
  color = 'display',
  strokeWidth = 'normal',
  style,
  animation = 'press',
  animateOnPress = true,
  duration = 300,
  onPress,
  containerStyle,
  ...rest
}) => {
  const scale = useSharedValue(1);
  const strokeOpacity = useSharedValue(1);
  const rotation = useSharedValue(0);

  const animatedStyle = useAnimatedStyle(() => {
    const styles: any = {};
    
    switch (animation) {
      case 'press':
        styles.transform = [{ scale: scale.value }];
        break;
      case 'draw':
        styles.opacity = strokeOpacity.value;
        break;
      case 'morph':
        styles.transform = [{ rotate: `${rotation.value}deg` }];
        break;
    }
    
    return styles;
  });

  const handlePressIn = () => {
    if (!animateOnPress) return;
    
    switch (animation) {
      case 'press':
        scale.value = withSpring(0.85, { damping: 22, stiffness: 300, overshootClamping: true });
        break;
      case 'draw':
        strokeOpacity.value = withTiming(0.6, { duration: duration * 0.3 });
        break;
      case 'morph':
        rotation.value = withTiming(15, { 
          duration: duration * 0.5,
          easing: Easing.out(Easing.quad)
        });
        break;
    }
  };

  const handlePressOut = () => {
    if (!animateOnPress) return;
    
    switch (animation) {
      case 'press':
        scale.value = withSpring(1, { damping: 22, stiffness: 300, overshootClamping: true });
        break;
      case 'draw':
        strokeOpacity.value = withTiming(1, { duration: duration * 0.3 });
        break;
      case 'morph':
        rotation.value = withTiming(0, { 
          duration: duration * 0.5,
          easing: Easing.out(Easing.quad)
        });
        break;
    }
  };

  const handlePress = () => {
    onPress?.();
  };

  // Trigger animation on mount for specific animations
  useEffect(() => {
    if (animation === 'draw') {
      strokeOpacity.value = 0;
      strokeOpacity.value = withSequence(
        withDelay(100, withTiming(1, { duration: duration })),
      );
    }
  }, [animation, duration, strokeOpacity]);

  const iconElement = (
    <Animated.View style={[animatedStyle, style]}>
      <Icon
        name={name}
        size={size}
        color={color}
        strokeWidth={strokeWidth}
        {...rest}
      />
    </Animated.View>
  );

  if (!onPress) {
    return iconElement;
  }

  return (
    <Pressable
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      onPress={handlePress}
      style={containerStyle}
    >
      {iconElement}
    </Pressable>
  );
};

/**
 * Create an animated icon with specific animation preset
 */
export function createAnimatedIcon(
  iconName: keyof typeof LucideIcons,
  animationType: AnimatedIconProps['animation'] = 'press'
) {
  return (props: Omit<AnimatedIconProps, 'name' | 'animation'>) => (
    <AnimatedIcon name={iconName} animation={animationType} {...props} />
  );
}