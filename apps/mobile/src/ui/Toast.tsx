import React, { useEffect } from 'react';
import { View, Pressable } from 'react-native';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, withTiming, runOnJS } from 'react-native-reanimated';
import { Text } from './Text';
import { theme, toneBg, toneFg, makeStyles, live } from './theme';
import { CheckCircle, AlertCircle, AlertTriangle, X, Info } from 'lucide-react-native';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

interface ToastProps {
  message: string;
  type?: ToastType;
  duration?: number;
  onDismiss?: () => void;
  visible: boolean;
}

const TOAST_CONFIG = live(() => ({
  success: {
    icon: CheckCircle,
    backgroundColor: toneBg('#10b981'),
    borderColor: toneBg('#059669'),
    iconColor: '#FFFFFF',
  },
  error: {
    icon: AlertCircle,
    backgroundColor: toneBg('#ef4444'),
    borderColor: toneBg('#dc2626'),
    iconColor: '#FFFFFF',
  },
  warning: {
    icon: AlertTriangle,
    backgroundColor: toneBg('#f59e0b'),
    borderColor: toneBg('#d97706'),
    iconColor: '#FFFFFF',
  },
  info: {
    icon: Info,
    backgroundColor: theme.colors.brand[600],
    borderColor: theme.colors.fill.brand700,
    iconColor: '#FFFFFF',
  },
}));

export const Toast: React.FC<ToastProps> = ({
  message,
  type = 'info',
  duration = 4000,
  onDismiss,
  visible,
}) => {
  const translateY = useSharedValue(-100);
  const opacity = useSharedValue(0);
  const scale = useSharedValue(0.8);

  useEffect(() => {
    if (visible) {
      // Enter animation
      translateY.value = withSpring(0, { damping: 22, stiffness: 150, overshootClamping: true });
      opacity.value = withTiming(1, { duration: 300 });
      scale.value = withSpring(1, { damping: 22, stiffness: 200, overshootClamping: true });

      // Auto dismiss
      if (duration > 0) {
        const timer = setTimeout(() => {
          handleDismiss();
        }, duration);
        return () => clearTimeout(timer);
      }
    }
  }, [visible]);

  const handleDismiss = () => {
    // Exit animation
    translateY.value = withTiming(-100, { duration: 300 });
    opacity.value = withTiming(0, { duration: 300 });
    scale.value = withTiming(0.8, { duration: 300 }, () => {
      if (onDismiss) {
        runOnJS(onDismiss)();
      }
    });
  };

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: translateY.value }, { scale: scale.value }],
    opacity: opacity.value,
  }));

  if (!visible) return null;

  const config = TOAST_CONFIG[type];
  const IconComponent = config.icon;

  return (
    <Animated.View style={[styles.container, animatedStyle]}>
      <View
        style={[
          styles.toast,
          {
            backgroundColor: config.backgroundColor,
            borderColor: config.borderColor,
          },
        ]}
      >
        <View style={styles.iconContainer}>
          <IconComponent size={20} color={config.iconColor} strokeWidth={2.5} />
        </View>
        <Text style={styles.message} numberOfLines={2}>
          {message}
        </Text>
        <Pressable onPress={handleDismiss} style={styles.closeButton} hitSlop={8}>
          <X size={18} color={toneFg('#FFFFFF')} strokeWidth={2.5} />
        </Pressable>
      </View>
    </Animated.View>
  );
};

const styles = makeStyles(() => ({
  container: {
    position: 'absolute',
    top: 60,
    left: 16,
    right: 16,
    zIndex: 9999,
  },
  toast: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: theme.radii.lg,
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderWidth: 1,
    ...theme.shadows.active,
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 8,
  },
  iconContainer: {
    marginRight: 12,
  },
  message: {
    flex: 1,
    color: toneFg('#FFFFFF'),
    fontSize: 14,
    fontWeight: '600',
    lineHeight: 20,
  },
  closeButton: {
    marginLeft: 8,
    padding: 4,
  },
}));
