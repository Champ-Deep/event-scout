import React from 'react';
import { ActivityIndicator, Pressable } from 'react-native';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, withTiming, FadeIn, FadeOut } from 'react-native-reanimated';
import { Text } from '../../ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../ui/theme';
import Svg, { Path, Rect } from 'react-native-svg';

// LinkedIn Icon
const LinkedInIcon = ({ size = 20, color = "#FFFFFF" }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <Rect width="24" height="24" rx="4" fill={color} />
    <Path d="M7.5 9.5H5V18.5H7.5V9.5Z" fill={toneFg('#0A66C2')} />
    <Path
      d="M6.25 5C5.56 5 5 5.56 5 6.25C5 6.94 5.56 7.5 6.25 7.5C6.94 7.5 7.5 6.94 7.5 6.25C7.5 5.56 6.94 5 6.25 5Z"
      fill={toneFg('#0A66C2')}
    />
    <Path
      d="M14 9.5C12.62 9.5 11.5 10.12 11 11V9.5H8.5V18.5H11V13.75C11 12.51 11.51 12 12.5 12C13.49 12 14 12.51 14 13.75V18.5H16.5V13.25C16.5 10.93 15.57 9.5 14 9.5Z"
      fill={toneFg('#0A66C2')}
    />
  </Svg>
);

interface SocialButtonProps {
  onPress: () => void;
  onPressIn?: () => void;
  onPressOut?: () => void;
  disabled?: boolean;
  animStyle?: any;
  text?: string;
  /** Shows a spinner + `loadingText` while the LinkedIn flow is running. */
  loading?: boolean;
  loadingText?: string;
}

export const SocialButton: React.FC<SocialButtonProps> = ({
  onPress,
  onPressIn,
  onPressOut,
  disabled = false,
  animStyle,
  text = 'Continue with LinkedIn',
  loading = false,
  loadingText = 'Connecting to LinkedIn…',
}) => {
  const buttonScale = useSharedValue(1);
  const dim = useSharedValue(1);

  React.useEffect(() => {
    dim.value = withTiming(loading ? 0.92 : disabled ? 0.55 : 1, { duration: 220 });
  }, [loading, disabled]);

  const handlePressIn = () => {
    buttonScale.value = withSpring(0.96, { damping: 22, stiffness: 400, overshootClamping: true });
    onPressIn?.();
  };

  const handlePressOut = () => {
    buttonScale.value = withSpring(1, { damping: 22, stiffness: 200, overshootClamping: true });
    onPressOut?.();
  };

  const buttonAnimStyle = useAnimatedStyle(() => ({
    transform: [{ scale: buttonScale.value }],
  }));

  const dimStyle = useAnimatedStyle(() => ({ opacity: dim.value }));

  return (
    <Animated.View style={[animStyle, buttonAnimStyle]}>
      <Animated.View style={dimStyle}>
        <Pressable
          style={styles.linkedinButton}
          onPress={onPress}
          onPressIn={handlePressIn}
          onPressOut={handlePressOut}
          disabled={disabled || loading}
          accessibilityRole="button"
          accessibilityState={{ busy: loading, disabled: disabled || loading }}
        >
          <Animated.View key={loading ? 'busy' : 'idle'} entering={FadeIn.duration(200)} exiting={FadeOut.duration(150)} style={styles.row}>
            {loading
              ? <ActivityIndicator size="small" color={toneFg('#FFFFFF')} />
              : <LinkedInIcon size={20} color={toneFg('#FFFFFF')} />}
            <Text style={styles.linkedinButtonText}>{loading ? loadingText : text}</Text>
          </Animated.View>
        </Pressable>
      </Animated.View>
    </Animated.View>
  );
};

const styles = makeStyles(() => ({
  linkedinButton: {
    backgroundColor: toneBg('#0A66C2'),
    borderRadius: theme.radii.full,
    paddingVertical: 14,
    paddingHorizontal: 28,
    minHeight: 52,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: theme.spacing.lg,
    ...theme.shadows.card,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  linkedinButtonText: {
    color: toneFg('#FFFFFF'),
    fontSize: 15,
    fontWeight: '700',
  },
}));
