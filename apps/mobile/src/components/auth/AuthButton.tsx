import React from 'react';
import { ActivityIndicator, Pressable, View } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  FadeIn,
  FadeOut,
} from 'react-native-reanimated';
import { Text } from '../../ui/Text';
import { theme, toneFg, makeStyles } from '../../ui/theme';
import { ArrowRight } from 'lucide-react-native';
import { useSlowHint } from './useSlowHint';

interface AuthButtonProps {
  text: string;
  onPress: () => void;
  disabled?: boolean;
  animStyle?: any;
  /** Shows a spinner + `loadingText` instead of the label (and a hint if it takes a while). */
  loading?: boolean;
  loadingText?: string;
}

export const AuthButton: React.FC<AuthButtonProps> = ({
  text,
  onPress,
  disabled = false,
  animStyle,
  loading = false,
  loadingText,
}) => {
  const buttonScale = useSharedValue(1);
  const arrowX = useSharedValue(0);
  const dim = useSharedValue(1);
  const slow = useSlowHint(loading);

  React.useEffect(() => {
    // Busy: a touch dimmer. Disabled by another action (e.g. LinkedIn in progress): clearly faded.
    dim.value = withTiming(loading ? 0.92 : disabled ? 0.55 : 1, { duration: 220 });
  }, [loading, disabled]);

  const handlePressIn = () => {
    buttonScale.value = withSpring(0.96, { damping: 22, stiffness: 400, overshootClamping: true });
    arrowX.value = withSpring(4, { damping: 22, stiffness: 400, overshootClamping: true });
  };

  const handlePressOut = () => {
    buttonScale.value = withSpring(1, { damping: 22, stiffness: 200, overshootClamping: true });
    arrowX.value = withSpring(0, { damping: 22, stiffness: 200, overshootClamping: true });
  };

  const buttonAnimStyle = useAnimatedStyle(() => ({
    transform: [{ scale: buttonScale.value }],
  }));

  const dimStyle = useAnimatedStyle(() => ({ opacity: dim.value }));

  const arrowAnimStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: arrowX.value }],
  }));

  return (
    <Animated.View style={[animStyle, buttonAnimStyle]}>
      <Animated.View style={dimStyle}>
        <Pressable
          style={styles.submitButton}
          onPress={onPress}
          onPressIn={handlePressIn}
          onPressOut={handlePressOut}
          disabled={disabled || loading}
          accessibilityRole="button"
          accessibilityState={{ busy: loading, disabled: disabled || loading }}
          accessibilityLabel={loading ? loadingText || text : text}
        >
          {loading ? (
            <Animated.View key="busy" entering={FadeIn.duration(200)} exiting={FadeOut.duration(150)} style={styles.row}>
              <ActivityIndicator size="small" color={toneFg('#FFFFFF')} />
              <Text style={styles.submitButtonText}>{loadingText || text}</Text>
            </Animated.View>
          ) : (
            <Animated.View key="idle" entering={FadeIn.duration(200)} exiting={FadeOut.duration(150)} style={styles.row}>
              <Text style={styles.submitButtonText}>{text}</Text>
              <Animated.View style={arrowAnimStyle}>
                <ArrowRight size={20} color={toneFg('#FFFFFF')} strokeWidth={2.5} />
              </Animated.View>
            </Animated.View>
          )}
        </Pressable>
      </Animated.View>
      <View style={styles.hintSlot}>
        {slow && (
          <Animated.View entering={FadeIn.duration(260)} exiting={FadeOut.duration(150)}>
            <Text style={styles.hint}>Still working… the server can take a few seconds to respond.</Text>
          </Animated.View>
        )}
      </View>
    </Animated.View>
  );
};

const styles = makeStyles(() => ({
  submitButton: {
    backgroundColor: theme.colors.fill.brand900,
    borderRadius: theme.radii.full,
    paddingVertical: 16,
    paddingHorizontal: 28,
    minHeight: 56,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 8,
    ...theme.shadows.active,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  submitButtonText: {
    color: toneFg('#FFFFFF'),
    fontSize: 16,
    fontWeight: '700',
  },
  // Keeps the space under the button stable so the layout doesn't jump when the hint appears.
  hintSlot: {
    minHeight: theme.spacing.lg,
    justifyContent: 'center',
    paddingTop: 6,
  },
  hint: {
    textAlign: 'center',
    fontSize: 12,
    color: theme.colors.text.muted,
  },
}));
