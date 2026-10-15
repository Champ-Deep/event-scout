import React, { useEffect } from 'react';
import { View, ImageBackground } from 'react-native';
import { useRouter } from 'expo-router';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, withTiming, withDelay } from 'react-native-reanimated';
import { Text } from '../../src/ui/Text';
import { Badge } from '../../src/ui/Badge';
import { theme, toneBg, toneFg, makeStyles } from '../../src/ui/theme';
import { ArrowRight } from 'lucide-react-native';


export default function WelcomeScreen() {
  const router = useRouter();

  // Animations
  const badgeScale = useSharedValue(0.8);
  const badgeOpacity = useSharedValue(0);
  const titleScale = useSharedValue(0.9);
  const titleOpacity = useSharedValue(0);
  const subtitleOpacity = useSharedValue(0);
  const statsScale = useSharedValue(0.95);
  const statsOpacity = useSharedValue(0);
  const buttonScale = useSharedValue(0.9);
  const buttonOpacity = useSharedValue(0);
  const footerOpacity = useSharedValue(0);

  // Microkit-style button animation
  const buttonPressScale = useSharedValue(1);
  const arrowTranslateX = useSharedValue(0);

  useEffect(() => {
    // Staggered entrance animations
    badgeScale.value = withDelay(100, withSpring(1, { damping: 22, stiffness: 150, overshootClamping: true }));
    badgeOpacity.value = withDelay(100, withTiming(1, { duration: 400 }));

    titleScale.value = withDelay(300, withSpring(1, { damping: 22, stiffness: 150, overshootClamping: true }));
    titleOpacity.value = withDelay(300, withTiming(1, { duration: 400 }));

    subtitleOpacity.value = withDelay(500, withTiming(1, { duration: 400 }));

    statsScale.value = withDelay(700, withSpring(1, { damping: 22, stiffness: 150, overshootClamping: true }));
    statsOpacity.value = withDelay(700, withTiming(1, { duration: 400 }));

    buttonScale.value = withDelay(900, withSpring(1, { damping: 22, stiffness: 150, overshootClamping: true }));
    buttonOpacity.value = withDelay(900, withTiming(1, { duration: 400 }));

    footerOpacity.value = withDelay(1100, withTiming(1, { duration: 400 }));
  }, []);

  const badgeAnimStyle = useAnimatedStyle(() => ({
    transform: [{ scale: badgeScale.value }],
    opacity: badgeOpacity.value,
  }));

  const titleAnimStyle = useAnimatedStyle(() => ({
    transform: [{ scale: titleScale.value }],
    opacity: titleOpacity.value,
  }));

  const subtitleAnimStyle = useAnimatedStyle(() => ({
    opacity: subtitleOpacity.value,
  }));

  const statsAnimStyle = useAnimatedStyle(() => ({
    transform: [{ scale: statsScale.value }],
    opacity: statsOpacity.value,
  }));

  const buttonAnimStyle = useAnimatedStyle(() => ({
    transform: [{ scale: buttonScale.value * buttonPressScale.value }],
    opacity: buttonOpacity.value,
  }));

  const arrowAnimStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: arrowTranslateX.value }],
  }));

  const footerAnimStyle = useAnimatedStyle(() => ({
    opacity: footerOpacity.value,
  }));

  const handleButtonPressIn = () => {
    // Animation on touch down (press-in)
    buttonPressScale.value = withSpring(0.96, { damping: 22, stiffness: 400, overshootClamping: true });
    arrowTranslateX.value = withSpring(4, { damping: 22, stiffness: 400, overshootClamping: true });
  };

  const handleButtonPressOut = () => {
    // Animation on release (press-out)
    buttonPressScale.value = withSpring(1, { damping: 22, stiffness: 200, overshootClamping: true });
    arrowTranslateX.value = withSpring(0, { damping: 22, stiffness: 200, overshootClamping: true });
  };

  const handleButtonPress = () => {
    router.push('/register');
  };

  return (
    <View style={styles.container}>
      <ImageBackground
        source={require('../../assets/landing_bg.jpg')}
        style={styles.imageBackground}
        resizeMode="cover"
      >
        <View style={styles.gradientOverlay} />
      </ImageBackground>

      <View style={styles.cardContainer}>
        <Animated.View style={[styles.badgeWrapper, badgeAnimStyle]}>
          <Badge label="ENTERPRISE EVENT COPILOT" variant="primary" showDot />
        </Animated.View>

        <Animated.Text style={[styles.title, titleAnimStyle]}>
          Turn Event Chaos{'\n'}Into Sales Pipeline
        </Animated.Text>

        <Animated.Text style={[styles.subtitle, subtitleAnimStyle]}>
          Scan badges, identify high-value buyers, auto-score against your ICP, and navigate event halls—all in real time.
        </Animated.Text>

        <Animated.View style={[styles.statsContainer, statsAnimStyle]}>
          <View style={styles.statBox}>
            <Text style={styles.statValue}>Scan</Text>
            <Text style={styles.statLabel}>cards & badges</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statBox}>
            <Text style={[styles.statValue, { color: theme.colors.brand[700] }]}>AI</Text>
            <Text style={styles.statLabel}>lead scoring</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statBox}>
            <Text style={styles.statValue}>Offline</Text>
            <Text style={styles.statLabel}>capture works</Text>
          </View>
        </Animated.View>

        {/* Custom animated button with arrow */}
        <Animated.View style={[{ width: '100%' }, buttonAnimStyle]}>
          <Animated.View
            onTouchStart={handleButtonPressIn}
            onTouchEnd={handleButtonPressOut}
            onTouchCancel={handleButtonPressOut}
          >
            <Animated.View style={styles.customButton} onStartShouldSetResponder={() => true} onResponderRelease={handleButtonPress}>
              <Text style={styles.customButtonText}>Get Started</Text>
              <Animated.View style={arrowAnimStyle}>
                <ArrowRight size={20} color={toneFg('#FFFFFF')} strokeWidth={2.5} />
              </Animated.View>
            </Animated.View>
          </Animated.View>
        </Animated.View>

        <Animated.View style={footerAnimStyle}>
          <Text style={styles.footerText}>
            Already have an account?{' '}
            <Text style={styles.footerLink} onPress={() => router.push('/login')}>Log in</Text>
          </Text>
        </Animated.View>
      </View>
    </View>
  );
}

const styles = makeStyles(() => ({
  container: {
    flex: 1,
    backgroundColor: toneBg('#09090b'),
  },
  imageBackground: {
    flex: 1,
    width: '100%',
    justifyContent: 'flex-end',
  },
  gradientOverlay: {
    position: 'absolute',
    top: 0, left: 0, right: 0, bottom: 0,
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
  cardContainer: {
    backgroundColor: theme.colors.bg.surface,
    borderTopLeftRadius: 36,
    borderTopRightRadius: 36,
    paddingHorizontal: theme.spacing.lg,
    paddingTop: 36,
    paddingBottom: 48,
    marginTop: -30,
    ...theme.shadows.card,
    shadowOffset: { width: 0, height: -10 },
    shadowOpacity: 0.1,
    shadowRadius: 20,
    elevation: 20,
  },
  badgeWrapper: {
    alignItems: 'center',
    marginBottom: theme.spacing.lg,
  },
  title: {
    color: theme.colors.text.display,
    fontSize: 34,
    fontWeight: '900',
    textAlign: 'center',
    lineHeight: 40,
    marginBottom: theme.spacing.md,
  },
  subtitle: {
    color: theme.colors.text.body,
    fontSize: theme.typography.sizes.body.fontSize,
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: theme.spacing.xl,
    paddingHorizontal: 8,
  },
  statsContainer: {
    flexDirection: 'row',
    backgroundColor: theme.colors.brand[50],
    borderRadius: theme.radii.lg,
    paddingVertical: theme.spacing.md,
    marginBottom: theme.spacing.xl,
    borderWidth: 1,
    borderColor: theme.colors.brand[100],
  },
  statBox: {
    flex: 1,
    alignItems: 'center',
  },
  statDivider: {
    width: 1,
    backgroundColor: theme.colors.bg.divider,
    height: '100%',
  },
  statValue: {
    color: theme.colors.text.display,
    fontSize: theme.typography.sizes.headline.fontSize,
    fontWeight: '800',
    marginBottom: 4,
  },
  statLabel: {
    color: theme.colors.text.body,
    fontSize: 11,
    fontWeight: '500',
  },
  customButton: {
    backgroundColor: theme.colors.fill.brand900,
    borderRadius: theme.radii.full,
    paddingVertical: 18,
    paddingHorizontal: 28,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginBottom: theme.spacing.lg,
    ...theme.shadows.active,
  },
  customButtonText: {
    color: toneFg('#FFFFFF'),
    fontSize: 16,
    fontWeight: '700',
  },
  footerText: {
    textAlign: 'center',
    color: theme.colors.text.body,
    fontSize: 14,
    fontWeight: '500',
  },
  footerLink: {
    color: theme.colors.brand[600],
    fontWeight: '700',
  },
}));
