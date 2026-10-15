import React, { useState } from 'react';
import { View, TextInput, Pressable, TextInputProps } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withSequence
} from 'react-native-reanimated';
import { Text } from '../../ui/Text';
import { theme, makeStyles } from '../../ui/theme';
import { Lock, Eye, EyeOff } from 'lucide-react-native';

interface PasswordInputProps extends Omit<TextInputProps, 'secureTextEntry'> {
  label: string;
  animStyle?: any;
  borderAnimStyle?: any;
  onFocus?: () => void;
  onBlur?: () => void;
  showStrengthIndicator?: boolean;
  passwordLength?: number;
  minLength?: number;
}

export const PasswordInput: React.FC<PasswordInputProps> = ({
  label,
  animStyle,
  borderAnimStyle,
  onFocus,
  onBlur,
  showStrengthIndicator = false,
  passwordLength = 0,
  minLength = 8,
  ...textInputProps
}) => {
  const [showPassword, setShowPassword] = useState(false);
  const eyeIconScale = useSharedValue(1);

  const handleTogglePassword = () => {
    eyeIconScale.value = withSequence(
      withSpring(0.8, { damping: 22, stiffness: 400, overshootClamping: true }),
      withSpring(1, { damping: 22, stiffness: 200, overshootClamping: true })
    );
    setShowPassword(!showPassword);
  };

  const eyeIconAnimStyle = useAnimatedStyle(() => ({
    transform: [{ scale: eyeIconScale.value }],
  }));

  return (
    <Animated.View style={[styles.inputGroup, animStyle]}>
      <Text style={styles.label}>{label}</Text>
      <Animated.View style={[styles.passwordContainer, borderAnimStyle]}>
        <View style={styles.inputIconContainer}>
          <Lock size={20} color={theme.colors.text.muted} strokeWidth={2} />
        </View>
        <TextInput
          style={styles.passwordInput}
          placeholderTextColor={theme.colors.text.muted}
          secureTextEntry={!showPassword}
          onFocus={onFocus}
          onBlur={onBlur}
          {...textInputProps}
        />
        <Animated.View style={eyeIconAnimStyle}>
          <Pressable onPress={handleTogglePassword} style={styles.eyeButton}>
            {showPassword ? (
              <EyeOff size={20} color={theme.colors.text.muted} strokeWidth={2} />
            ) : (
              <Eye size={20} color={theme.colors.text.muted} strokeWidth={2} />
            )}
          </Pressable>
        </Animated.View>
      </Animated.View>
      {showStrengthIndicator && passwordLength > 0 && (
        <View style={styles.passwordHintContainer}>
          <Text
            style={[
              styles.passwordHintText,
              passwordLength >= minLength ? styles.passwordHintValid : styles.passwordHintInvalid,
            ]}
          >
            {passwordLength >= minLength
              ? `✓ Strong password (${passwordLength} characters)`
              : `${passwordLength}/${minLength} characters minimum`}
          </Text>
        </View>
      )}
    </Animated.View>
  );
};

const styles = makeStyles(() => ({
  inputGroup: {
    marginBottom: 20,
  },
  label: {
    fontSize: 11,
    fontWeight: '700',
    color: theme.colors.text.display,
    letterSpacing: 1,
    marginBottom: 8,
  },
  passwordContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.bg.surface,
    borderRadius: theme.radii.md,
  },
  inputIconContainer: {
    paddingLeft: 16,
    paddingRight: 8,
    justifyContent: 'center',
  },
  passwordInput: {
    flex: 1,
    paddingRight: 16,
    paddingVertical: 14,
    fontSize: theme.typography.sizes.body.fontSize,
    color: theme.colors.text.display,
  },
  eyeButton: {
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  passwordHintContainer: {
    marginTop: 8,
    paddingHorizontal: 4,
  },
  passwordHintText: {
    fontSize: 12,
    fontWeight: '500',
  },
  passwordHintValid: {
    color: theme.colors.semantic.success,
  },
  passwordHintInvalid: {
    color: theme.colors.text.muted,
  },
}));
