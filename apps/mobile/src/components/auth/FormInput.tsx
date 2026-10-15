import React from 'react';
import { View, TextInput, TextInputProps } from 'react-native';
import Animated from 'react-native-reanimated';
import { Text } from '../../ui/Text';
import { theme, makeStyles } from '../../ui/theme';
import { LucideIcon } from 'lucide-react-native';

interface FormInputProps extends TextInputProps {
  label: string;
  icon: LucideIcon;
  animStyle?: any;
  borderAnimStyle?: any;
  onFocus?: () => void;
  onBlur?: () => void;
}

export const FormInput: React.FC<FormInputProps> = ({
  label,
  icon: Icon,
  animStyle,
  borderAnimStyle,
  onFocus,
  onBlur,
  ...textInputProps
}) => {
  return (
    <Animated.View style={[styles.inputGroup, animStyle]}>
      <Text style={styles.label}>{label}</Text>
      <Animated.View style={[styles.inputContainer, borderAnimStyle]}>
        <View style={styles.inputIconContainer}>
          <Icon size={20} color={theme.colors.text.muted} strokeWidth={2} />
        </View>
        <TextInput
          style={styles.input}
          placeholderTextColor={theme.colors.text.muted}
          onFocus={onFocus}
          onBlur={onBlur}
          {...textInputProps}
        />
      </Animated.View>
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
  inputContainer: {
    backgroundColor: theme.colors.bg.surface,
    borderRadius: theme.radii.md,
    flexDirection: 'row',
    alignItems: 'center',
  },
  inputIconContainer: {
    paddingLeft: 16,
    paddingRight: 8,
    justifyContent: 'center',
  },
  input: {
    flex: 1,
    paddingRight: 16,
    paddingVertical: 14,
    fontSize: theme.typography.sizes.body.fontSize,
    color: theme.colors.text.display,
  },
}));
