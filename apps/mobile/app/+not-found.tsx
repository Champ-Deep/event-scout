import React from 'react';
import { View } from 'react-native';
import { Link, Stack } from 'expo-router';
import { Text } from '../src/ui/Text';
import { tokens } from '../src/ui/tokens';

import { makeStyles } from '../src/ui/theme';
export default function NotFoundScreen() {
  return (
    <>
      <Stack.Screen options={{ title: 'Oops!' }} />
      <View style={styles.container}>
        <Text variant="title">Route Not Found</Text>
        <Link href="/" style={styles.link}>
          <Text variant="body" color="emerald">Go to Home</Text>
        </Link>
      </View>
    </>
  );
}

const styles = makeStyles(() => ({
  container: {
    flex: 1,
    backgroundColor: tokens.colors.bg.primary,
    alignItems: 'center',
    justifyContent: 'center',
    padding: tokens.spacing.lg,
  },
  link: {
    marginTop: tokens.spacing.md,
  },
}));
