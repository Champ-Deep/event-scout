import React, { useEffect, useState } from 'react';
import { View, ActivityIndicator, UIManager } from 'react-native';
import { theme, makeStyles } from './theme';

// Conditional import for Rive to avoid runtime errors if not available
let RiveComponent: any = null;
try {
  // Dynamic import for Rive
  // Only when its native view is actually in this build (it's left out of the dev build).
  if ((UIManager as any).getViewManagerConfig?.('RiveReactNativeView')) {
    const Rive = require('rive-react-native');
    RiveComponent = Rive.default || Rive;
  }
} catch (error) {
  console.warn('Rive animation library not available:', error);
}

/**
 * Rive animation component for microkit.io 4 key product moments:
 * 1. Scan shutter
 * 2. Score reveal  
 * 3. Sync state
 * 4. Offline badge
 */
export interface RiveAnimationProps {
  /** Animation type for the 4 key product moments */
  type: 'scan' | 'score' | 'sync' | 'offline';
  /** Size of the animation */
  size?: 'sm' | 'md' | 'lg';
  /** Whether the animation is active */
  active?: boolean;
  /** Callback when animation completes */
  onComplete?: () => void;
  /** Additional styles */
  style?: any;
}

/**
 * Rive animation component for stateful icons
 * Uses one .riv file driving both mobile and web runtimes
 */
export const RiveAnimation: React.FC<RiveAnimationProps> = ({
  type,
  size = 'md',
  active = false,
  onComplete,
  style,
}) => {
  const [animationLoaded, setAnimationLoaded] = useState(false);
  const [riveError] = useState(false);

  // Animation configuration for each type
  const animationConfig = {
    scan: {
      resourceName: 'scout_icons',
      stateMachineName: 'scan_shutter',
      artboardName: 'scan',
      fallbackIcon: 'Camera', // Lucide icon name for fallback
      size: getSize(size),
    },
    score: {
      resourceName: 'scout_icons',
      stateMachineName: 'score_reveal',
      artboardName: 'score',
      fallbackIcon: 'Star',
      size: getSize(size),
    },
    sync: {
      resourceName: 'scout_icons',
      stateMachineName: 'sync_state',
      artboardName: 'sync',
      fallbackIcon: 'RefreshCw',
      size: getSize(size),
    },
    offline: {
      resourceName: 'scout_icons',
      stateMachineName: 'offline_badge',
      artboardName: 'offline',
      fallbackIcon: 'WifiOff',
      size: getSize(size),
    },
  };

  const config = animationConfig[type];

  useEffect(() => {
    // Set loaded state (in reality would wait for Rive asset load)
    setAnimationLoaded(true);
  }, [type]);

  // Handle Rive state machine triggers
  useEffect(() => {
    if (!RiveComponent || !animationLoaded) return;

    // In a real implementation, we would trigger Rive state machine
    // based on the 'active' prop and animation type
    console.log(`Rive animation ${type} ${active ? 'activated' : 'deactivated'}`);
    
    // Trigger completion callback for specific animations
    if (active && (type === 'score' || type === 'scan')) {
      const timer = setTimeout(() => {
        onComplete?.();
      }, 1500); // Animation duration
        
      return () => clearTimeout(timer);
    }
  }, [active, type, animationLoaded, onComplete]);

  // If Rive is not available, show fallback
  if (!RiveComponent || riveError) {
    return (
      <View style={[styles.fallbackContainer, { width: config.size, height: config.size }, style]}>
        <View style={styles.fallbackIcon}>
          {/* Fallback to ActivityIndicator for now */}
          <ActivityIndicator 
            size={config.size > 32 ? 'large' : 'small'}
            color={theme.colors.brand[600]}
          />
        </View>
      </View>
    );
  }

  return (
    <View style={[styles.container, { width: config.size, height: config.size }, style]}>
      {RiveComponent && animationLoaded ? (
        <RiveComponent
          resourceName={config.resourceName}
          stateMachineName={config.stateMachineName}
          artboardName={config.artboardName}
          style={{ width: config.size, height: config.size }}
          autoplay={active}
        />
      ) : (
        <ActivityIndicator 
          size={config.size > 32 ? 'large' : 'small'}
          color={theme.colors.brand[600]}
        />
      )}
    </View>
  );
};

// Helper function to get size in pixels
function getSize(size: 'sm' | 'md' | 'lg'): number {
  switch (size) {
    case 'sm': return 24;
    case 'md': return 32;
    case 'lg': return 48;
    default: return 32;
  }
}

const styles = makeStyles(() => ({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  fallbackContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.colors.bg.surface,
    borderRadius: theme.radii.md,
  },
  fallbackIcon: {
    alignItems: 'center',
    justifyContent: 'center',
  },
}));

/**
 * Pre-configured Rive animation components for the 4 key moments
 */
export const ScanShutterAnimation = (props: Omit<RiveAnimationProps, 'type'>) => (
  <RiveAnimation type="scan" {...props} />
);

export const ScoreRevealAnimation = (props: Omit<RiveAnimationProps, 'type'>) => (
  <RiveAnimation type="score" {...props} />
);

export const SyncStateAnimation = (props: Omit<RiveAnimationProps, 'type'>) => (
  <RiveAnimation type="sync" {...props} />
);

export const OfflineBadgeAnimation = (props: Omit<RiveAnimationProps, 'type'>) => (
  <RiveAnimation type="offline" {...props} />
);