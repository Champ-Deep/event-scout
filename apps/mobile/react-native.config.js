// Native modules left out of the dev build.
// rive-react-native: the app ships no .riv animation file, so Rive never had anything to play;
// src/ui/RiveAnimation.tsx shows its fallback without it (as it always did in Expo Go).
module.exports = {
  dependencies: {
    'rive-react-native': { platforms: { android: null, ios: null } },
  },
};
