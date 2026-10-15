import { Platform } from 'react-native';

const REFRESH_TOKEN_KEY = 'eventscout_secure_refresh_token';
const AUTH_TOKEN_KEY = 'eventscout_secure_auth_token';

// Create a mock SecureStore for all environments initially
const createMockSecureStore = () => {
  const storage: Record<string, string> = {};
  
  return {
    getItemAsync: async (key: string): Promise<string | null> => {
      return storage[key] || null;
    },
    
    setItemAsync: async (key: string, value: string): Promise<void> => {
      storage[key] = value;
    },
    
    deleteItemAsync: async (key: string): Promise<void> => {
      delete storage[key];
    },
  };
};

// Start with mock, try to load real SecureStore for native
let SecureStore = createMockSecureStore();

// Try to load expo-secure-store for native platforms
if (Platform.OS !== 'web') {
  try {
    // Dynamically import expo-secure-store
    const expoSecureStore = require('expo-secure-store');
    
    // Check if the module has the expected methods
    if (expoSecureStore && typeof expoSecureStore.getItemAsync === 'function') {
      SecureStore = expoSecureStore;
      console.log('Using expo-secure-store for native platform');
    } else {
      console.warn('expo-secure-store loaded but missing expected methods');
    }
  } catch (error) {
    console.warn('Failed to load expo-secure-store, using mock:', error);
  }
}

export const secureStorage = {
  getRefreshToken: async (): Promise<string | null> => {
    try {
      return await SecureStore.getItemAsync(REFRESH_TOKEN_KEY);
    } catch (error) {
      console.warn('Failed to get refresh token:', error);
      return null;
    }
  },

  setRefreshToken: async (token: string): Promise<void> => {
    try {
      await SecureStore.setItemAsync(REFRESH_TOKEN_KEY, token);
    } catch (error) {
      console.warn('Failed to set refresh token:', error);
      // Fallback to localStorage for web
      if (Platform.OS === 'web') {
        localStorage.setItem(REFRESH_TOKEN_KEY, token);
      }
    }
  },

  getAuthToken: async (): Promise<string | null> => {
    try {
      return await SecureStore.getItemAsync(AUTH_TOKEN_KEY);
    } catch (error) {
      console.warn('Failed to get auth token:', error);
      return null;
    }
  },

  setAuthToken: async (token: string): Promise<void> => {
    try {
      await SecureStore.setItemAsync(AUTH_TOKEN_KEY, token);
    } catch (error) {
      console.warn('Failed to set auth token:', error);
      // Fallback to localStorage for web
      if (Platform.OS === 'web') {
        localStorage.setItem(AUTH_TOKEN_KEY, token);
      }
    }
  },

  clearTokens: async (): Promise<void> => {
    try {
      await Promise.all([
        SecureStore.deleteItemAsync(REFRESH_TOKEN_KEY),
        SecureStore.deleteItemAsync(AUTH_TOKEN_KEY),
      ]);
    } catch (error) {
      console.warn('Failed to clear tokens:', error);
      // Fallback for web
      if (Platform.OS === 'web') {
        localStorage.removeItem(REFRESH_TOKEN_KEY);
        localStorage.removeItem(AUTH_TOKEN_KEY);
      }
    }
  },

  // Additional helper methods
  hasTokens: async (): Promise<boolean> => {
    const token = await secureStorage.getAuthToken();
    return !!token;
  },

  // Get both tokens at once
  getAllTokens: async (): Promise<{ authToken: string | null; refreshToken: string | null }> => {
    const [authToken, refreshToken] = await Promise.all([
      secureStorage.getAuthToken(),
      secureStorage.getRefreshToken(),
    ]);
    return { authToken, refreshToken };
  },

  // Set both tokens at once
  setAllTokens: async (authToken: string, refreshToken: string): Promise<void> => {
    await Promise.all([
      secureStorage.setAuthToken(authToken),
      secureStorage.setRefreshToken(refreshToken),
    ]);
  },
};
