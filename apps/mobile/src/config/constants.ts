import { Platform } from 'react-native';

const resolveBaseUrl = () => {
  const envUrl = process.env.EXPO_PUBLIC_API_BASE_URL || 'http://localhost:8000';
  if (Platform.OS === 'android') {
    return envUrl.replace(/localhost|127\.0\.0\.1/, '10.0.2.2');
  }
  return envUrl;
};

// API Configuration
export const API_CONFIG = {
  // Default API base URL - automatically handles Android emulator localhost rewriting
  BASE_URL: resolveBaseUrl(),
  
  // API timeout in milliseconds
  TIMEOUT: 15000,
  
  // Retry configuration
  MAX_RETRIES: 3,
  RETRY_DELAY: 1000,
  
  // Offline mode configuration
  OFFLINE_MODE_ENABLED: true,
  OFFLINE_CACHE_DURATION: 24 * 60 * 60 * 1000, // 24 hours in milliseconds
  
  // Sync configuration
  SYNC_BATCH_SIZE: 50,
  SYNC_INTERVAL: 30 * 1000, // 30 seconds
  
  // Debug mode
  DEBUG: __DEV__,
};

// Feature flags
export const FEATURE_FLAGS = {
  ENABLE_AI_ASSISTANT: true,
  ENABLE_OFFLINE_SYNC: true,
  ENABLE_QR_SCANNING: true,
  ENABLE_CAMERA_SCAN: true,
  ENABLE_EXPORT: true,
  ENABLE_NOTIFICATIONS: false,
};

// EventScout colors (matching design system)
export const COLORS = {
  primary: '#0f172a',
  secondary: '#2563eb',
  success: '#059669',
  warning: '#d97706',
  danger: '#ef4444',
  background: '#f8fafc',
  surface: '#ffffff',
  textPrimary: '#0f172a',
  textSecondary: '#64748b',
  textTertiary: '#94a3b8',
  border: '#e2e8f0',
  overlay: 'rgba(0, 0, 0, 0.5)',
  
  // Score band colors
  scoreA: { light: '#dcfce7', dark: '#16a34a' },
  scoreB: { light: '#dbeafe', dark: '#2563eb' },
  scoreC: { light: '#fef3c7', dark: '#d97706' },
  scoreD: { light: '#fee2e2', dark: '#dc2626' },
  scoreF: { light: '#f3f4f6', dark: '#6b7280' },
};

// Screen names for navigation
export const SCREENS = {
  CONTACTS: 'Contacts',
  SCAN: 'Scan',
  EVENTS: 'Events',
  ASSISTANT: 'Assistant',
  PROFILE: 'Profile',
  
  // Modal screens
  CONTACT_DETAIL: 'ContactDetail',
  SCAN_CAMERA: 'ScanCamera',
  SCAN_QR: 'ScanQR',
  ADD_CONTACT: 'AddContact',
  EVENT_DETAIL: 'EventDetail',
  SETTINGS: 'Settings',
};

// API Endpoint constants
export const API_ENDPOINTS = {
  HEALTH: '/api/health',
  CONTACTS: '/api/contacts',
  CONTACT_DETAIL: '/api/contacts/:id',
  EVENTS: '/api/events',
  EVENT_SEATS: '/api/events/:id/seats',
  SYNC_PUSH: '/api/sync/push',
  SYNC_PULL: '/api/sync/pull',
  LIVE_EVENTS: '/api/live',
  ACTIVITY: '/api/admin/activity',
  MODELS: '/api/models',
  CONVERSE: '/api/converse',
  EXHIBITORS: '/api/exhibitors',
  EXHIBITOR_CATEGORIES: '/api/exhibitors/categories',
  FILES_LIST: '/api/files/list',
  CONTACT_FILES: '/api/contact/:id/files',
  SCORE_CONTACT: '/api/contact/:id/score',
  SCORE_ALL: '/api/contacts/score_all',
  ENRICH_CONTACT: '/api/contact/:id/enrich',
  CONTACT_QR: '/api/contact/:id/qr',
  PIPELINE_RUN: '/api/contact/:id/pipeline/run',
  PIPELINE_RETRY: '/api/contact/:id/pipeline/retry',
  PIPELINE_STATUS: '/api/contact/:id/pipeline',
  
  // Legacy endpoints
  REGISTER: '/register/',
  LOGIN: '/login/',
  SEARCH: '/search/',
  EXPORT: '/export_contacts/',
  USER_PROFILE: '/user/profile/',
  DASHBOARD: '/dashboard',
  ADD_CONTACT_FROM_IMAGE: '/add_contact_from_image/',
  SCAN_QR: '/scan_qr/',
  ADD_CONTACT: '/add_contact/',
  GENERATE_QR: '/generate_qr/',
};

// Database constants
export const DATABASE = {
  NAME: 'eventscout.db',
  VERSION: 1,
  TABLES: {
    CONTACTS: 'contacts',
    EVENTS: 'events',
    OUTBOX: 'outbox',
    SYNC_META: 'sync_meta',
  },
};

// Storage keys
export const STORAGE_KEYS = {
  AUTH_TOKEN: 'auth_token',
  REFRESH_TOKEN: 'refresh_token',
  USER_ID: 'user_id',
  USER_PROFILE: 'user_profile',
  LAST_SYNC: 'last_sync',
  SYNC_STATE: 'sync_state',
  PREFERENCES: 'preferences',
  RECENT_SEARCHES: 'recent_searches',
  FAVORITES: 'favorites',
};

// Validation constants
export const VALIDATION = {
  EMAIL_REGEX: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  PHONE_REGEX: /^[\d\s\-\+\(\)\.]{10,20}$/,
  NAME_MIN_LENGTH: 2,
  NAME_MAX_LENGTH: 100,
  NOTES_MAX_LENGTH: 1000,
  PASSWORD_MIN_LENGTH: 8,
};

// Error messages
export const ERROR_MESSAGES = {
  NETWORK: 'Network error. Please check your connection.',
  TIMEOUT: 'Request timed out. Please try again.',
  UNAUTHORIZED: 'Session expired. Please log in again.',
  FORBIDDEN: 'You do not have permission to perform this action.',
  NOT_FOUND: 'The requested resource was not found.',
  SERVER_ERROR: 'Server error. Please try again later.',
  VALIDATION_ERROR: 'Please check your input and try again.',
  UNKNOWN_ERROR: 'An unexpected error occurred.',
  OFFLINE: 'You are offline. Changes will be synced when connected.',
};

// Success messages
export const SUCCESS_MESSAGES = {
  CONTACT_SAVED: 'Contact saved successfully.',
  CONTACT_UPDATED: 'Contact updated successfully.',
  CONTACT_DELETED: 'Contact deleted successfully.',
  SYNC_COMPLETE: 'Sync completed successfully.',
  EXPORT_COMPLETE: 'Export completed successfully.',
  SCORE_UPDATED: 'Score updated successfully.',
  PROFILE_UPDATED: 'Profile updated successfully.',
};