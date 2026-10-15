/**
 * JWT token utilities for extracting user information
 */

/**
 * Decode a JWT token without verification (for client-side use only)
 * Returns null if token is invalid or cannot be decoded
 */
export function decodeJWT(token: string): { userId?: string; sub?: string; email?: string; is_admin?: boolean; exp?: number; iat?: number } | null {
  try {
    // JWT format: header.payload.signature
    const parts = token.split('.');
    if (parts.length !== 3) {
      return null;
    }

    const payload = parts[1];
    if (!payload) {
      return null;
    }
    // Add padding if needed
    const base64 = payload.replace(/-/g, '+').replace(/_/g, '/');
    const decoded = atob(base64);
    
    return JSON.parse(decoded);
  } catch (error) {
    console.warn('Failed to decode JWT token:', error);
    return null;
  }
}

/**
 * Extract user ID from JWT token
 * Supports various token formats:
 * - Standard JWT with 'sub' or 'user_id' claim
 * - Legacy token formats: 'user_<uuid>' or 'legacy-token-<uuid>'
 */
export function extractUserIdFromToken(token: string | null): string | null {
  if (!token) return null;
  
  // Try to decode as JWT first
  const decoded = decodeJWT(token);
  if (decoded) {
    // Check for standard JWT claims
    if (decoded.userId) return decoded.userId;
    if (decoded.sub) return decoded.sub;
  }
  
  // Fallback to legacy token formats
  if (token.startsWith('user_')) {
    return token.replace('user_', '');
  }
  
  if (token.startsWith('legacy-token-')) {
    return token.replace('legacy-token-', '');
  }
  
  // If token doesn't match any known format, return null
  return null;
}

/**
 * Check if token is expired (client-side check, not secure)
 */
export function isTokenExpired(token: string): boolean {
  const decoded = decodeJWT(token);
  if (!decoded || !decoded.exp) return false;
  
  const now = Math.floor(Date.now() / 1000);
  return decoded.exp < now;
}

/**
 * Get user info from token
 */
export function getUserInfoFromToken(token: string | null): { userId?: string; email?: string } | null {
  if (!token) return null;

  const decoded = decodeJWT(token);
  if (!decoded) return null;

  const info: { userId?: string; email?: string } = {};
  const userId = decoded.userId || decoded.sub;
  if (userId) info.userId = userId;
  if (decoded.email) info.email = decoded.email;
  return info;
}