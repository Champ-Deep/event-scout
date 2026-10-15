import React from 'react';
import { Outlet } from '@tanstack/react-router';

export function AuthLayout() {
  // Session check placeholder: in production validates session token edge cookie
  return <Outlet />;
}
