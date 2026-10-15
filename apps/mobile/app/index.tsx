import React from 'react';
import { Redirect } from 'expo-router';
import { useSession } from '../src/providers/SessionProvider';


export default function Index() {
  const { isAuthenticated } = useSession();
  
  if (isAuthenticated) {
    return <Redirect href="/feed" />;
  }
  
  return <Redirect href="/welcome" />;
}
