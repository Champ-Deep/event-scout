#!/usr/bin/env python3
"""
Test script to verify authentication endpoints
"""

import requests
import json
import sys

# API configuration
BASE_URL = "http://localhost:8000"
TEST_EMAIL = "test@example.com"
TEST_PASSWORD = "testpassword123"
TEST_NAME = "Test User"

def test_register():
    """Test user registration endpoint"""
    print("Testing registration endpoint...")
    
    url = f"{BASE_URL}/register/"
    data = {
        "name": TEST_NAME,
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Registration successful!")
            print(f"   User ID: {result.get('user_id')}")
            print(f"   Name: {result.get('name')}")
            print(f"   Email: {result.get('email')}")
            return result.get('user_id')
        else:
            print(f"❌ Registration failed")
            return None
            
    except Exception as e:
        print(f"❌ Error during registration: {e}")
        return None

def test_login():
    """Test user login endpoint"""
    print("\nTesting login endpoint...")
    
    url = f"{BASE_URL}/login/"
    data = {
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Login successful!")
            print(f"   User ID: {result.get('user_id')}")
            print(f"   Name: {result.get('name')}")
            print(f"   Email: {result.get('email')}")
            print(f"   Is Admin: {result.get('is_admin')}")
            return result.get('user_id')
        else:
            print(f"❌ Login failed")
            return None
            
    except Exception as e:
        print(f"❌ Error during login: {e}")
        return None

def test_linkedin_auth_url():
    """Test LinkedIn OAuth URL endpoint"""
    print("\nTesting LinkedIn OAuth URL endpoint...")
    
    url = f"{BASE_URL}/auth/linkedin/url"
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('url'):
                print(f"✅ LinkedIn OAuth URL generated!")
                print(f"   URL: {result.get('url')[:50]}...")
            else:
                print(f"⚠️  LinkedIn OAuth not configured")
        else:
            print(f"❌ LinkedIn OAuth endpoint failed")
            
    except Exception as e:
        print(f"❌ Error during LinkedIn OAuth test: {e}")

def test_protected_endpoint(user_id):
    """Test a protected endpoint with API key"""
    print("\nTesting protected endpoint...")
    
    url = f"{BASE_URL}/dashboard"
    params = {
        "user_id": user_id
    }
    headers = {
        "X-API-Key": "test_api_key_1234"  # From .env file
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ Protected endpoint accessible!")
        else:
            print(f"❌ Protected endpoint failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during protected endpoint test: {e}")

def main():
    print("=" * divided_line_width)
    print("EventScout Authentication Test")
    print("=" * divided_line_width)
    
    # Test 1: Registration
    user_id = test_register()
    
    # Test 2: Login (even if registration failed, try with test credentials)
    if not user_id:
        print("\nTrying login with existing test credentials...")
        # Try with default test credentials
        test_login()
    else:
        test_login()
    
    # Test 3: LinkedIn OAuth
    test_linkedin_auth_url()
    
    # Test 4: Protected endpoint
    if user_id:
        test_protected_endpoint(user_id)
    
    print("\n" + "=" * divided_line_width)
    print("Test Complete")
    print("=" * divided_line_width)

if __name__ == "__main__":
    # Set a reasonable line width for display
    divided_line_width = 50
    main()