import requests
import time
import sys
import os
import uuid
from datetime import datetime
import test_production

# Configuration
API_BASE = test_production.API_BASE
API_KEY = test_production.API_KEY

def register_new_user():
    timestamp = int(time.time())
    email = f"auto_test_{timestamp}@example.com"
    password = "password123"
    name = f"Auto Test User {timestamp}"
    
    print(f"[-] Registering new user: {email}")
    
    try:
        r = requests.post(
            f"{API_BASE}/register/",
            json={"name": name, "email": email, "password": password},
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            print(f"[+] Registration successful: {data}")
            return email, password
        else:
            print(f"[!] Registration failed: {r.text}")
            return None, None
    except Exception as e:
        print(f"[!] Registration error: {e}")
        return None, None

def main():
    print("============================================================")
    print("SETTING UP TEST ENVIRONMENT WITH NEW USER")
    print("============================================================")
    
    email, password = register_new_user()
    if not email:
        print("Failed to setup test user. Exiting.")
        sys.exit(1)
        
    # Inject credentials into test_production module
    test_production.TEST_EMAIL = email
    test_production.TEST_PASSWORD = password
    
    print(f"[-] Running test_production suite with {email}...")
    
    # Run the tests
    success = test_production.main()
    
    if success:
        print("\nAll Backend Tests passed with new user.")
        sys.exit(0)
    else:
        print("\nSome Backend Tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
