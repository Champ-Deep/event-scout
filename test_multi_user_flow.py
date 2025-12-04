"""
Test script to validate the multi-user authentication flow.
This demonstrates how the system works without running the actual server.

To run actual tests, you need to:
1. Install dependencies: pip install -r requirements.txt
2. Start server: python app.py
3. Run: python test_multi_user_flow.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"
API_KEY = "1234"

def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_register_user(name, email, password):
    """Test user registration"""
    print(f"\n=== Testing User Registration: {email} ===")
    payload = {
        "name": name,
        "email": email,
        "password": password
    }
    response = requests.post(
        f"{BASE_URL}/register/",
        json=payload
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    if response.status_code == 200:
        print(f"✅ User registered successfully!")
        print(f"📝 User ID: {result['user_id']}")
        return result['user_id']
    else:
        print(f"❌ Registration failed: {result}")
        return None

def test_login_user(email, password):
    """Test user login"""
    print(f"\n=== Testing User Login: {email} ===")
    payload = {
        "email": email,
        "password": password
    }
    response = requests.post(
        f"{BASE_URL}/login/",
        json=payload
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    if response.status_code == 200:
        print(f"✅ Login successful!")
        print(f"📝 User ID: {result['user_id']}")
        return result['user_id']
    else:
        print(f"❌ Login failed: {result}")
        return None

def test_add_contact(user_id, contact_data):
    """Test adding a contact"""
    print(f"\n=== Testing Add Contact for User: {user_id} ===")
    headers = {
        "x-api-key": API_KEY,
        "x-user-id": user_id
    }
    response = requests.post(
        f"{BASE_URL}/add_contact/",
        json=contact_data,
        headers=headers
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    if response.status_code == 200:
        print(f"✅ Contact added successfully!")
        print(f"📝 Contact ID: {result['contact_id']}")
        return result['contact_id']
    else:
        print(f"❌ Add contact failed: {result}")
        return None

def test_list_contacts(user_id):
    """Test listing contacts for a user"""
    print(f"\n=== Testing List Contacts for User: {user_id} ===")
    headers = {
        "x-api-key": API_KEY,
        "x-user-id": user_id
    }
    response = requests.get(
        f"{BASE_URL}/list_contacts/",
        headers=headers
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Total Contacts: {result.get('total_contacts', 0)}")
    print(f"Response: {json.dumps(result, indent=2)}")

    if response.status_code == 200:
        print(f"✅ Listed contacts successfully!")
        return result
    else:
        print(f"❌ List contacts failed: {result}")
        return None

def test_search_contacts(user_id, query):
    """Test searching contacts"""
    print(f"\n=== Testing Search Contacts: '{query}' ===")
    headers = {
        "x-api-key": API_KEY,
        "x-user-id": user_id
    }
    payload = {"query": query}
    response = requests.post(
        f"{BASE_URL}/search/",
        json=payload,
        headers=headers
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    if response.status_code == 200:
        print(f"✅ Search successful!")
        return result
    else:
        print(f"❌ Search failed: {result}")
        return None

def test_get_contact(user_id, contact_id):
    """Test getting a specific contact"""
    print(f"\n=== Testing Get Contact: {contact_id} ===")
    headers = {
        "x-api-key": API_KEY,
        "x-user-id": user_id
    }
    response = requests.get(
        f"{BASE_URL}/contact/{contact_id}",
        headers=headers
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    if response.status_code == 200:
        print(f"✅ Got contact successfully!")
        return result
    else:
        print(f"❌ Get contact failed: {result}")
        return None

def test_update_contact(user_id, contact_id, update_data):
    """Test updating a contact"""
    print(f"\n=== Testing Update Contact: {contact_id} ===")
    headers = {
        "x-api-key": API_KEY,
        "x-user-id": user_id
    }
    response = requests.put(
        f"{BASE_URL}/contact/{contact_id}",
        json=update_data,
        headers=headers
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    if response.status_code == 200:
        print(f"✅ Contact updated successfully!")
        return result
    else:
        print(f"❌ Update contact failed: {result}")
        return None

def test_delete_contact(user_id, contact_id):
    """Test deleting a contact"""
    print(f"\n=== Testing Delete Contact: {contact_id} ===")
    headers = {
        "x-api-key": API_KEY,
        "x-user-id": user_id
    }
    response = requests.delete(
        f"{BASE_URL}/contact/{contact_id}",
        headers=headers
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    if response.status_code == 200:
        print(f"✅ Contact deleted successfully!")
        return result
    else:
        print(f"❌ Delete contact failed: {result}")
        return None

def test_data_isolation():
    """Test that users can't access each other's data"""
    print("\n" + "="*60)
    print("=== Testing Multi-User Data Isolation ===")
    print("="*60)

    # Register two users
    user1_id = test_register_user("Alice Johnson", "alice@example.com", "alice123")
    user2_id = test_register_user("Bob Smith", "bob@example.com", "bob123")

    if not user1_id or not user2_id:
        print("❌ User registration failed, skipping isolation test")
        return

    # Add contacts for User 1
    print("\n--- Adding contacts for Alice ---")
    alice_contact1 = test_add_contact(user1_id, {
        "name": "Charlie Brown",
        "email": "charlie@techcorp.com",
        "phone": "555-1234",
        "linkedin": "linkedin.com/in/charlie",
        "company_name": "TechCorp"
    })

    alice_contact2 = test_add_contact(user1_id, {
        "name": "Diana Prince",
        "email": "diana@startupco.com",
        "phone": "555-5678",
        "linkedin": "linkedin.com/in/diana",
        "company_name": "StartupCo"
    })

    # Add contacts for User 2
    print("\n--- Adding contacts for Bob ---")
    bob_contact1 = test_add_contact(user2_id, {
        "name": "Eve Adams",
        "email": "eve@bigcompany.com",
        "phone": "555-9999",
        "linkedin": "linkedin.com/in/eve",
        "company_name": "BigCompany"
    })

    # List contacts for each user
    print("\n--- Listing Alice's contacts ---")
    alice_contacts = test_list_contacts(user1_id)
    alice_count = alice_contacts.get('total_contacts', 0) if alice_contacts else 0

    print("\n--- Listing Bob's contacts ---")
    bob_contacts = test_list_contacts(user2_id)
    bob_count = bob_contacts.get('total_contacts', 0) if bob_contacts else 0

    # Verify isolation
    print("\n" + "="*60)
    print("=== Data Isolation Verification ===")
    print(f"Alice has {alice_count} contacts")
    print(f"Bob has {bob_count} contacts")

    if alice_count == 2 and bob_count == 1:
        print("✅ Data isolation working correctly!")
        print("✅ Each user sees only their own contacts")
    else:
        print("❌ Data isolation may have issues")

    # Test search isolation
    print("\n--- Testing search isolation ---")
    print("Searching for 'Charlie' in Alice's database:")
    alice_search = test_search_contacts(user1_id, "Charlie")

    print("\nSearching for 'Charlie' in Bob's database:")
    bob_search = test_search_contacts(user2_id, "Charlie")

    alice_results = len(alice_search.get('results', [])) if alice_search else 0
    bob_results = len(bob_search.get('results', [])) if bob_search else 0

    if alice_results > 0 and bob_results == 0:
        print("✅ Search isolation working correctly!")
        print("✅ Bob cannot see Alice's contacts")
    else:
        print("❌ Search isolation may have issues")

    return user1_id, user2_id, alice_contact1

def main():
    """Run all tests"""
    print("="*60)
    print("Multi-User Contact Management System - Test Suite")
    print("="*60)

    try:
        # Test health check
        test_health_check()

        # Test full multi-user flow with data isolation
        user1_id, user2_id, contact_id = test_data_isolation()

        # Test additional operations on User 1
        if user1_id and contact_id:
            print("\n" + "="*60)
            print("=== Testing Additional Operations ===")
            print("="*60)

            # Test get contact
            test_get_contact(user1_id, contact_id)

            # Test update contact
            test_update_contact(user1_id, contact_id, {
                "phone": "555-0000",
                "company_name": "TechCorp Updated"
            })

            # Verify update
            test_get_contact(user1_id, contact_id)

            # Test login (re-authentication)
            print("\n--- Testing Re-authentication ---")
            test_login_user("alice@example.com", "alice123")

            # Test wrong password
            print("\n--- Testing Wrong Password ---")
            test_login_user("alice@example.com", "wrongpassword")

        print("\n" + "="*60)
        print("=== Test Suite Completed ===")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
