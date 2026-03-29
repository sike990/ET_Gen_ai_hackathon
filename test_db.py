from database import init_db, create_user, authenticate_user, save_user_profile, get_user_profile

def run_tests():
    init_db()
    
    # Test User Registration
    success, msg = create_user("testuser", "password123")
    print("Registration:", msg)
    
    # Test Duplicate Registration
    success, msg = create_user("testuser", "password123")
    print("Duplicate Registration Test:", "Passed" if not success else "Failed")
    
    # Test Successful Auth
    auth_success, user_id = authenticate_user("testuser", "password123")
    print("Authentication:", "Passed" if auth_success else "Failed")
    
    # Test Failed Auth
    auth_fail, _ = authenticate_user("testuser", "wrongpassword")
    print("Failed Auth Test:", "Passed" if not auth_fail else "Failed")
    
    # Test Profile Save
    dummy_profile = {"income": 1000, "expenses": 500}
    save_user_profile(user_id, dummy_profile)
    print("Profile Saved!")
    
    # Test Profile Retrieval
    retrieved = get_user_profile(user_id)
    print("Retrieved Profile:", retrieved)
    print("Profile Test:", "Passed" if retrieved == dummy_profile else "Failed")

if __name__ == "__main__":
    run_tests()
