import sqlite3
import hashlib
import os
import json

DB_PATH = "finance_app.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create Users table for authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL
        )
    ''')
    
    # Create Profiles table for user data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_profiles (
            user_id INTEGER PRIMARY KEY,
            profile_data TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16).hex()
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
    return pwd_hash, salt

def create_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if user already exists
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cursor.fetchone() is not None:
        conn.close()
        return False, "Username already exists."
        
    pwd_hash, salt = hash_password(password)
    
    try:
        cursor.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)", 
                       (username, pwd_hash, salt))
        conn.commit()
        success = True
        msg = "User created successfully."
    except Exception as e:
        success = False
        msg = str(e)
    finally:
        conn.close()
        
    return success, msg

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, password_hash, salt FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user is None:
        return False, None
        
    user_id, stored_hash, salt = user
    pwd_hash, _ = hash_password(password, salt)
    
    if pwd_hash == stored_hash:
        return True, user_id
    else:
        return False, None

def save_user_profile(user_id, profile_dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    profile_json = json.dumps(profile_dict)
    
    # Check if a profile exists for this user to upsert
    cursor.execute("SELECT user_id FROM financial_profiles WHERE user_id = ?", (user_id,))
    existing = cursor.fetchone()
    
    if existing is None:
        cursor.execute("INSERT INTO financial_profiles (user_id, profile_data) VALUES (?, ?)", 
                       (user_id, profile_json))
    else:
        cursor.execute("UPDATE financial_profiles SET profile_data = ? WHERE user_id = ?", 
                       (profile_json, user_id))
                       
    conn.commit()
    conn.close()
    return True

def get_user_profile(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT profile_data FROM financial_profiles WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
        
    try:
        return json.loads(row[0])
    except:
        return None
