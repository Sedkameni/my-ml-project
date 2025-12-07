import bcrypt
import sqlite3


def hash_password(password):
    """Hash password with salt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed


def store_user(username, password):
    """Store user with hashed password"""
    hashed_password = hash_password(password)

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO users (username, password_hash) 
                     VALUES (?, ?)''', (username, hashed_password))
    conn.commit()
    conn.close()


def verify_password(username, password):
    """Verify password against stored hash"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
    stored_hash = cursor.fetchone()[0]
    conn.close()

    return bcrypt.checkpw(password.encode('utf-8'), stored_hash)


# Usage
store_user("john_doe", "mySecurePassword123")
print(verify_password("john_doe", "mySecurePassword123"))  # True