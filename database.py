import sqlite3
import os
import config

DB_PATH = os.path.join(config.BASE_DIR, "face_database.db")


def connect():
    return sqlite3.connect(DB_PATH)


def create_database():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        date_registered TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def add_user(name):
    conn = connect()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users(name) VALUES(?)",
            (name,)
        )
        conn.commit()
        print(f"{name} added successfully.")

    except sqlite3.IntegrityError:
        print("User already exists.")

    conn.close()


def show_users():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users")

    users = cursor.fetchall()

    for user in users:
        print(user)

    conn.close()


if __name__ == "__main__":
    create_database()

    while True:

        print("\n===== MENU =====")
        print("1. Add User")
        print("2. Show Users")
        print("3. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            name = input("Enter Name: ")
            add_user(name)

        elif choice == "2":
            show_users()

        elif choice == "3":
            break

        else:
            print("Invalid Choice")