"""Database operations — mix of correct and buggy patterns."""

def query_users(conn):
    """Correct: execute, fetch, close."""
    conn.execute("SELECT * FROM users")
    conn.close()

def update_and_commit(conn):
    """Correct: execute, commit, close."""
    conn.execute("UPDATE users SET name='bob'")
    conn.commit()
    conn.close()

def forgot_commit(conn):
    """Bug: executes but never commits or closes."""
    conn.execute("INSERT INTO users VALUES (1, 'alice')")

def rollback_flow(conn):
    """Correct: execute, rollback, close."""
    conn.execute("DELETE FROM users")
    conn.rollback()
    conn.close()
