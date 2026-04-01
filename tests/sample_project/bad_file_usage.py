"""Buggy file usage patterns — should have violations."""

def forgot_to_close(f):
    """Bug: reads but never closes — resource leak."""
    data = f.read()
    return data

def double_close(f):
    """Bug: closes twice."""
    f.close()
    f.close()

def write_after_close(f):
    """Bug: writes after closing."""
    f.close()
    f.write("oops")
