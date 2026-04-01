"""Correct file usage patterns — should all conform."""

def read_config(f):
    """Read entire config file."""
    data = f.read()
    f.close()
    return data

def read_lines(f):
    """Read file line by line."""
    f.readline()
    f.readline()
    f.close()

def write_and_close(f):
    """Write data then close."""
    f.write("data")
    f.flush()
    f.close()

def immediate_close(f):
    """Open and immediately close."""
    f.close()
