"""Small Linux process metric used by the local sustained-run evidence."""
import os
from pathlib import Path


def process_rss_mib():
    try:
        return int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")/2**20
    except (OSError,ValueError,IndexError):
        return None
