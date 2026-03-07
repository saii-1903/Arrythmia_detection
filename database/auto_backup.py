import os
import subprocess
from datetime import datetime

# ==========================================
# POSTGRESQL BACKUP CONFIGURATION
# ==========================================
DB_NAME = "ecg_analysis"
DB_USER = "ecg_user"
DB_PASS = "sais"
DB_HOST = "127.0.0.1"

# Directory to store the backups
BACKUP_DIR = os.path.join(os.path.dirname(__file__), "backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

def run_backup():
    """Creates a compressed, lossless snapshot of the entire database."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"ecg_backup_{timestamp}.dump")

    # Pass the password to the environment so the script doesn't pause to ask for it
    os.environ["PGPASSWORD"] = DB_PASS

    # The pg_dump command using Custom format (-F c)
    # This format is highly compressed and explicitly designed for pg_restore
    cmd = [
        "pg_dump",
        "-U", DB_USER,
        "-h", DB_HOST,
        "-F", "c", 
        "-f", backup_file,
        DB_NAME
    ]

    print(f"Starting backup of '{DB_NAME}'...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ SUCCESS: Full database backed up to -> {backup_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Backup failed: {e}")
    finally:
        # Clear the password from environment variables for security
        os.environ.pop("PGPASSWORD", None)

if __name__ == "__main__":
    run_backup()