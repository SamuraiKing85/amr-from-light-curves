import sys
from pathlib import Path
from importlib import import_module

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.manifest import DataManifest
from common.progress import CheckpointManager
from common.network import create_session

# Import the downloader dynamically to fetch the remote list
mmt9_downloader = import_module("1_acquire.mmt9_downloader")

def run_backfill():
    print("Fetching the current list of files from the MMT-9 server...")
    session = create_session("Manifest-Backfill")
    session.verify = False 
    
    remote_files = mmt9_downloader.get_remote_file_list(session)
    
    if not remote_files:
        print("Could not find any files. Check your connection.")
        return

    if len(remote_files) < 2:
        print("Fewer than two files found on the server. Cannot proceed.")
        return

    # Split the list: everything EXCEPT the last two files is marked as "old"
    old_files = remote_files[:-2]
    new_files = remote_files[-2:]

    print(f"\nFound {len(remote_files)} total files on the server.")
    print(f"Marking {len(old_files)} older files as 'downloaded' and 'processed'...")
    print(f"Leaving these {len(new_files)} files as NEW to be downloaded:")
    for f in new_files:
        print(f"  - {f}")

    manifest = DataManifest("mmt9_raw")
    checkpoint = CheckpointManager("mmt9_download")

    # Update DataManifest records
    manifest.record_batch_downloaded(old_files)
    manifest.record_batch_processed(old_files)

    # Update CheckpointManager records
    for f in old_files:
        checkpoint.mark_done(f)
    checkpoint.save()

    print("\nSuccess! Your manifest is now up to date.")
    print("You can now safely run the downloader to get the remaining two files.")

if __name__ == "__main__":
    run_backfill()
