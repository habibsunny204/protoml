# protoml/__main__.py
import os
import sys
import subprocess


def main():
    """The entry point for the ProtoML CLI."""
    # 1. Dynamically find where pip actually installed your app.py on their computer
    dir_path = os.path.dirname(os.path.realpath(__file__))
    app_path = os.path.join(dir_path, "app.py")

    # 2. Silently trigger the Streamlit server from within Python
    print("🧬 Starting ProtoML Zero-Code Dashboard on localhost...")
    sys.exit(subprocess.run(["streamlit", "run", app_path]).returncode)


if __name__ == "__main__":
    main()
