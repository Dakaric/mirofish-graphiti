"""
MiroFish Backend entry point
"""

import os
import sys

# Fix garbled Chinese on Windows console: set UTF-8 encoding before all imports
if sys.platform == 'win32':
    # Ensure Python uses UTF-8 via env var
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    # Reconfigure stdout/stderr to UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.config import Config


def main():
    """Main entry."""
    # Validate config
    errors = Config.validate()
    if errors:
        print("Config error:")
        for err in errors:
            print(f"  - {err}")
        print("\nCheck the configuration in your .env file")
        sys.exit(1)

    # Create app
    app = create_app()

    # Read run config
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))
    debug = Config.DEBUG

    # Start server
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()

