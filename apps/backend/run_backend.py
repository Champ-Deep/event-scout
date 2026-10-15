import os
import subprocess
import sys

# Set environment variables
os.environ['PORT'] = '8001'
os.environ['APP_API_KEY'] = 'test_api_key_1234'

# Start the app
subprocess.run([sys.executable, 'app.py'])