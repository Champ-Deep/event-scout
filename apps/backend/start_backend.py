import os
import subprocess
import sys
import time

# Set environment variables
os.environ['PORT'] = '8002'
os.environ['APP_API_KEY'] = 'test_api_key_1234'

print(f"Starting backend on port 8002...")
print(f"Python executable: {sys.executable}")

# Start the app
proc = subprocess.Popen([sys.executable, 'app.py'], 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True)

# Wait a bit for startup
time.sleep(3)

# Check if process is still running
if proc.poll() is None:
    print("Backend started successfully!")
else:
    print("Backend failed to start. Output:")
    try:
        for line in proc.stdout:
            print(line, end='')
    except:
        pass