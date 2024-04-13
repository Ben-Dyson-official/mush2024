import os
os.system(f"python3 -m flask --app {os.path.join(os.path.dirname(__file__), 'flaskr')} --debug run")
