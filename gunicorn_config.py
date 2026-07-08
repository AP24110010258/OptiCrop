"""
Gunicorn configuration for OptiCrop deployment on Render.
"""

import os

# Bind to the port Render provides via the PORT env variable
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# Workers — Render free tier has limited memory, so keep it lean
workers = 2
threads = 2

# Timeout — allow enough time for model loading on cold starts
timeout = 120

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
