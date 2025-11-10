# Restart Backend Service on Vultr Server

## Problem
The `/health` endpoint is returning "Authorization header required" even though the code shows it should be public. This means the backend is running old code.

## Solution: Restart the Backend Service

### SSH into your Vultr server:
```bash
ssh your_user@108.61.84.48
```

### Restart the backend service:
```bash
# Restart the service
sudo systemctl restart atlas.service

# Check status
sudo systemctl status atlas.service

# View logs to verify it started correctly
sudo journalctl -u atlas.service -n 50
```

### Test the endpoint:
```bash
# Test public health endpoint (should work without auth)
curl http://localhost:8001/health

# Should return: {"status": "ok", "message": "Backend is running"}
```

### If you updated the code, make sure it's on the server:
```bash
# Navigate to your app directory
cd /path/to/your/app

# Pull latest code (if using git)
git pull

# Or copy the updated app.py file to the server
```

### Verify the code on the server:
```bash
# Check the /health endpoint definition
grep -A 5 '@app.get("/health")' app.py

# Should show:
# @app.get("/health")
# async def health_check():
#     """Public health check endpoint (no auth required)"""
#     return {
#         "status": "ok",
#         "message": "Backend is running"
#     }
```

## After Restarting

### Test from your local machine:
```bash
# Test public endpoint (should work)
curl http://108.61.84.48/health

# Should return: {"status": "ok", "message": "Backend is running"}

# Test user endpoint (requires auth - should fail without token)
curl http://108.61.84.48/health/user

# Should return: {"detail":"Authorization header required"}
```

## If It Still Doesn't Work

### Check if the service is using the correct code:
```bash
# Check where the service is running from
sudo systemctl status atlas.service

# Check the service file
sudo cat /etc/systemd/system/atlas.service

# Verify the working directory and Python path
```

### Check if there are multiple Python processes:
```bash
# Check what's running on port 8001
sudo lsof -i :8001
# Or
sudo netstat -tlnp | grep 8001
```

### Check Nginx configuration:
```bash
# Check Nginx is proxying correctly
sudo nginx -t

# View Nginx config
sudo cat /etc/nginx/sites-available/api.goatlas.tech
# Or if using default
sudo cat /etc/nginx/sites-available/default
```

## Quick Fix

If you just need to restart quickly:

```bash
# SSH into server
ssh your_user@108.61.84.48

# Restart service
sudo systemctl restart atlas.service

# Test
curl http://localhost:8001/health
```

If that returns the correct response, then test from your local machine:
```bash
curl http://108.61.84.48/health
```

