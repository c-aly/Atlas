# Connecting Frontend to Vultr Backend

## Backend Setup (Already Done ✓)
Your backend is running on Vultr via systemd. Good!

## Steps to Connect Frontend

### 1. Get Your Vultr Server IP/Domain
- Note your Vultr server's IP address or domain name
- Example: `123.45.67.89` or `api.goatlas.tech`

### 2. Verify Backend Port is Open
Your backend runs on port 8001 (or whatever PORT env var is set).

**On your Vultr server, run:**
```bash
# Check if port is open
sudo ufw status
# If ufw is active, allow port 8001:
sudo ufw allow 8001
# Or if using iptables:
sudo iptables -A INPUT -p tcp --dport 8001 -j ACCEPT
```

### 3. Test Backend Accessibility
From your local machine, test if the backend is accessible:
```bash
curl http://YOUR_VULTR_IP:8001/health
```

If this works, your backend is accessible!

### 4. Update Frontend to Use Vultr Backend

#### Option A: Set Environment Variable During Build (Recommended)
1. Create a `.env.production` file in the `frontend` directory:
   ```
   VITE_API_URL=http://YOUR_VULTR_IP:8001
   ```
   Or if you have a domain:
   ```
   VITE_API_URL=https://api.goatlas.tech
   ```

2. Rebuild the frontend:
   ```bash
   cd frontend
   npm run build
   ```

3. Deploy the new `dist` folder to GitHub Pages (copy to `docs` folder)

#### Option B: Update Frontend Code (Quick Test)
Edit `frontend/src/api.js` line 7:
```javascript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://YOUR_VULTR_IP:8001'
```

Then rebuild and deploy.

### 5. Update CORS (If Needed)
Your backend already allows all origins with `"*"`, so this should work. But if you want to restrict it, update `app.py`:
```python
allow_origins=[
    "https://goatlas.tech",
    "http://YOUR_VULTR_IP:8001",  # Add your Vultr IP if needed
    # ... other origins
]
```

### 6. Verify Connection
1. Open your frontend (goatlas.tech)
2. Open browser DevTools (F12) → Console
3. Check for any connection errors
4. Try uploading an image - it should connect to your Vultr backend

## Troubleshooting

### Backend not accessible?
- Check firewall rules on Vultr server
- Verify backend is listening on `0.0.0.0:8001` (not just `127.0.0.1`)
- Check systemd logs: `sudo journalctl -u atlas.service -f`

### CORS errors?
- Backend already allows all origins with `"*"`
- If issues persist, check browser console for specific error

### Images not loading?
- Verify Supabase Storage is configured correctly
- Check that signed URLs are being generated properly
- Check backend logs for errors

## Optional: Use HTTPS
If you want to use HTTPS for your backend:
1. Set up Nginx as reverse proxy
2. Use Let's Encrypt for SSL certificate
3. Update frontend to use `https://api.goatlas.tech`

