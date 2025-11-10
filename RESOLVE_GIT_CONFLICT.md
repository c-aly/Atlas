# Resolving Git Merge Conflict on Server

## Problem
You have local changes to `main.py` on the server that conflict with the remote changes.

## Solution Options

### Option 1: Stash Local Changes (Recommended if you want to keep them)
```bash
# Save your local changes temporarily
git stash

# Pull the latest changes
git pull

# If you want to reapply your local changes later:
git stash pop
```

### Option 2: Discard Local Changes (If you don't need them)
```bash
# Discard local changes to main.py
git checkout -- main.py

# Or discard all local changes
git reset --hard HEAD

# Then pull
git pull
```

### Option 3: Commit Local Changes First
```bash
# See what changed
git diff main.py

# If you want to keep the changes, commit them
git add main.py
git commit -m "Local changes to main.py"

# Then pull (might need to merge)
git pull
```

## Recommended: Check What Changed First

```bash
# See what your local changes are
git diff main.py

# See what the remote changes are
git diff HEAD origin/main main.py
```

## Quick Fix (If you don't need local changes)

```bash
# Discard local changes and pull
git reset --hard HEAD
git pull

# Restart the service
sudo systemctl restart atlas.service
```

## After Resolving

1. **Pull the latest code:**
   ```bash
   git pull
   ```

2. **Restart the backend:**
   ```bash
   sudo systemctl restart atlas.service
   ```

3. **Test:**
   ```bash
   curl http://localhost:8001/health
   ```

