# PA Predictor Backend Deployment Guide

## Quick Fix: Deploy to Render.com (Recommended)

### Step 1: Prepare Your Repository
1. Make sure your backend code is in a GitHub repository
2. Ensure you have these files in your repo:
   - `server.py` (your FastAPI app)
   - `requirements.txt` (dependencies)
   - `render.yaml` (deployment config)
   - All data files (cpt4.csv, icd10_2025.txt, etc.)

### Step 2: Deploy to Render.com
1. Go to [render.com](https://render.com) and sign up/login
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `pa-predictor-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free (or paid if you need more resources)

### Step 3: Set Environment Variables (if needed)
In Render dashboard, add these environment variables:
- `AWS_ACCESS_KEY_ID` (if using S3)
- `AWS_SECRET_ACCESS_KEY` (if using S3)
- `AWS_REGION` (if using S3)

### Step 4: Deploy
Click "Create Web Service" and wait for deployment (usually 2-3 minutes).

### Step 5: Update Frontend
Once deployed, you'll get a URL like `https://your-app-name.onrender.com`

Update your frontend:
1. In Vercel dashboard, go to your project settings
2. Add environment variable: `REACT_APP_API_URL=https://your-app-name.onrender.com`
3. Redeploy your frontend

## Alternative: Quick ngrok Fix (Temporary)

If you want to keep your EC2 backend temporarily:

```bash
# On your EC2 instance
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Run ngrok (you'll need to sign up at ngrok.com for a free account)
ngrok http 8001
```

Then update your Vercel environment variable with the ngrok HTTPS URL.

## Why Render.com is Better Than EC2

✅ **Automatic HTTPS** - No SSL certificate setup needed  
✅ **Free Tier** - No server costs  
✅ **Auto-deployment** - Push to GitHub, auto-deploys  
✅ **Better reliability** - Managed infrastructure  
✅ **Easy scaling** - Upgrade plans as needed  
✅ **No server maintenance** - No SSH, updates, security patches  

## Troubleshooting

### Common Issues:
1. **Build fails**: Check `requirements.txt` has all dependencies
2. **App crashes**: Check logs in Render dashboard
3. **CORS errors**: The server.py already has CORS configured
4. **Environment variables**: Make sure AWS credentials are set if using S3

### Testing Your Deployment:
1. Visit `https://your-app-name.onrender.com/health`
2. Should return: `{"status": "healthy", "model_loaded": true}`
3. Test prediction endpoint: `https://your-app-name.onrender.com/predict`

## Next Steps

After successful deployment:
1. Update your frontend environment variable
2. Test the full application
3. Consider setting up a custom domain
4. Monitor usage and upgrade if needed 