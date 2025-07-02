# 🚀 Deploy to Render.com NOW

## ✅ Project Structure Ready!
Your project is now properly organized:
```
paproject/
├── backend/           ← Backend files for Render
│   ├── server.py
│   ├── requirements.txt
│   ├── render.yaml
│   ├── cpt4.csv
│   ├── icd10_2025.txt
│   └── ...
├── pa-predictor-react/ ← Frontend files for Vercel
└── render.yaml        ← Root config pointing to backend/
```

## Step 1: Go to Render.com
1. Open [render.com](https://render.com) in your browser
2. Sign up/login with your GitHub account

## Step 2: Create New Web Service
1. Click **"New"** → **"Web Service"**
2. Connect your GitHub repository: `ravisuresh229/authorizationiq`

## Step 3: Configure the Service
Fill in these exact settings:

- **Name**: `pa-predictor-api`
- **Environment**: `Python 3`
- **Region**: `Oregon (US West)` (or closest to you)
- **Branch**: `main`
- **Root Directory**: `backend` ← **IMPORTANT!**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
- **Plan**: `Free`

## Step 4: Add Environment Variables (if using S3)
If your app uses AWS S3, add these in the Environment Variables section:
- `AWS_ACCESS_KEY_ID` = your AWS access key
- `AWS_SECRET_ACCESS_KEY` = your AWS secret key  
- `AWS_REGION` = `us-east-1` (or your region)

## Step 5: Deploy
Click **"Create Web Service"** and wait 2-3 minutes for deployment.

## Step 6: Get Your HTTPS URL
Once deployed, you'll get a URL like:
`https://pa-predictor-api.onrender.com`

## Step 7: Update Frontend
1. Go to your Vercel dashboard
2. Find your PA Predictor project
3. Go to Settings → Environment Variables
4. Add: `REACT_APP_API_URL` = `https://your-render-url.onrender.com`
5. Redeploy your frontend

## Step 8: Test
Visit your Vercel app - it should now work perfectly with HTTPS!

---

## Quick Test Commands
Once deployed, test these URLs:
- `https://your-app.onrender.com/health` - Should return `{"status": "healthy", "model_loaded": true}`
- `https://your-app.onrender.com/about` - Should return app info

## Troubleshooting
- **Build fails**: Make sure "Root Directory" is set to `backend`
- **App crashes**: Check Render logs for errors
- **Model not loading**: Make sure S3 credentials are set if using S3
- **File not found**: All backend files are now in the `backend/` directory 