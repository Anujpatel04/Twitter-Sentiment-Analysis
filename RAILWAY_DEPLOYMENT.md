# 🚂 Railway Deployment Guide

Quick guide to deploy your Twitter Sentiment Analysis app on Railway.

## 🚀 Quick Start (5 Minutes)

### Step 1: Sign Up
1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project"
3. Sign in with GitHub

### Step 2: Deploy from GitHub
1. Click "Deploy from GitHub repo"
2. Select: `Anujpatel04/Twitter-Sentiment-Analysis`
3. Railway will automatically:
   - Detect Python
   - Read `requirements.txt`
   - Use `Procfile` for start command
   - Build and deploy your app

### Step 3: Wait for Deployment
- Watch build logs in real-time
- Build takes ~3-5 minutes
- Deployment happens automatically

### Step 4: Get Your URL
- Railway provides: `https://your-app-name.up.railway.app`
- Click on the service → Settings → Generate Domain
- Or use the auto-generated domain

## ✅ What Railway Auto-Detects

- ✅ **Python** from `requirements.txt`
- ✅ **Start Command** from `Procfile` (`gunicorn MLapp:app`)
- ✅ **Port** (sets `PORT` environment variable automatically)
- ✅ **Build Command** from `railway.json` or auto-detects

## 📋 Configuration Files

Your project includes:
- ✅ `Procfile` - Start command
- ✅ `requirements.txt` - Dependencies
- ✅ `railway.json` - Railway configuration (optional)
- ✅ `runtime.txt` - Python version (optional)

## 🔧 Manual Configuration (If Needed)

### Build Command
```
pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
```

### Start Command
```
gunicorn MLapp:app
```

### Environment Variables (Optional)
- `PYTHON_VERSION` = `3.11.0`
- `FLASK_ENV` = `production`

## 📊 Monitoring

- **Logs**: View real-time logs in Railway dashboard
- **Metrics**: Check CPU, Memory, Network usage
- **Deployments**: View deployment history

## 🔄 Updates

Railway automatically redeploys when you push to GitHub:
1. Push changes to `main` branch
2. Railway detects changes
3. Automatically rebuilds and redeploys
4. Zero-downtime deployment

## 💰 Pricing

- **Free Tier**: $5 credit/month
- **Hobby Plan**: $5/month (if you exceed free tier)
- **Pro Plan**: $20/month (for production)

## 🐛 Troubleshooting

### Build Fails
- Check build logs in Railway dashboard
- Ensure all dependencies are in `requirements.txt`
- Verify Python version compatibility

### App Not Starting
- Check service logs
- Verify `gunicorn` is in `requirements.txt`
- Ensure `Procfile` is correct

### Model Files Not Found
- Model files are in Git, should be included
- Check `models/` directory exists
- Verify file paths in `MLapp.py`

## 🎯 Success Checklist

- [ ] Railway account created
- [ ] GitHub repo connected
- [ ] Service created and building
- [ ] Build completed successfully
- [ ] App is accessible via Railway URL
- [ ] Sentiment analysis working
- [ ] Analytics page accessible

## 📞 Support

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Check logs in Railway dashboard for errors

---

**Your app is Railway-ready! 🚂**

