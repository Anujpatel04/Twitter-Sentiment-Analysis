# 🚀 Deployment Guide

This guide will help you deploy your Twitter Sentiment Analysis application to the cloud so it can be accessed from anywhere, anytime.

## 📋 Pre-Deployment Checklist

- [x] ✅ All code is committed to GitHub
- [x] ✅ Requirements.txt is up to date
- [x] ✅ Procfile created for web server
- [x] ✅ Application configured for production
- [ ] ⚠️ Model files need to be uploaded (see below)

## ⚠️ Important: Model Files

Your model files (`model_correct.joblib` and `vectorizer_correct.pkl`) are in the `models/` directory but are currently excluded from Git. You need to:

1. **Upload model files to your hosting platform** OR
2. **Include them in Git** (if they're not too large)

### Option A: Include Models in Git (Recommended)
If your model files are under 100MB each:
```bash
git add models/
git commit -m "Add trained model files"
git push
```

### Option B: Upload After Deployment
After deployment, upload the model files to the hosting platform's file system or use cloud storage.

---

## 🌐 Deployment Options

### Option 1: Render.com (Recommended - FREE Tier Available)

**Best for:** Quick deployment, free tier, easy setup

#### Steps:

1. **Sign up at [Render.com](https://render.com)** (free account)

2. **Create New Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `Anujpatel04/Twitter-Sentiment-Analysis`
   - Select the repository

3. **Configure Service:**
   - **Name:** `twitter-sentiment-analysis`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn MLapp:app`
   - **Plan:** Free (or choose paid for better performance)

4. **Add Environment Variables (Optional):**
   - `PYTHON_VERSION` = `3.9.18`
   - `FLASK_ENV` = `production`

5. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - Your app will be live at: `https://twitter-sentiment-analysis.onrender.com`

6. **Upload Model Files:**
   - After first deployment, use Render's Shell or upload via:
   - Go to your service → Shell
   - Or use SFTP to upload `models/` directory

**Pros:**
- ✅ Free tier available
- ✅ Automatic deployments from GitHub
- ✅ Easy to use
- ✅ Custom domain support

**Cons:**
- ⚠️ Free tier spins down after inactivity (takes 30-60 seconds to wake up)
- ⚠️ Limited resources on free tier

---

### Option 2: Railway.app (Recommended - FREE Tier Available)

**Best for:** Modern platform, good free tier, fast deployments

#### Steps:

1. **Sign up at [Railway.app](https://railway.app)** (free account)

2. **Create New Project:**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `Anujpatel04/Twitter-Sentiment-Analysis`

3. **Configure:**
   - Railway will auto-detect Python
   - **Start Command:** `gunicorn MLapp:app`
   - Add environment variable: `PORT` (Railway sets this automatically)

4. **Deploy:**
   - Railway automatically builds and deploys
   - Get your live URL: `https://your-app-name.railway.app`

5. **Upload Model Files:**
   - Use Railway CLI or upload via dashboard

**Pros:**
- ✅ Free tier with $5 credit monthly
- ✅ Fast deployments
- ✅ No cold starts
- ✅ Easy GitHub integration

**Cons:**
- ⚠️ Free tier has usage limits
- ⚠️ Credit-based pricing after free tier

---

### Option 3: Fly.io (FREE Tier Available)

**Best for:** Global edge deployment, good performance

#### Steps:

1. **Install Fly CLI:**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Sign up:**
   ```bash
   fly auth signup
   ```

3. **Create Fly App:**
   ```bash
   fly launch
   ```
   - Follow prompts
   - Select region
   - Don't deploy yet

4. **Create `fly.toml`** (I'll create this for you)

5. **Deploy:**
   ```bash
   fly deploy
   ```

**Pros:**
- ✅ Free tier available
- ✅ Global edge deployment
- ✅ Good performance

**Cons:**
- ⚠️ Requires CLI setup
- ⚠️ Slightly more complex

---

### Option 4: PythonAnywhere (FREE Tier Available)

**Best for:** Simple Python hosting, beginner-friendly

#### Steps:

1. **Sign up at [PythonAnywhere.com](https://www.pythonanywhere.com)** (free account)

2. **Upload Files:**
   - Go to Files tab
   - Upload your project files
   - Upload model files to `models/` directory

3. **Create Web App:**
   - Go to Web tab
   - Click "Add a new web app"
   - Select Flask
   - Choose Python version
   - Set source code path

4. **Configure WSGI:**
   - Edit WSGI file to point to `MLapp.py`

5. **Reload:**
   - Click "Reload" button
   - Your app is live!

**Pros:**
- ✅ Free tier available
- ✅ Easy file upload
- ✅ Simple interface

**Cons:**
- ⚠️ Limited resources on free tier
- ⚠️ Manual deployments

---

### Option 5: Heroku (Paid - $5/month minimum)

**Best for:** Reliability, features, but now requires payment

#### Steps:

1. **Install Heroku CLI:**
   ```bash
   brew install heroku/brew/heroku  # macOS
   ```

2. **Login:**
   ```bash
   heroku login
   ```

3. **Create App:**
   ```bash
   heroku create twitter-sentiment-analysis
   ```

4. **Deploy:**
   ```bash
   git push heroku main
   ```

**Pros:**
- ✅ Very reliable
- ✅ Many features
- ✅ Good documentation

**Cons:**
- ❌ No free tier (paid only)
- ⚠️ $5-7/month minimum

---

## 🎯 Quick Start: Render.com (Recommended)

I recommend **Render.com** for the easiest deployment. Here's the quickest path:

1. Go to https://render.com and sign up
2. Click "New +" → "Web Service"
3. Connect GitHub repo: `Anujpatel04/Twitter-Sentiment-Analysis`
4. Use these settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn MLapp:app`
5. Click "Create Web Service"
6. Wait for deployment (5-10 minutes)
7. Upload model files after deployment

Your app will be live! 🎉

---

## 📝 Post-Deployment Steps

1. **Test your live URL:**
   - Visit your deployed app
   - Test sentiment analysis
   - Check analytics page

2. **Upload Model Files:**
   - Use hosting platform's file upload
   - Or SSH/Shell access to upload files

3. **Set Custom Domain (Optional):**
   - Most platforms support custom domains
   - Update DNS settings

4. **Monitor:**
   - Check logs for errors
   - Monitor usage (free tiers have limits)

---

## 🔧 Troubleshooting

### Issue: "Module not found"
- **Solution:** Ensure all dependencies are in `requirements.txt`

### Issue: "Model files not found"
- **Solution:** Upload model files to `models/` directory on hosting platform

### Issue: "Application error"
- **Solution:** Check logs, ensure `gunicorn` is installed, check Procfile

### Issue: "Port already in use"
- **Solution:** Use environment variable `PORT` (handled automatically)

---

## 📞 Need Help?

- Check hosting platform's documentation
- Review application logs
- Test locally first with: `gunicorn MLapp:app`

---

## ✅ Deployment Checklist

Before deploying:
- [ ] All code committed to GitHub
- [ ] Requirements.txt updated
- [ ] Procfile created
- [ ] Model files ready to upload
- [ ] Tested locally
- [ ] Environment variables configured

After deploying:
- [ ] App is accessible
- [ ] Model files uploaded
- [ ] Tested sentiment analysis
- [ ] Analytics page works
- [ ] Custom domain configured (optional)

---

**Good luck with your deployment! 🚀**

