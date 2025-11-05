# GitHub Setup Instructions

## Step 1: Create a New Repository on GitHub

1. Go to https://github.com and log in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the details:
   - Repository name: `twitter-sentiment-analysis` (or your preferred name)
   - Description: `AI-Powered Twitter Sentiment Analysis using Logistic Regression - Capstone Project`
   - Visibility: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

## Step 2: Connect and Push

After creating the repository, GitHub will show you commands. Use these commands:

```bash
# Add your GitHub repository as remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/twitter-sentiment-analysis.git

# Push to GitHub
git push -u origin main
```

## Alternative: Using SSH (if you have SSH keys set up)

```bash
git remote add origin git@github.com:YOUR_USERNAME/twitter-sentiment-analysis.git
git push -u origin main
```

## If you need to authenticate:

GitHub may ask for authentication. Use one of these methods:

1. **Personal Access Token** (recommended):
   - Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
   - Generate a new token with `repo` permissions
   - Use this token as your password when pushing

2. **GitHub CLI**:
   ```bash
   gh auth login
   ```

3. **SSH Keys**: Set up SSH keys on GitHub for passwordless authentication

## Current Status

✅ Git repository initialized
✅ Initial commit created
✅ Ready to push to GitHub

Just run the commands above after creating your GitHub repository!

