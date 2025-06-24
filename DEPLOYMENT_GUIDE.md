# Fintel Deployment Guide

This guide will walk you through the steps to deploy your Fintel app on Streamlit Cloud.

## Prerequisites

Before you deploy the app, make sure you have:

1. A GitHub account
2. Git installed on your local machine
3. Prepared all necessary data (run `python prepare_data_for_deploy.py`)
4. The trained model files in the `models/` directory

## Step 1: Create a GitHub Repository

If you don't already have a GitHub repository for your project:

1. Go to [GitHub](https://github.com) and sign in
2. Click on the "+" icon in the top right corner and select "New repository"
3. Name your repository (e.g., "fintel")
4. Choose "Public" visibility (required for the free tier of Streamlit Cloud)
5. Click "Create repository"
6. Follow the instructions to push your existing repository to GitHub:

```bash
git init
git add .
git commit -m "Initial commit for deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fintel.git
git push -u origin main
```

## Step 2: Deploy on Streamlit Cloud

Now that your code is on GitHub, you can deploy it on Streamlit Cloud:

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch (main), and specify the main file path as `streamlit_app.py`
5. Click "Deploy"

Your app will be built and deployed, and you'll get a URL to access it (e.g., https://username-fintel-streamlit-app-12345.streamlit.app).

## Step 3: Configure Environment Variables (Optional)

If you want your app to be able to fetch fresh news data:

1. In Streamlit Cloud, go to your app's settings
2. In the "Secrets" section, add your News API key:
   ```
   NEWS_API_KEY = "your_api_key_here"
   ```
3. Save the settings
4. Reboot the app

## Troubleshooting

If your app fails to deploy, check the build logs for errors:

1. Missing dependencies: Make sure all packages are correctly listed in `requirements.txt`
2. File paths: Make sure all file paths are consistent
3. Data files: Confirm that all necessary data files are included in your GitHub repository

## Local Testing

You can always test your app locally before deployment:

```bash
streamlit run streamlit_app.py
```

## Production Considerations

For a production deployment, consider:

1. Setting up scheduled data updates
2. Implementing user authentication
3. Using a database instead of CSV files
4. Setting up monitoring and alerts
5. Adding more comprehensive error handling

For now, this prototype deployment should be sufficient for demonstration purposes.
