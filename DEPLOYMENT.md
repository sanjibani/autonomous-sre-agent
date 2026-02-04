# Deploying to Streamlit Community Cloud

Your Autonomous SRE Agent is ready for the cloud! Follow these steps to deploy it for free.

## Prerequisites
1.  **GitHub Account**: You must have the code pushed to a public (or private) GitHub repository.
2.  **OpenAI API Key**: You need a key starting with `sk-...`.

## Step 1: Sign Up
1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click "Sign up with GitHub".

## Step 2: Create App
1.  Click **"New app"**.
2.  Select **"Use existing repo"**.
3.  Choose your repository: `sanjibani/autonomous-sre-agent`.
4.  Branch: `main`.
5.  Main file path: `streamlit_app.py`.
6.  Click **"Deploy!"**.

## Step 3: Configure Secrets (CRITICAL)
Your app will fail to start initially because it needs the API Key.

1.  In your app dashboard, click **"Manage app"** (bottom right) or the **three dots** (top right) -> **Settings**.
2.  Go to the **"Secrets"** tab.
3.  Paste the following TOML configuration:

    ```toml
    # .streamlit/secrets.toml
    OPENAI_API_KEY = "sk-..."
    LLM_PROVIDER = "openai"
    USE_OPENAI_EMBEDDINGS = "true"
    ```

4.  Replace `sk-...` with your actual OpenAI API Key.
5.  Click **"Save"**.

## Step 4: Verify
1.  The app should auto-restart.
2.  Once loaded, click "Initialize Runbooks".
3.  Load Sample Logs.
4.  Ask the chat a question (e.g., "How to fix?").

## Troubleshooting
*   **"Sqlite3 version too old"**: If ChromaDB complains, add `pysqlite3-binary` to `requirements.txt`.
*   **"Ollama connection refused"**: Ensure you switched `LLM_PROVIDER` to "openai" in the secrets. Cloud apps cannot access your localhost Ollama.
