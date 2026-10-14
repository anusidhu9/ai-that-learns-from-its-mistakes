# Publishing the project

How to put MIND (or a copy of it) on GitHub, open it in Google Colab, and host the dashboard on Hugging Face Spaces. Commands are for Windows PowerShell; on macOS or Linux they are the same apart from activating the virtual environment.

## 1. Before you publish: keep secrets out

Access tokens (Hugging Face, GitHub, API keys) must never be committed. A public repository is copied and indexed quickly, and a leaked token can be used by anyone until you revoke it.

- Never save a token inside the project folder. If you have to, `.gitignore` already ignores `hf_token`, `*_token`, `*.token`, `*token*.txt` and `.env`.
- Never paste a token into a chat, an issue or a commit message.
- Check what you are about to commit with `git status` and `git diff --cached` before every commit.
- If a token does leak, **revoke it first** (https://huggingface.co/settings/tokens or https://github.com/settings/tokens), then remove it from the files.

## 2. Put the project on GitHub

### First time

1. **Create an empty repository** at https://github.com/new. Give it a name without spaces (e.g. `ai-that-learns-from-its-mistakes`) and choose Public or Private. **Do not** tick "Add a README", ".gitignore" or "license": the project already has them, and an empty repository avoids a rejected first push.

2. **Commit the project locally** (skip if it is already a git repository: `git status` will say so):

   ```powershell
   cd "C:\path\to\the\project"
   git init -b main
   git add -A
   git status                      # check: no tokens, no .venv, no runs/
   git commit -m "Initial commit"
   ```

3. **Connect and push:**

   ```powershell
   git remote add origin https://github.com/<your-username>/<repository>.git
   git push -u origin main
   ```

   The first push opens a GitHub sign-in window (Git Credential Manager, included with Git for Windows). After that, `git push` works without signing in again.

### Every change after that

```powershell
python -m pytest -q -m "not slow"   # make sure the tests pass
git add <the files you changed>
git commit -m "Describe what changed and why"
git push
```

Record notable changes in `CHANGELOG.md`.

### Make it look good on GitHub

- The `README.md` at the root is the repository's front page.
- `LICENSE` (MIT) makes reuse legal; GitHub shows it in the sidebar.
- On the repository page, click the gear icon next to **About** to add a description, the topics (e.g. `reinforcement-learning`, `q-learning`, `dqn`, `pytorch`, `gradio`), and a website link once the Space is running.

## 3. Open it in Google Colab

The badge at the top of the README opens the notebook straight from GitHub:

```
https://colab.research.google.com/github/<your-username>/<repository>/blob/main/MIND_Colab.ipynb
```

If you fork or rename the repository, update that link in `README.md` and the `REPO_URL` in the notebook's first code cell. For a **private** repository, Colab needs access: open the notebook through *File → Open notebook → GitHub* and sign in, or make the repository public.

## 4. Host the dashboard on Hugging Face Spaces

1. **Create a Space** at https://huggingface.co/new-space: pick a name, license MIT, SDK **Gradio** (Blank template), hardware **CPU basic**, visibility Public.
2. **Create an access token** with **Write** permission at https://huggingface.co/settings/tokens. Keep it to yourself.
3. **Deploy** from the project folder:

   ```powershell
   .venv\Scripts\Activate.ps1
   python space/deploy.py --space <hf-username>/<space-name>
   ```

   A sign-in window appears: enter your Hugging Face username, and paste the **token** (not your password) into the password field.

`space/deploy.py` uploads only what the app needs: the `.py` files in `src/`, `results/`, `LICENSE`, and the Space's own `README.md` (whose header sets the Gradio version and Python version), `app.py` and `requirements.txt` from the `space/` folder. The Space then builds for a few minutes, mostly installing PyTorch. Run the same command again after future changes to update it.

**Things to know about the free tier**

- A free Space sleeps after about 48 hours without visitors. The next visit wakes it up, which takes a minute or so; the address never changes.
- All visitors share the Space's CPU. Training requests wait in a queue when several people use it at once.
- If the Space shows **Paused** with "Quota exceeded for flavor cpu-basic", your account's free CPU allowance is used up or set to zero. Pause your other Spaces, upgrade your account, or contact Hugging Face support. The code itself does not need to change.

## 5. Without hosting: a temporary public link

From Colab (or any machine running the project):

```python
from src.app import launch_app
launch_app(share=True)
```

or `python -m src.app --share`. Gradio prints a `https://….gradio.live` link that works for about 72 hours while the process runs.
