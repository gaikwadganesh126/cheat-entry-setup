---
title: Cheat Entry Ranking System
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
 
# ⚡ Cheat Entry — Ranking System (Streamlit)
 
NSE weekly-timeframe screener: 🟢 PASS · 🟡 WATCH · 🔴 FAIL.
Full project documentation: see the repository root `README.md`.
 
## Run locally (VS Code terminal)
```
cd cheat_entry_hf
pip install -r requirements.txt
streamlit run app.py          # live data
streamlit run demo_app.py     # dummy data, offline
```
Opens http://localhost:8501 (port 7860 is used only inside the Docker container).
 
## Deploy to Hugging Face (free)
1. Create an account at https://huggingface.co → **New Space**
   - SDK: **Docker** (Blank template), Hardware: **CPU basic (free)**, Visibility: **Private**
2. Upload all files from this folder (`app.py`, `screener.py`, `requirements.txt`,
   `Dockerfile`, `README.md`, `.streamlit/config.toml`) via **Files → Add file → Upload**,
   or with git:
   ```
   git clone https://huggingface.co/spaces/<your-user>/cheat-entry
   copy all files into it, then: git add . ; git commit -m "app" ; git push
   ```
   (password = an HF access token with *write* rights: Settings → Access Tokens)
3. Wait for the build (~2-3 min). The app is at
   `https://huggingface.co/spaces/<your-user>/cheat-entry`.
4. Team access: create an **Organization**, move the Space into it and invite teammates
   (a private Space is visible only to its members).
 
## Notes
- Keep the `---` header at the top of this README: Hugging Face reads it.
- Scans are cached for 4 hours and shared by all users (**🔄 Clear cache** forces fresh data).
- Free Spaces sleep after 48 h of no use; the first visit then takes ~1 minute to wake.
- To change rules: edit `CFG` in `screener.py`, then upload it again.