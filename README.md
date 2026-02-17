# 🎬 Original Meme Factory

> AI-powered meme creation pipeline that turns raw Reddit clips into original, copyright-free memes for Instagram.

## 🧠 How It Works

```
Reddit Clips → AI Analysis → Original Meme Text → Professional Rendering → Instagram
```

### Pipeline Stages

| # | Stage | What It Does | Where |
|---|-------|-------------|-------|
| 1 | **Scrape** | Fetches raw clips from 14+ subreddits | n8n (WF1) |
| 2 | **Download** | yt-dlp with cookies, best quality | GitHub Actions |
| 3 | **Clean Audio** | FFmpeg highpass/lowpass/dynaudnorm | GitHub Actions |
| 4 | **Whisper** | `small` model transcription (high accuracy) | GitHub Actions |
| 5 | **Frame Extract** | Scene-change detection, max 5 frames | GitHub Actions |
| 6 | **BLIP Vision** | `blip-image-captioning-large` descriptions | GitHub Actions |
| 7 | **Context Build** | Merge audio + vision + emotion + scene type | GitHub Actions |
| 8 | **AI Generate** | Groq `llama-3.3-70b` → Hinglish meme text | GitHub Actions |
| 9 | **Validate** | Confidence ≥ 6, has humans, clear context | GitHub Actions |
| 10 | **FFmpeg Render** | Text overlay + SFX + BGM + anti-detection | GitHub Actions |
| 11 | **Upload** | Catbox temp hosting (72h) | GitHub Actions |
| 12 | **Publish** | Instagram Reels via Meta Graph API | n8n (WF3) |

## 📁 Project Structure

```
11-OriginalMemeFactory/
├── .github/workflows/
│   └── meme-processor.yml       # GitHub Actions pipeline
├── scripts/
│   ├── process_video.py          # Main processing script (11 stages)
│   └── requirements.txt         # Python dependencies
├── workflows/
│   ├── 01-Content-Researcher.json  # n8n: scrapes Reddit
│   ├── 02-Meme-Creator.json       # n8n: orchestrator + callback
│   └── 03-Meme-Publisher.json     # n8n: Instagram publisher
├── assets/
│   ├── sfx/                      # Sound effects (add your .mp3 files)
│   └── bgm/                      # Background music (add your .mp3 files)
└── README.md
```

## ⚡ Setup

### 1. Create GitHub Repository

```bash
# Create repo on GitHub named "Original-Meme-Factory"
# Then push this project:
cd 11-OriginalMemeFactory
git init
git add .
git commit -m "Initial: Original Meme Factory v2"
git remote add origin https://github.com/VikashMeena777/Original-Meme-Factory.git
git push -u origin main
```

### 2. GitHub Secrets

Go to **Settings → Secrets → Actions** and add:

| Secret | Description |
|--------|-------------|
| `GROQ_API_KEY` | API key from [console.groq.com](https://console.groq.com) |
| `N8N_WEBHOOK_URL` | Webhook URL from Meme Creator workflow |
| `REDDIT_COOKIES` | Reddit cookies for yt-dlp (optional) |

### 3. Google Sheets Setup

Add these columns to the **Content Pool** sheet:

```
ID | Title | URL | Video URL | Source | Subreddit | Upvotes | Flair | Download Status | Download URL | Queued At | Processed At
```

Create a **Publishing Queue** sheet tab with:

```
Video ID | Title | Download URL | Caption | Status | IG Post ID | Published At
```

### 4. Sound Effects & Background Music

Add royalty-free audio files to the `assets/` folder:

**SFX:** (name them exactly as shown)
- `assets/sfx/vine_boom.mp3`
- `assets/sfx/laugh.mp3`
- `assets/sfx/bruh.mp3`
- `assets/sfx/suspense.mp3`

**BGM:** (any name, one is randomly picked per meme)
- `assets/bgm/chill_beat_1.mp3`
- `assets/bgm/trap_loop.mp3`

> 🎵 Free SFX: [freesound.org](https://freesound.org) | Free BGM: [pixabay.com/music](https://pixabay.com/music/)

### 5. Import n8n Workflows

Import all 3 workflow JSON files from the `workflows/` folder into n8n:
1. **Content Researcher** → Runs every 6 hours
2. **Meme Creator** → Runs every 15 minutes + callback webhook
3. **Meme Publisher** → Runs every 15 min (4am-11pm)

> ⚠️ Update the **GitHub repo URL** in the Meme Creator workflow's "Trigger GitHub Actions" node.

### 6. Update Credentials

In n8n, update these credential references:
- **Google Sheets** → Your OAuth2 credentials
- **Telegram Bot** → Your bot token
- **Meta API** → Your Facebook/Instagram access token
- **GitHub** → Your personal access token

## 🤖 AI Models Used

| Model | Size | Purpose |
|-------|------|---------|
| **Whisper small** | 461MB | Audio transcription (multi-language) |
| **BLIP-large** | ~1.8GB | Frame captioning (scene descriptions) |
| **Llama 3.3 70B** | API | Meme text generation (Groq) |

Models are **cached** on GitHub Actions — first run downloads them (~3 min), subsequent runs use cache.

## 🛡️ Anti-Copyright Features

1. **Original text** — AI generates new meme text, not copied
2. **Visual shifts** — Random brightness/contrast/saturation adjustments
3. **Slight crop** — 1-3% edge crop to change fingerprint
4. **New audio layer** — SFX + BGM added, changes audio fingerprint
5. **Text overlay** — Visual modification via drawtext
6. **Watermark** — Custom branding overlay

## 🎯 Target Meme Styles

| Style | Description | Example |
|-------|-------------|---------|
| **POV** | First-person relatable text at top | "POV: Jab exam ke din pata chale..." |
| **Top/Bottom** | Classic meme format | Top: setup / Bottom: punchline |
| **Caption** | Semi-transparent bar at top | Caption bar with Hinglish text |
| **Subtitle** | Clean dialogue at bottom | Whisper-based, AI-rewritten |

## 📊 Validation Rules

A meme is **rejected** if:
- Confidence score < 6
- No `top_text` generated
- Both audio AND vision are unclear
- Text is too long (> 100 chars)

Rejected memes are marked as "skipped" in the sheet with the reason.

## 🔧 Manual Testing

```bash
# Test the processing pipeline locally
export VIDEO_URL="https://v.redd.it/example"
export VIDEO_ID="test_001"
export REDDIT_TITLE="When you realize exam is tomorrow"
export REDDIT_SUB="IndianTeenagers"
export GROQ_API_KEY="your-key-here"

python scripts/process_video.py
```

## 📋 GitHub Actions Usage

With 6 hours of free Actions per month:
- Each video takes ~5-15 min (depending on length)
- That's approximately **24-72 memes per month**
- Batch mode processes up to 3 videos in one run

## 🏗️ Built With

- **n8n** — Workflow orchestration
- **GitHub Actions** — Cloud processing
- **FFmpeg** — Video manipulation
- **OpenAI Whisper** — Speech-to-text
- **Salesforce BLIP** — Image captioning
- **Groq API** — LLM meme generation
- **Catbox** — Temporary file hosting
- **Meta Graph API** — Instagram publishing
