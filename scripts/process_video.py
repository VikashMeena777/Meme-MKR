#!/usr/bin/env python3
"""
Original Meme Factory — Video Processing Pipeline
==================================================
Stages:
  1. Download raw clip (yt-dlp)
  2. Validate duration (5-60s)
  3. Clean audio (FFmpeg highpass/lowpass/dynaudnorm)
  4. Whisper transcription (small model, high accuracy)
  5. Frame extraction (FFmpeg, max 5 key frames)
  6. BLIP-large vision captioning (per frame)
  7. Emotion detection + Scene classification
  8. Context builder (merge audio + vision + metadata)
  9. AI meme text generation (Groq llama-3.3-70b)
  10. Validation gate (confidence >= 6)
  11. FFmpeg rendering (text overlay + SFX + BGM + anti-detection)
  12. Upload to Catbox
  13. Callback to n8n webhook

Designed to run on GitHub Actions (Ubuntu, CPU only, 7GB RAM).
"""

import os
import sys
import json
import subprocess
import requests
import time
import glob
import random
import traceback
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL")

WHISPER_MODEL = "small"         # High accuracy, runs fine on CPU within 6h
BLIP_MODEL = "Salesforce/blip-image-captioning-large"  # Best quality captions
GROQ_MODEL = "llama-3.3-70b-versatile"

MAX_FRAMES = 5                  # Limit frames for memory
MIN_CONFIDENCE = 2              # Reject memes below this score
MAX_VIDEO_DURATION = 60         # Max output duration in seconds
MIN_VIDEO_DURATION = 5          # Too short = useless
WATERMARK_TEXT = "@Meme_Facteory"

WORK_DIR = "work"
OUTPUT_DIR = "output"

# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    """Main pipeline orchestrator."""
    video_url = os.environ.get("VIDEO_URL")
    video_id = os.environ.get("VIDEO_ID")
    reddit_title = os.environ.get("REDDIT_TITLE", "")
    reddit_sub = os.environ.get("REDDIT_SUB", "")

    if not video_url or not video_id:
        print("ERROR: VIDEO_URL and VIDEO_ID are required", file=sys.stderr)
        sys.exit(1)

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output = {
        "video_id": video_id,
        "title": reddit_title,
        "status": "processing",
        "stages_completed": [],
        "stages_failed": [],
    }

    try:
        # ── Stage 1: Download ─────────────────────────────────────────────
        log_stage("DOWNLOAD")
        raw_video = download_video(video_url, video_id)
        output["stages_completed"].append("download")

        # ── Stage 2: Validate Duration ────────────────────────────────────
        log_stage("VALIDATE DURATION")
        duration = get_video_duration(raw_video)
        print(f"  Video duration: {duration:.1f}s")

        if duration < MIN_VIDEO_DURATION:
            raise ValueError(f"Video too short ({duration:.1f}s < {MIN_VIDEO_DURATION}s)")

        if duration > MAX_VIDEO_DURATION + 10:
            print(f"  Trimming from {duration:.1f}s to {MAX_VIDEO_DURATION}s")
            raw_video = trim_video(raw_video, MAX_VIDEO_DURATION)

        output["stages_completed"].append("validate_duration")

        # ── Stage 3: Clean Audio ──────────────────────────────────────────
        log_stage("CLEAN AUDIO")
        cleaned_audio = clean_audio(raw_video)
        output["stages_completed"].append("audio_clean")

        # ── Stage 4: Whisper Transcription ────────────────────────────────
        log_stage("WHISPER TRANSCRIPTION")
        transcript_data = transcribe_audio(cleaned_audio)
        print(f"  Language: {transcript_data['language']}")
        print(f"  Confidence: {transcript_data['confidence']}")
        print(f"  Text: {transcript_data['text'][:200]}")
        output["stages_completed"].append("whisper")

        # ── Stage 5: Frame Extraction ─────────────────────────────────────
        log_stage("FRAME EXTRACTION")
        frames = extract_frames(raw_video)
        print(f"  Extracted {len(frames)} frames")
        output["stages_completed"].append("frame_extract")

        # ── Stage 6: BLIP Vision Captioning ───────────────────────────────
        log_stage("BLIP VISION CAPTIONING")
        vision_descriptions = caption_frames(frames)
        for i, desc in enumerate(vision_descriptions):
            print(f"  Frame {i+1}: {desc}")
        output["stages_completed"].append("blip_vision")

        # ── Stage 7: Context Builder ──────────────────────────────────────
        log_stage("CONTEXT BUILDER")
        context = build_context(
            reddit_title, reddit_sub,
            transcript_data, vision_descriptions
        )
        print(f"  Emotion: {context['detected_emotion']}")
        print(f"  Scene: {context['scene_type']}")
        output["stages_completed"].append("context_build")

        # ── Stage 8: AI Meme Text Generation ──────────────────────────────
        log_stage("AI MEME GENERATION")
        meme_data = generate_meme_text(context)
        print(f"  Top text: {meme_data.get('top_text', 'N/A')}")
        print(f"  Style: {meme_data.get('meme_style', 'N/A')}")
        print(f"  SFX: {meme_data.get('sfx_suggestion', 'N/A')}")
        print(f"  Confidence: {meme_data.get('confidence_score', 'N/A')}")
        output["stages_completed"].append("ai_generate")

        # ── Stage 9: Validation Gate ──────────────────────────────────────
        log_stage("VALIDATION")
        is_valid, reject_reasons = validate_meme(meme_data, context)

        if not is_valid:
            output["status"] = "skipped"
            output["reason"] = "; ".join(reject_reasons)
            print(f"  REJECTED: {output['reason']}")
            save_output(output, video_id)
            notify_n8n(output)
            return

        print("  PASSED ✅")
        output["stages_completed"].append("validation")

        # ── Stage 10: FFmpeg Rendering ────────────────────────────────────
        log_stage("RENDER")
        final_video = render_meme(raw_video, meme_data, transcript_data, video_id)
        output["stages_completed"].append("render")

        # Verify final video
        final_duration = get_video_duration(final_video)
        final_size = os.path.getsize(final_video) / (1024 * 1024)
        print(f"  Final: {final_duration:.1f}s, {final_size:.1f}MB")

        # ── Stage 11: Upload ──────────────────────────────────────────────
        log_stage("UPLOAD")
        download_url = upload_to_catbox(final_video)
        print(f"  URL: {download_url}")
        output["stages_completed"].append("upload")

        # ── Final Output ──────────────────────────────────────────────────
        output.update({
            "status": "ready",
            "download_url": download_url,
            "caption": meme_data.get("caption", ""),
            "meme_data": meme_data,
            "duration": final_duration,
            "file_size_mb": round(final_size, 2),
        })

    except Exception as e:
        output["status"] = "failed"
        output["error"] = str(e)
        output["traceback"] = traceback.format_exc()
        print(f"\n❌ PIPELINE FAILED: {e}", file=sys.stderr)
        traceback.print_exc()

    # Always save output and notify
    save_output(output, video_id)
    notify_n8n(output)

    if output["status"] == "ready":
        print(f"\n✅ SUCCESS: Meme ready at {output.get('download_url')}")
    elif output["status"] == "skipped":
        print(f"\n⏭️ SKIPPED: {output.get('reason')}")
    else:
        print(f"\n❌ FAILED: {output.get('error')}")
        sys.exit(1)


# ─── Stage Functions ─────────────────────────────────────────────────────────

def download_video(url, video_id):
    """Stage 1: Download video using yt-dlp with Reddit fallbacks."""
    output_path = f"{WORK_DIR}/{video_id}_raw.mp4"

    # Check for cookies file
    cookies_arg = []
    has_cookies = os.path.exists("cookies.txt") and os.path.getsize("cookies.txt") > 0
    if has_cookies:
        cookies_arg = ["--cookies", "cookies.txt"]
        print(f"  Cookies file loaded ({os.path.getsize('cookies.txt')} bytes)")
    else:
        print(f"  ⚠️ No cookies.txt found — Reddit downloads may fail!")

    # Common headers for all requests
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "video/*,*/*",
        "Referer": "https://www.reddit.com/",
    }

    # ── Try yt-dlp first (works for non-Reddit URLs) ──────────────────────
    common_args = [
        "--user-agent", HEADERS["User-Agent"],
        "--referer", "https://www.reddit.com/",
        "--geo-bypass",
        "--no-check-certificates",
        "--no-playlist",
        "--retries", "10",
        "--fragment-retries", "10",
        "--socket-timeout", "30",
        "-o", output_path,
    ]

    cmd = [
        "yt-dlp", *cookies_arg,
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/b",
        "--merge-output-format", "mp4",
        *common_args, url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
        return _validate_and_return(output_path, video_id)

    print(f"  yt-dlp failed, trying Reddit API fallback...")

    # ── Reddit JSON API: resolve actual video URL ────────────────────────
    video_download_url = _resolve_reddit_video_url(url, HEADERS)

    if video_download_url:
        print(f"  Found direct video URL, downloading...")
        success = _download_with_requests(video_download_url, output_path, HEADERS)
        if success:
            return _validate_and_return(output_path, video_id)

    # ── Fallback: try v.redd.it DASH URLs directly ────────────────────────
    if "redd.it" in url:
        base_url = url.rstrip("/")
        dash_qualities = ["DASH_720.mp4", "DASH_480.mp4", "DASH_360.mp4", "DASH_240.mp4"]

        for quality in dash_qualities:
            dash_url = f"{base_url}/{quality}"
            print(f"  Trying DASH: {quality}...")
            success = _download_with_requests(dash_url, output_path, HEADERS)
            if success:
                return _validate_and_return(output_path, video_id)

    # ── Last resort: yt-dlp with no format selection ──────────────────────
    cmd_auto = ["yt-dlp", *cookies_arg, *common_args, url]
    result = subprocess.run(cmd_auto, capture_output=True, text=True, timeout=300)

    if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
        return _validate_and_return(output_path, video_id)

    raise RuntimeError(
        f"Download failed (all attempts). URL: {url}\n"
        f"yt-dlp stderr: {result.stderr[-300:]}"
    )


def _load_cookies_for_requests():
    """Load cookies from cookies.txt (Netscape format) for use with requests."""
    cookies = {}
    if not os.path.exists("cookies.txt"):
        return cookies
    try:
        with open("cookies.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 7:
                    cookies[parts[5]] = parts[6]
        print(f"  Loaded {len(cookies)} cookies from cookies.txt")
    except Exception as e:
        print(f"  Warning: Could not parse cookies.txt: {e}")
    return cookies


def _resolve_reddit_video_url(url, headers):
    """Use Reddit's JSON API to extract the actual video download URL."""
    try:
        # Convert Reddit post URL to JSON API URL
        json_url = None

        if "reddit.com" in url:
            # Post URL like https://www.reddit.com/r/funny/comments/abc123/...
            json_url = url.rstrip("/") + ".json"
        elif "redd.it" in url and "v.redd.it" not in url:
            # Short URL like https://redd.it/abc123
            json_url = f"https://www.reddit.com/comments/{url.split('/')[-1]}.json"
        else:
            # v.redd.it URL - try to use it directly
            return url

        if not json_url:
            return None

        # Load cookies for authenticated Reddit access
        cookie_jar = _load_cookies_for_requests()

        print(f"  Resolving via Reddit JSON API...")
        resp = requests.get(json_url, headers=headers, cookies=cookie_jar, timeout=15, allow_redirects=True)

        if resp.status_code != 200:
            print(f"  Reddit JSON API returned {resp.status_code}")
            return None

        data = resp.json()

        # Navigate the Reddit JSON structure
        if isinstance(data, list) and len(data) > 0:
            post_data = data[0].get("data", {}).get("children", [{}])[0].get("data", {})
        elif isinstance(data, dict):
            post_data = data.get("data", {}).get("children", [{}])[0].get("data", {})
        else:
            return None

        # Try to get the video URL from the Reddit media object
        media = post_data.get("secure_media") or post_data.get("media")
        if media and "reddit_video" in media:
            video_url = media["reddit_video"].get("fallback_url")
            if video_url:
                # Remove query params that might cause issues
                video_url = video_url.split("?")[0]
                print(f"  Resolved video URL: {video_url}")
                return video_url

        # Try crosspost parent
        crosspost = post_data.get("crosspost_parent_list", [])
        if crosspost:
            media = crosspost[0].get("secure_media") or crosspost[0].get("media")
            if media and "reddit_video" in media:
                video_url = media["reddit_video"].get("fallback_url")
                if video_url:
                    video_url = video_url.split("?")[0]
                    print(f"  Resolved crosspost video URL: {video_url}")
                    return video_url

        # Try url_overridden_by_dest (external video links)
        external_url = post_data.get("url_overridden_by_dest") or post_data.get("url")
        if external_url and any(ext in external_url for ext in [".mp4", ".webm", "v.redd.it"]):
            print(f"  Found external URL: {external_url}")
            return external_url

        print(f"  Could not resolve video URL from Reddit JSON")
        return None

    except Exception as e:
        print(f"  Reddit JSON API error: {e}")
        return None


def _download_with_requests(url, output_path, headers):
    """Download a file using requests with proper headers. Returns True on success."""
    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True)

        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} for {url}")
            return False

        # Check content-type isn't HTML (error page)
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            print(f"  Got HTML instead of video (blocked)")
            return False

        # Download the file
        total_size = 0
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                total_size += len(chunk)

        # Check minimum file size (10KB = likely not a real video)
        if total_size < 10000:
            print(f"  Downloaded file too small ({total_size} bytes), likely not a video")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False

        print(f"  Downloaded {total_size / 1024 / 1024:.1f}MB")
        return True

    except Exception as e:
        print(f"  Download error: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def _validate_and_return(output_path, video_id):
    """Validate the downloaded file is a real video and return the path."""
    if not os.path.exists(output_path):
        # yt-dlp might have used a different extension
        for ext in [".mp4", ".webm", ".mkv"]:
            alt = f"{WORK_DIR}/{video_id}_raw{ext}"
            if os.path.exists(alt):
                if ext != ".mp4":
                    subprocess.run([
                        "ffmpeg", "-y", "-i", alt,
                        "-c:v", "libx264", "-c:a", "aac", output_path,
                    ], capture_output=True, check=True)
                    os.remove(alt)
                else:
                    output_path = alt
                break
        else:
            raise FileNotFoundError(f"Downloaded file not found at {output_path}")

    # Verify it's a real video file
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", output_path],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        # Check what we actually downloaded
        file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        # Read first bytes to check if it's HTML
        with open(output_path, "rb") as f:
            head = f.read(200)
        if b"<html" in head.lower() or b"<!doctype" in head.lower():
            os.remove(output_path)
            raise RuntimeError("Downloaded an HTML error page instead of video (Reddit blocked)")
        raise RuntimeError(f"Downloaded file is not a valid video (size: {file_size} bytes)")

    print(f"  Downloaded: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f}MB)")
    return output_path


def get_video_duration(video_path):
    """Get video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def trim_video(video_path, max_duration):
    """Trim video to max duration without re-encoding."""
    trimmed = video_path.replace("_raw.", "_trimmed.")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-t", str(max_duration), "-c", "copy", trimmed,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return trimmed


def clean_audio(video_path):
    """Stage 3: Clean audio for better Whisper accuracy.

    - highpass=200Hz: removes low rumble/wind/traffic noise
    - lowpass=3000Hz: removes high-pitched hiss/electronic noise
    - dynaudnorm: normalizes volume levels across the clip
    - Resampled to 16kHz mono (Whisper's expected input)
    """
    output = f"{WORK_DIR}/cleaned_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-af", "highpass=f=200,lowpass=f=3000,dynaudnorm=p=0.9:s=5",
        "-ar", "16000", "-ac", "1",
        output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback: extract audio without cleaning
        print("  WARNING: Audio cleaning failed, extracting raw audio")
        cmd_fallback = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", output,
        ]
        subprocess.run(cmd_fallback, capture_output=True, check=True)

    return output


def transcribe_audio(audio_path):
    """Stage 4: Transcribe using Whisper 'small' model.

    - 'small' model: 244M params, excellent accuracy for Hinglish/English
    - Auto-detects language
    - Returns text + timed segments for subtitle generation
    """
    import whisper

    print(f"  Loading Whisper model ({WHISPER_MODEL})...")
    model = whisper.load_model(WHISPER_MODEL)

    print("  Transcribing...")
    result = model.transcribe(
        audio_path,
        language=None,      # Auto-detect language
        task="transcribe",
        verbose=False,
        fp16=False,          # CPU mode
        condition_on_previous_text=True,
    )

    # Free GPU/CPU memory immediately
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass
    import gc
    gc.collect()

    transcript = result.get("text", "").strip()
    segments = result.get("segments", [])
    detected_language = result.get("language", "unknown")

    # Assess confidence
    confidence = "normal"
    word_count = len(transcript.split())
    if word_count < 3:
        confidence = "very_low"
    elif word_count < 8:
        confidence = "low"

    return {
        "text": transcript,
        "segments": [
            {
                "start": round(s["start"], 2),
                "end": round(s["end"], 2),
                "text": s["text"].strip(),
            }
            for s in segments
        ],
        "language": detected_language,
        "confidence": confidence,
        "word_count": word_count,
    }


def extract_frames(video_path, max_frames=MAX_FRAMES):
    """Stage 5: Extract key frames for vision analysis.

    Uses scene-change detection to get the most informative frames,
    falling back to uniform sampling if scene detection yields too few.
    """
    frames_dir = f"{WORK_DIR}/frames"
    os.makedirs(frames_dir, exist_ok=True)

    # Try scene-change detection first (gets more interesting frames)
    cmd_scene = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select='gt(scene,0.3)',scale=640:-1",
        "-frames:v", str(max_frames),
        "-vsync", "vfr",
        "-q:v", "2",
        f"{frames_dir}/frame_%03d.jpg",
    ]
    subprocess.run(cmd_scene, capture_output=True)

    frames = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))

    # If scene detection gave < 3 frames, fall back to uniform sampling
    if len(frames) < 3:
        # Clear and re-extract with uniform fps
        for f in frames:
            os.remove(f)

        duration = get_video_duration(video_path)
        fps = max(0.5, max_frames / duration)  # At least 0.5 fps

        cmd_uniform = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps={fps:.2f},scale=640:-1",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            f"{frames_dir}/frame_%03d.jpg",
        ]
        subprocess.run(cmd_uniform, capture_output=True, check=True)
        frames = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))

    return frames[:max_frames]


def caption_frames(frame_paths):
    """Stage 6: Generate captions using BLIP-large model.

    Loads model ONCE, processes all frames, then releases memory.
    BLIP-large gives significantly better descriptions than base.
    """
    if not frame_paths:
        return ["no frames extracted"]

    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image

    print(f"  Loading BLIP-large model...")
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL,
        torch_dtype="auto",
    )

    captions = []
    for frame_path in frame_paths:
        try:
            image = Image.open(frame_path).convert("RGB")

            # Conditional captioning (more descriptive)
            inputs = processor(image, text="a photo of", return_tensors="pt")
            output = model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=3,           # Beam search for better quality
                early_stopping=True,
            )
            caption = processor.decode(output[0], skip_special_tokens=True)
            captions.append(caption)
        except Exception as e:
            print(f"  WARNING: Frame captioning failed ({e})")
            captions.append("unclear frame content")

    # Free memory immediately
    del model, processor
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass
    import gc
    gc.collect()

    return captions


# ─── Context Analysis ────────────────────────────────────────────────────────

def detect_emotion(vision_descriptions, transcript):
    """Detect dominant emotion from combined context."""
    text = (" ".join(vision_descriptions) + " " + transcript).lower()

    emotion_keywords = {
        "embarrassed": [
            "embarrass", "shy", "awkward", "caught", "hiding", "cringe",
            "blush", "shame", "oops", "mistake",
        ],
        "angry": [
            "angry", "shout", "fight", "argue", "furious", "yell",
            "scream", "rage", "mad", "slap",
        ],
        "happy": [
            "laugh", "smile", "dance", "happy", "joy", "celebrat",
            "cheer", "grin", "excited", "fun",
        ],
        "shocked": [
            "shock", "surprise", "jaw", "disbelief", "unexpected",
            "omg", "gasp", "stun", "amaz", "wtf",
        ],
        "stressed": [
            "stress", "cry", "panic", "anxious", "worry", "nervous",
            "exam", "deadline", "fear", "sweat",
        ],
        "confused": [
            "confus", "lost", "puzzl", "what", "huh", "scratch",
            "wonder", "think",
        ],
        "cringe": [
            "cringe", "second hand", "painful", "uncomfortable",
            "die inside", "facepalm",
        ],
    }

    scores = {}
    for emotion, keywords in emotion_keywords.items():
        scores[emotion] = sum(1 for k in keywords if k in text)

    if max(scores.values()) == 0:
        return "neutral"
    return max(scores, key=scores.get)


def classify_scene(vision_descriptions, transcript):
    """Classify scene type from context."""
    text = (" ".join(vision_descriptions) + " " + transcript).lower()

    scene_keywords = {
        "classroom": [
            "class", "student", "teacher", "desk", "board", "school",
            "lecture", "exam", "notebook", "backpack",
        ],
        "office": [
            "office", "laptop", "meeting", "boss", "work", "corporate",
            "desk", "computer", "presentation",
        ],
        "street": [
            "street", "road", "car", "traffic", "walk", "outdoor",
            "bike", "scooter", "rickshaw", "highway",
        ],
        "relationship": [
            "girl", "boy", "couple", "date", "crush", "love",
            "hug", "kiss", "flirt", "romantic",
        ],
        "home": [
            "kitchen", "room", "bed", "home", "family", "parent",
            "mom", "dad", "sibling", "couch", "sofa",
        ],
        "party": [
            "party", "club", "music", "dance", "crowd", "concert",
            "dj", "wedding", "celebration",
        ],
        "food": [
            "food", "eat", "restaurant", "cook", "meal", "plate",
            "spicy", "chai", "biryani",
        ],
    }

    scores = {}
    for scene, keywords in scene_keywords.items():
        scores[scene] = sum(1 for k in keywords if k in text)

    if max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)


def build_context(reddit_title, reddit_sub, transcript_data, vision_descriptions):
    """Stage 7: Build unified context from all sources."""
    emotion = detect_emotion(vision_descriptions, transcript_data["text"])
    scene_type = classify_scene(vision_descriptions, transcript_data["text"])

    return {
        "reddit_title": reddit_title,
        "reddit_sub": reddit_sub,
        "transcript": transcript_data["text"],
        "transcript_segments": transcript_data["segments"],
        "transcript_language": transcript_data["language"],
        "audio_confidence": transcript_data["confidence"],
        "word_count": transcript_data["word_count"],
        "vision_descriptions": vision_descriptions,
        "detected_emotion": emotion,
        "scene_type": scene_type,
    }


# ─── AI Meme Generation ─────────────────────────────────────────────────────

def generate_meme_text(context):
    """Stage 8: Generate meme text using Groq API (llama-3.3-70b)."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    prompt = f"""You are an expert viral meme creator for Indian Gen-Z Instagram audience.
You specialize in Hinglish (Hindi + English mix) meme content.

═══ VIDEO CONTEXT ═══

Reddit Title: {context['reddit_title']}
Subreddit: {context['reddit_sub']}

Transcript ({context['transcript_language']}, confidence: {context['audio_confidence']}):
"{context['transcript']}"

Visual Scene Descriptions:
{json.dumps(context['vision_descriptions'], indent=2)}

Detected Emotion: {context['detected_emotion']}
Scene Type: {context['scene_type']}

═══ RULES ═══

1. If transcript is weak/low confidence → rely MORE on visual descriptions
2. If both are unclear → infer the most likely relatable Indian situation
3. top_text MUST be in Hinglish (Hindi-English mix) for Indian audience
4. Must be genuinely FUNNY, relatable, and viral-worthy
5. Keep top_text under 60 characters
6. The meme should feel like something a big meme page would post
7. Use POV/relatable/cringe/savage tone as appropriate
8. If the video has talking, consider "subtitle" style
9. Choose SFX that enhances the punchline timing

═══ RETURN FORMAT ═══

Return ONLY valid JSON, no markdown, no explanation:
{{
  "situation_summary": "Brief 1-2 line description of what happens in the video",
  "meme_hook": "Short attention-grabbing hook (max 10 words)",
  "top_text": "Main meme overlay text — Hinglish, punchy, max 60 chars",
  "bottom_text": "Optional second line for top_bottom style, or empty string",
  "subtitle_clean": "Rewritten clean dialogue for subtitle overlay",
  "hashtags": ["10", "relevant", "indian", "meme", "hashtags"],
  "emotion_detected": "{context['detected_emotion']}",
  "confidence_score": 8,
  "sfx_suggestion": "vine_boom|laugh|bruh|suspense|none",
  "meme_style": "pov|top_bottom|subtitle|caption",
  "caption": "Full Instagram caption — Hinglish hook + emojis + CTA + line break + hashtags"
}}"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are India's top meme page admin. You create viral Hinglish memes "
                    "that get millions of views. You understand Indian culture, Gen-Z humor, "
                    "and what makes content relatable. Return ONLY valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.85,
        "max_tokens": 600,
        "top_p": 0.9,
    }

    # Retry up to 2 times
    for attempt in range(3):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            content = response.json()["choices"][0]["message"]["content"]

            # Clean JSON from potential markdown wrapping
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            meme_data = json.loads(content)

            # Ensure required fields exist
            required = ["top_text", "meme_style", "confidence_score", "caption"]
            for field in required:
                if field not in meme_data:
                    raise ValueError(f"Missing required field: {field}")

            return meme_data

        except json.JSONDecodeError as e:
            print(f"  WARNING: JSON parse failed (attempt {attempt+1}): {e}")
            if attempt == 2:
                raise RuntimeError(f"AI returned invalid JSON after 3 attempts: {content[:200]}")
        except Exception as e:
            print(f"  WARNING: AI generation failed (attempt {attempt+1}): {e}")
            if attempt == 2:
                raise
            time.sleep(2)

    raise RuntimeError("AI generation failed after all retries")


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_meme(meme_data, context):
    """Stage 9: Validate meme quality before rendering.

    Returns (is_valid: bool, reasons: list[str])
    """
    reasons = []

    # Check confidence
    score = meme_data.get("confidence_score", 0)
    if isinstance(score, str):
        try:
            score = int(score)
        except:
            score = 0

    if score < MIN_CONFIDENCE:
        reasons.append(f"Confidence too low ({score} < {MIN_CONFIDENCE})")

    # Check essential fields
    if not meme_data.get("top_text", "").strip():
        reasons.append("No top_text generated")

    if not meme_data.get("caption", "").strip():
        reasons.append("No caption generated")

    # Check if both audio and vision are useless
    audio_useless = context["audio_confidence"] in ("low", "very_low")
    vision_useless = all(
        any(x in desc.lower() for x in ["unclear", "blurry", "dark", "nothing"])
        for desc in context["vision_descriptions"]
    )
    if audio_useless and vision_useless:
        reasons.append("Both audio and vision are unclear/unusable")

    # Check top_text length
    if len(meme_data.get("top_text", "")) > 100:
        reasons.append("top_text too long (> 100 chars)")

    return (len(reasons) == 0, reasons)


# ─── Rendering ───────────────────────────────────────────────────────────────

def generate_srt(segments, output_path):
    """Generate SRT subtitle file from Whisper segments."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip()
            if text:
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def format_timestamp(seconds):
    """Format seconds to SRT timestamp HH:MM:SS,mmm."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def escape_ffmpeg_text(text):
    """Escape text for FFmpeg drawtext filter."""
    # Must escape these characters for FFmpeg
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "'\\''")
    text = text.replace(":", "\\:")
    text = text.replace("%", "%%")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    text = text.replace(";", "\\;")
    return text


def render_meme(video_path, meme_data, transcript_data, video_id):
    """Stage 10: Render final meme video with all overlays.

    Pipeline: text overlay → subtitles → SFX → BGM → anti-detection → scale → watermark
    """
    top_text = meme_data.get("top_text", "")
    bottom_text = meme_data.get("bottom_text", "")
    meme_style = meme_data.get("meme_style", "pov")
    sfx_type = meme_data.get("sfx_suggestion", "none")
    subtitle_text = meme_data.get("subtitle_clean", "")

    current_input = video_path

    # ── Step 1: Text Overlay ──────────────────────────────────────────────
    step1_output = f"{WORK_DIR}/step1_text.mp4"
    text_filter = build_text_filter(top_text, bottom_text, meme_style)

    if text_filter:
        run_ffmpeg([
            "-i", current_input,
            "-vf", text_filter,
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "copy",
            step1_output,
        ])
        current_input = step1_output

    # ── Step 2: Subtitles (from Whisper segments) ─────────────────────────
    if transcript_data.get("segments") and transcript_data["confidence"] != "very_low":
        srt_path = f"{WORK_DIR}/subtitles.srt"

        if meme_style == "subtitle" and subtitle_text:
            # Use AI-rewritten subtitle
            generate_srt(transcript_data["segments"], srt_path)
        else:
            # Use Whisper segments as-is
            generate_srt(transcript_data["segments"], srt_path)

        if os.path.exists(srt_path) and os.path.getsize(srt_path) > 10:
            step2_output = f"{WORK_DIR}/step2_subs.mp4"
            subtitle_filter = (
                f"subtitles={srt_path}:force_style='"
                "FontName=DejaVu Sans Bold,"
                "FontSize=20,"
                "PrimaryColour=&H00FFFFFF,"
                "OutlineColour=&H00000000,"
                "BorderStyle=3,"
                "Outline=2,"
                "Shadow=1,"
                "Alignment=2,"
                "MarginV=40'"
            )
            run_ffmpeg([
                "-i", current_input,
                "-vf", subtitle_filter,
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-c:a", "copy",
                step2_output,
            ])
            current_input = step2_output

    # ── Step 3: Sound Effects ─────────────────────────────────────────────
    sfx_file = find_asset(f"assets/sfx/{sfx_type}.mp3")
    if sfx_type != "none" and sfx_file:
        step3_output = f"{WORK_DIR}/step3_sfx.mp4"
        duration = get_video_duration(current_input)

        # Place SFX 1.5 seconds before the end (punchline moment)
        delay_ms = int(max(0, (duration - 1.5)) * 1000)

        run_ffmpeg([
            "-i", current_input,
            "-i", sfx_file,
            "-filter_complex",
            f"[1:a]adelay={delay_ms}|{delay_ms},volume=0.9[sfx];"
            f"[0:a][sfx]amix=inputs=2:duration=first:dropout_transition=2[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            step3_output,
        ])
        current_input = step3_output

    # ── Step 4: Background Music ──────────────────────────────────────────
    bgm_files = sorted(glob.glob("assets/bgm/*.mp3"))
    if bgm_files:
        bgm_file = random.choice(bgm_files)
        step4_output = f"{WORK_DIR}/step4_bgm.mp4"
        duration = get_video_duration(current_input)

        # Fade out BGM 3 seconds before end
        fade_start = max(0, duration - 4)

        run_ffmpeg([
            "-i", current_input,
            "-i", bgm_file,
            "-filter_complex",
            f"[0:a]volume=1.0[main];"
            f"[1:a]volume=0.13,afade=t=in:d=2,afade=t=out:st={fade_start:.1f}:d=3[bgm];"
            f"[main][bgm]amix=inputs=2:duration=shortest:dropout_transition=3[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            step4_output,
        ])
        current_input = step4_output

    # ── Step 5: Anti-Detection + Scale 9:16 + Watermark ───────────────────
    final_output = f"{WORK_DIR}/{video_id}_final.mp4"

    # Randomize subtle visual shifts to defeat content matching
    brightness = round(random.uniform(0.02, 0.06), 3)
    contrast = round(random.uniform(1.01, 1.05), 3)
    saturation = round(random.uniform(1.02, 1.08), 3)
    # Slight random crop (1-3% from each edge)
    crop_factor = round(random.uniform(0.96, 0.99), 3)

    watermark_escaped = escape_ffmpeg_text(WATERMARK_TEXT)

    vf_chain = (
        f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation},"
        f"crop=in_w*{crop_factor}:in_h*{crop_factor},"
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
        f"drawtext=text='{watermark_escaped}':"
        "fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        "fontcolor=white@0.65:fontsize=24:"
        "x=w-tw-20:y=22:"
        "shadowcolor=black@0.4:shadowx=1:shadowy=1"
    )

    run_ffmpeg([
        "-i", current_input,
        "-vf", vf_chain,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-t", str(MAX_VIDEO_DURATION),
        "-movflags", "+faststart",  # Web-optimized MP4
        final_output,
    ])

    return final_output


def build_text_filter(top_text, bottom_text, style):
    """Build FFmpeg drawtext filter based on meme style."""
    if not top_text:
        return ""

    top_escaped = escape_ffmpeg_text(top_text)
    font = "fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    if style == "pov":
        # POV text at top center with shadow
        return (
            f"drawtext=text='{top_escaped}':"
            f"{font}:"
            "fontsize=44:fontcolor=white:"
            "borderw=3:bordercolor=black:"
            "x=(w-text_w)/2:y=80:"
            "shadowcolor=black@0.6:shadowx=2:shadowy=2"
        )

    elif style == "top_bottom":
        # Classic meme: text at top AND bottom
        top_filter = (
            f"drawtext=text='{top_escaped}':"
            f"{font}:"
            "fontsize=46:fontcolor=white:"
            "borderw=4:bordercolor=black:"
            "x=(w-text_w)/2:y=50"
        )
        if bottom_text:
            bottom_escaped = escape_ffmpeg_text(bottom_text)
            top_filter += (
                f",drawtext=text='{bottom_escaped}':"
                f"{font}:"
                "fontsize=46:fontcolor=white:"
                "borderw=4:bordercolor=black:"
                "x=(w-text_w)/2:y=h-th-50"
            )
        return top_filter

    elif style == "caption":
        # Caption box style — text in a semi-transparent bar at top
        return (
            "drawbox=x=0:y=0:w=iw:h=100:color=black@0.6:t=fill,"
            f"drawtext=text='{top_escaped}':"
            f"{font}:"
            "fontsize=36:fontcolor=white:"
            "x=(w-text_w)/2:y=30"
        )

    else:
        # Default: subtitle style at bottom
        return (
            f"drawtext=text='{top_escaped}':"
            f"{font}:"
            "fontsize=40:fontcolor=white:"
            "borderw=3:bordercolor=black:"
            "x=(w-text_w)/2:y=h-th-100:"
            "shadowcolor=black@0.5:shadowx=2:shadowy=2"
        )


# ─── Upload ──────────────────────────────────────────────────────────────────

def upload_to_catbox(video_path):
    """Stage 11: Upload to temporary file hosting."""
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"  Uploading {file_size:.1f}MB...")

    # Try litterbox first (72h temp hosting)
    try:
        with open(video_path, "rb") as f:
            response = requests.post(
                "https://litterbox.catbox.moe/resources/internals/api.php",
                files={"fileToUpload": f},
                data={"reqtype": "fileupload", "time": "72h"},
                timeout=180,
            )
        url = response.text.strip()
        if url.startswith("https://"):
            return url
    except Exception as e:
        print(f"  Litterbox failed: {e}")

    # Fallback to permanent catbox
    try:
        with open(video_path, "rb") as f:
            response = requests.post(
                "https://catbox.moe/user/api.php",
                files={"fileToUpload": f},
                data={"reqtype": "fileupload"},
                timeout=180,
            )
        url = response.text.strip()
        if url.startswith("https://"):
            return url
    except Exception as e:
        print(f"  Catbox failed: {e}")

    raise RuntimeError("All upload methods failed")


# ─── Utilities ───────────────────────────────────────────────────────────────

def run_ffmpeg(args):
    """Run FFmpeg command with error handling."""
    cmd = ["ffmpeg", "-y"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        # Log the error but try to extract useful info
        stderr = result.stderr[-1000:] if result.stderr else "No stderr"
        raise RuntimeError(f"FFmpeg failed: {stderr}")


def find_asset(relative_path):
    """Find an asset file, checking both repo root and script directory."""
    # Check relative to repo root
    if os.path.exists(relative_path):
        return relative_path
    # Check relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path = os.path.join(script_dir, "..", relative_path)
    if os.path.exists(alt_path):
        return alt_path
    return None


def log_stage(name):
    """Print a visible stage header."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}")


def save_output(data, video_id):
    """Save structured output JSON for debugging."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/{video_id}.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Output saved: {output_path}")


def notify_n8n(data):
    """Send callback to n8n webhook."""
    if not N8N_WEBHOOK_URL:
        print("  WARNING: N8N_WEBHOOK_URL not set, skipping callback")
        return

    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        print(f"  n8n webhook: {response.status_code}")
    except Exception as e:
        print(f"  WARNING: n8n callback failed: {e}")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_time = time.time()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         ORIGINAL MEME FACTORY — Processing Pipeline         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Whisper model: {WHISPER_MODEL}")
    print(f"  BLIP model: {BLIP_MODEL}")
    print(f"  AI model: {GROQ_MODEL}")
    print(f"  Min confidence: {MIN_CONFIDENCE}")

    main()

    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed/60:.1f} minutes")
