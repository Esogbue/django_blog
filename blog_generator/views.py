from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings
import json
from yt_dlp import YoutubeDL
import os
import time
import assemblyai as aai
from groq import Groq
import logging
import traceback
from .models import BlogPost

logger = logging.getLogger(__name__)

# Create your views here.
@login_required
def index(request):
    return render(request, 'blog_generator/index.html')

@login_required
@csrf_exempt
def generate_blog(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            yt_link = data['link']
        except(KeyError, json.JSONDecodeError):
            return JsonResponse({'error': 'Invalid data sent.'}, status=400)
        

        # get yt title
        title = yt_title(yt_link)

        # get transcript
        try:
            transcription = get_transcription(yt_link)
        except RuntimeError as e:
            logger.error(f"Transcription error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
        except Exception as e:
            logger.error(f"Unexpected error in transcription: {str(e)}\n{traceback.format_exc()}")
            return JsonResponse({'error': "Failed to get transcript."}, status=500)

        if not transcription:
            return JsonResponse({'error': "Failed to get transcript."}, status=500)

        # use OpenAI to generate blog
        try:
            blog_content = generate_blog_from_transcription(transcription)
        except RuntimeError as e:
            logger.error(f"Blog generation error: {str(e)}")
            msg = str(e)
            if 'rate limit' in msg.lower() or 'quota' in msg.lower():
                return JsonResponse({'error': msg}, status=503)
            return JsonResponse({'error': msg}, status=500)
        except Exception as e:
            logger.error(f"Unexpected error in blog generation: {str(e)}\n{traceback.format_exc()}")
            return JsonResponse({'error': f"Failed to generate blog article: {str(e)}"}, status=500)

        if not blog_content:
            return JsonResponse({'error': "Failed to generate blog article."}, status=500)

        # save blog article to database
        new_blog_article = BlogPost.objects.create(
            user=request.user,
            youtube_title=title,
            youtube_link=yt_link,
            content=blog_content
        )
        new_blog_article.save()

        # return blog article as a response
        return JsonResponse({'content': blog_content, 'title': title})
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=405)

def yt_title(link):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(link, download=False)
            return info.get('title')
    except Exception:
        return None

def download_audio(link):
    # Add some robustness: longer socket timeout, retries, and resume support
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(settings.MEDIA_ROOT, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 60,
        'retries': 3,
        'continuedl': True,
    }

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(link, download=True)
                # The postprocessor should have created an mp3 file using the video id
                expected_file = os.path.join(settings.MEDIA_ROOT, f"{info.get('id')}.mp3")
                if os.path.exists(expected_file):
                    return expected_file
                # Fallback: search for a matching mp3 file in MEDIA_ROOT
                for root, _, files in os.walk(settings.MEDIA_ROOT):
                    for f in files:
                        if f.startswith(info.get('id', '')) and f.endswith('.mp3'):
                            return os.path.join(root, f)
                # If not found, return None
                return None
        except Exception:
            # On the final attempt, give up and return None
            if attempt == max_attempts:
                return None
            # Exponential backoff before retrying
            time.sleep(2 ** attempt)
            continue

def get_transcription(link):
    audio_file = download_audio(link)
    if not audio_file:
        return None
    aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')
    if not aai.settings.api_key:
        raise RuntimeError("AssemblyAI API key not configured. Set ASSEMBLYAI_API_KEY in your environment.")

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)

    return getattr(transcript, 'text', None)

def generate_blog_from_transcription(transcription):
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise RuntimeError("Groq API key not configured. Set GROQ_API_KEY in your environment.")

    client = Groq(api_key=api_key)
    
    prompt = f"Based on the following transcript from a YouTube video, write a comprehensive blog article, write it based on the transcript, but dont make it look like a youtube video, make it look like a proper blog article:\n\n{transcription}\n\nArticle:"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that converts video transcripts into well-structured, professional blog articles."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Groq API error: {str(e)}") from e

def blog_list(request):
    blog_articles = BlogPost.objects.filter(user=request.user)
    return render(request, 'blog_generator/all-blogs.html', {"blog_articles": blog_articles})

def blog_details(request, pk):
    blog_article_detail = BlogPost.objects.get(id=pk)
    if request.user == blog_article_detail.user:
        return render(request, 'blog_generator/blog-details.html', {"blog_article_detail": blog_article_detail})
    else:
        return redirect('/')
