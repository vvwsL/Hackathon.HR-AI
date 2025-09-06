import requests

AI_VIDEO_API_URL = "http://remote-video-server:8002/analyze_video"

def analyze_video(video_bytes: bytes, filename: str):
    files = {"file": (filename, video_bytes, "video/webm")}
    try:
        response = requests.post(AI_VIDEO_API_URL, files=files)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}
