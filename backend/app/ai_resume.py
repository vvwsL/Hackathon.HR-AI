import requests

AI_RESUME_API_URL = "http://remote-resume-server:8001/analyze_resume"

def analyze_resume(resume_bytes: bytes):
    files = {"file": ("resume.pdf", resume_bytes, "application/pdf")}
    try:
        response = requests.post(AI_RESUME_API_URL, files=files)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}
