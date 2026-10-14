from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, submissions, reports

app = FastAPI(title="CheckReality AI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(submissions.router, prefix="/submissions", tags=["submissions"])
app.include_router(reports.router, prefix="/reports", tags=["reports"])

@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}

# Mount the static frontend build if it exists
import os
import mimetypes
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Fix for Windows registry missing standard MIME types
# Media MIME types registration for video and audio playback
MEDIA_MIME_TYPES = {
    ".js": "application/javascript",
    ".css": "text/css",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".ogv": "video/ogg",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
for ext, mime in MEDIA_MIME_TYPES.items():
    mimetypes.add_type(mime, ext)

frontend_dist = os.path.join(os.path.dirname(__file__), "../../frontend/dist")

if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

from fastapi import Request, Response, HTTPException
from fastapi.responses import StreamingResponse

@app.get("/uploads/{filename:path}")
async def serve_upload_file(filename: str, request: Request):
    """
    Streams uploaded media files supporting HTTP 206 Partial Content (Range requests).
    This ensures HTML5 video and audio players in browsers like Chrome, Edge, and Safari
    can buffer, seek, and decode audio and video streams with full fidelity and sound.
    """
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(uploads_dir, safe_filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    file_size = os.path.getsize(file_path)
    _, ext = os.path.splitext(safe_filename)
    ext = ext.lower()
    content_type = MEDIA_MIME_TYPES.get(ext) or mimetypes.guess_type(file_path)[0] or "application/octet-stream"

    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(
            file_path,
            media_type=content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Cache-Control": "public, max-age=3600",
            }
        )

    try:
        range_str = range_header.replace("bytes=", "").strip()
        parts = range_str.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{file_size}"}
            )
        chunk_size = (end - start) + 1
    except Exception:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    def file_chunk_generator():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = chunk_size
            buffer_size = 128 * 1024
            while remaining > 0:
                read_bytes = f.read(min(buffer_size, remaining))
                if not read_bytes:
                    break
                remaining -= len(read_bytes)
                yield read_bytes

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Content-Type": content_type,
        "Cache-Control": "no-cache",
    }
    return StreamingResponse(
        file_chunk_generator(),
        status_code=206,
        headers=headers,
        media_type=content_type
    )

if os.path.exists(frontend_dist):

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        path = os.path.join(frontend_dist, full_path)
        if os.path.exists(path) and os.path.isfile(path):
            return FileResponse(path)
        return FileResponse(
            os.path.join(frontend_dist, "index.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
