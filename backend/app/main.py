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
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

frontend_dist = os.path.join(os.path.dirname(__file__), "../../frontend/dist")

if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

if os.path.exists(frontend_dist):

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        path = os.path.join(frontend_dist, full_path)
        if os.path.exists(path) and os.path.isfile(path):
            return FileResponse(path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
