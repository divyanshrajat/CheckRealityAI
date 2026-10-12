from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

client = AsyncIOMotorClient(settings.MONGO_URI)
db = client.checkreality

# Collections
users_collection = db.users
submissions_collection = db.submissions
modality_results_collection = db.modality_results
reports_collection = db.reports
model_registry_collection = db.model_registry

async def ping_db():
    try:
        await db.command("ping")
        return True
    except Exception:
        return False
