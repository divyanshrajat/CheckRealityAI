from fastapi import APIRouter, Depends, HTTPException
from bson import ObjectId
from app.routers.auth import get_current_user
from app.db import reports_collection

router = APIRouter()

@router.get("/{id}")
async def get_report(id: str, current_user: dict = Depends(get_current_user)):
    report = await reports_collection.find_one({"_id": ObjectId(id), "user_id": current_user["_id"]})
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    report["_id"] = str(report["_id"])
    report["user_id"] = str(report["user_id"])
    report["submission_id"] = str(report["submission_id"])
    return report
