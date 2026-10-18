from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import Optional
from bson import ObjectId
from datetime import datetime
from app.routers.auth import get_current_user
from app.db import submissions_collection
import os
import shutil
import asyncio

router = APIRouter()

async def process_submission_background(submission_id: ObjectId, modality: str, file_path: Optional[str], text_content: Optional[str]):
    from app.inference import run_inference, multimodal_fusion, generate_xai_explanation
    import asyncio
    
    loop = asyncio.get_event_loop()
    
    try:
        # Run synchronous inference in a threadpool
        inference_result = await loop.run_in_executor(None, run_inference, modality, file_path, text_content)
        
        # Explainable AI (XAI)
        xai_result = await loop.run_in_executor(None, generate_xai_explanation, modality, file_path, text_content)
        
        # Multimodal Fusion (Single or Multi)
        fusion_inputs = {}
        fusion_features = {}
        if modality == 'multimodal' and isinstance(inference_result, dict):
            for m, res in inference_result.items():
                if isinstance(res, dict):
                    if 'fake_probability' in res: fusion_inputs[m] = res['fake_probability']
                    elif 'spoof_probability' in res: fusion_inputs[m] = res['spoof_probability']
                    elif 'misinformation_probability' in res: fusion_inputs[m] = res['misinformation_probability']
                    if '_feature_vector' in res: fusion_features[m] = res['_feature_vector']
        else:
            if 'fake_probability' in inference_result:
                fusion_inputs[modality] = inference_result['fake_probability']
                if '_feature_vector' in inference_result: fusion_features[modality] = inference_result['_feature_vector']
                if 'audio_stream' in inference_result and isinstance(inference_result['audio_stream'], dict):
                    if 'spoof_probability' in inference_result['audio_stream']:
                        fusion_inputs['audio'] = inference_result['audio_stream']['spoof_probability']
                    if '_feature_vector' in inference_result['audio_stream']:
                        fusion_features['audio'] = inference_result['audio_stream']['_feature_vector']
            elif 'spoof_probability' in inference_result:
                fusion_inputs[modality] = inference_result['spoof_probability']
                if '_feature_vector' in inference_result: fusion_features[modality] = inference_result['_feature_vector']
            elif 'misinformation_probability' in inference_result:
                prob = inference_result['misinformation_probability']
                if 'ai_generated_probability' in inference_result:
                    prob = max(prob, inference_result['ai_generated_probability'])
                fusion_inputs[modality] = prob
                if '_feature_vector' in inference_result: fusion_features[modality] = inference_result['_feature_vector']
            
        fusion_result = multimodal_fusion(fusion_inputs, features=fusion_features)
        
        # Remove internal tensor objects before database persistence
        if isinstance(inference_result, dict):
            for k, v in inference_result.items():
                if isinstance(v, dict):
                    v.pop('_feature_vector', None)
            inference_result.pop('_feature_vector', None)
        
        final_result = {
            "inference": inference_result,
            "fusion": fusion_result,
            "xai": xai_result
        }
        
        await submissions_collection.update_one(
            {"_id": submission_id},
            {"$set": {"status": "completed", "result": final_result, "updated_at": datetime.utcnow()}}
        )
    except Exception as e:
        await submissions_collection.update_one(
            {"_id": submission_id},
            {"$set": {"status": "failed", "error": str(e), "updated_at": datetime.utcnow()}}
        )
    finally:
        # File is kept so frontend can display it in the report
        pass

@router.post("/")
async def create_submission(
    background_tasks: BackgroundTasks,
    modality: str = Form(...),
    text_content: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    file_path = None
    if file:
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{ObjectId()}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    new_sub = {
        "user_id": current_user["_id"],
        "modalities": [modality],
        "original_filename": file.filename if file else None,
        "file_url": f"/uploads/{os.path.basename(file_path)}" if file_path else None,
        "text_content": text_content,
        "status": "processing",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    res = await submissions_collection.insert_one(new_sub)
    
    background_tasks.add_task(
        process_submission_background, 
        res.inserted_id, 
        modality, 
        file_path, 
        text_content
    )

    # Return updated submission
    updated_sub = await submissions_collection.find_one({"_id": res.inserted_id})
    updated_sub["_id"] = str(updated_sub["_id"])
    updated_sub["user_id"] = str(updated_sub["user_id"])
    return updated_sub

@router.get("/")
async def list_submissions(current_user: dict = Depends(get_current_user)):
    # TODO: Implement history listing
    cursor = submissions_collection.find({"user_id": current_user["_id"]}).sort("created_at", -1)
    subs = await cursor.to_list(length=100)
    for sub in subs:
        sub["_id"] = str(sub["_id"])
        sub["user_id"] = str(sub["user_id"])
    return subs

@router.get("/{id}")
async def get_submission(id: str, current_user: dict = Depends(get_current_user)):
    sub = await submissions_collection.find_one({"_id": ObjectId(id), "user_id": current_user["_id"]})
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    sub["_id"] = str(sub["_id"])
    sub["user_id"] = str(sub["user_id"])
    return sub

@router.delete("/{id}")
async def delete_submission(id: str, current_user: dict = Depends(get_current_user)):
    res = await submissions_collection.delete_one({"_id": ObjectId(id), "user_id": current_user["_id"]})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Submission not found")
    return {"message": "Deleted successfully"}
