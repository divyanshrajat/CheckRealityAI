from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime
from bson import ObjectId
from app.models import UserCreate, UserLogin, Token, UserResponse, SignupResponse
from app.db import users_collection
from app.security import get_password_hash, verify_password, create_access_token, decode_access_token

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = await users_collection.find_one({"email": payload.get("sub")})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

@router.post("/signup", response_model=SignupResponse)
async def signup(user: UserCreate):
    existing = await users_collection.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_name = user.get_name()
    hashed_pw = get_password_hash(user.password)
    new_user = {
        "name": user_name,
        "email": user.email,
        "password_hash": hashed_pw,
        "created_at": datetime.utcnow()
    }
    result = await users_collection.insert_one(new_user)
    token = create_access_token(data={"sub": new_user["email"], "id": str(result.inserted_id)})
    user_resp = UserResponse(id=str(result.inserted_id), name=user_name, email=new_user["email"])
    return SignupResponse(access_token=token, token_type="bearer", user=user_resp)

@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    if user.email == "demo@checkreality.ai" and user.password == "demo123":
        db_user = await users_collection.find_one({"email": user.email})
        if not db_user:
            new_user = {
                "name": "Demo Analyst",
                "email": user.email,
                "password_hash": get_password_hash(user.password),
                "created_at": datetime.utcnow()
            }
            result = await users_collection.insert_one(new_user)
            db_user = await users_collection.find_one({"_id": result.inserted_id})
    else:
        db_user = await users_collection.find_one({"email": user.email})
        if not db_user or not verify_password(user.password, db_user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token(data={"sub": db_user["email"], "id": str(db_user["_id"])})
    user_resp = UserResponse(id=str(db_user["_id"]), name=db_user.get("name", "Analyst"), email=db_user["email"])
    return {"access_token": token, "token_type": "bearer", "user": user_resp}

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(id=str(current_user["_id"]), name=current_user["name"], email=current_user["email"])
