from pydantic import BaseModel, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    name: Optional[str] = None
    full_name: Optional[str] = None
    email: EmailStr
    password: str

    def get_name(self) -> str:
        return self.name or self.full_name or self.email.split('@')[0]

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    name: str
    email: EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str
    user: Optional[UserResponse] = None

class SignupResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse
