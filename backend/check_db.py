import asyncio
import motor.motor_asyncio
from bson import ObjectId

async def main():
    client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://localhost:27017')
    db = client['checkreality']
    
    users = await db.users.find().to_list(10)
    print("USERS:")
    for u in users:
        print(f"ID: {u['_id']}, Email: {u.get('email')}")
        
    subs = await db.submissions.find().to_list(10)
    print("\nSUBMISSIONS:")
    for s in subs:
        print(f"Sub ID: {s['_id']}, User ID: {s.get('user_id')} (Type: {type(s.get('user_id'))})")

if __name__ == '__main__':
    asyncio.run(main())
