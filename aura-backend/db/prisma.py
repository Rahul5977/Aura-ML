from prisma import Prisma
import asyncio

# Global Prisma client instance
db = Prisma()

async def connect_db():
    """Connect to the database using Prisma client"""
    try:
        await db.connect()
        print("Database connected successfully")
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        raise

async def disconnect_db():
    """Disconnect from the database"""
    try:
        await db.disconnect()
        print("Database disconnected successfully")
    except Exception as e:
        print(f"Error disconnecting from database: {e}")

async def get_db():
    """Dependency function to get database instance"""
    return db
