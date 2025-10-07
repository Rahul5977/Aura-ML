from fastapi import FastAPI
from core.config import load_config
from db.prisma import connect_db, disconnect_db
from api.auth import router as auth_router
from api.conversation import router as conversation_router
from api.message import router as message_router

load_config()

app = FastAPI(title="Aura Backend API", version="1.0.0")

# Startup and shutdown events
@app.on_event("startup")
async def startup():
    await connect_db()

@app.on_event("shutdown")
async def shutdown():
    await disconnect_db()

@app.get("/health")
def health():
    return {"status": "ok"}

# Include routers
app.include_router(auth_router)
app.include_router(conversation_router)
app.include_router(message_router)
