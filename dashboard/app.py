from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import AsyncIterator, Dict, Set, List, Optional, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import asyncio
import json
import uuid
import structlog

from .models import *

# Configure structured logging
logger = structlog.get_logger()

# Initialize FastAPI with metadata
app = FastAPI(
    title="Release Workflow Dashboard API",
    description="Multi-client SSE streaming API for release workflow management",
    version="1.0.0",
    docs_url="/docs",  # Enable OpenAPI docs
    redoc_url="/redoc"  # Enable ReDoc
)

# Add CORS middleware for browser compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Internal demo - open CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")


# Global client tracking
@dataclass
class Client:
    id: str
    connected_at: datetime
    user_agent: str = ""

# Store active clients and their queues
active_clients: Dict[str, asyncio.Queue] = {}
client_info: Dict[str, Client] = {}

# Mock change data (in production, this would come from database)
mock_changes = {
    'PROJ-1234': ChangeUpdate(
        id='PROJ-1234',
        title='Fix payment processing bug',
        services=['UserService', 'PaymentAPI'],
        risk=RiskLevel.MEDIUM,
        status=ChangeStatus.DEPLOYING,
        assignee='Sarah Chen',
        maintenance_window='MW-2024-001',
        eta='12 minutes',
        is_my_change=True,
        current_step=3,
        total_steps=5,
        progress=60,
        steps=[
            WorkflowStep(name='Approvals', status=StepStatus.COMPLETE, completed_at='2h ago', 
                        details='PM ✓ Sarah Chen, Dev ✓ John Smith, QA ✓ Maria Rodriguez'),
            WorkflowStep(name='Window', status=StepStatus.COMPLETE, completed_at='Active now',
                        details='Dec 16, 2024 02:00-06:00 GMT • Remaining: 3h 15m'),
            WorkflowStep(name='Merge PRs', status=StepStatus.RUNNING, completed_at='In Progress',
                        details='1 of 2 PRs merged • Current: BB-4522 (Backend API changes)', eta_minutes=3),
            WorkflowStep(name='Deploy', status=StepStatus.NOTSTARTED, completed_at='Waiting',
                        details='5 deployment actions • Services: UserService, PaymentAPI', eta_minutes=15),
            WorkflowStep(name='Verify', status=StepStatus.NOTSTARTED, completed_at='Pending',
                        details='Health endpoints, integration tests, monitoring alerts', eta_minutes=5)
        ]
    ),
    'PROJ-1235': ChangeUpdate(
        id='PROJ-1235',
        title='Database schema migration',
        services=['Database', 'CoreAPI'],
        risk=RiskLevel.HIGH,
        status=ChangeStatus.BLOCKED,
        assignee='John Smith',
        is_my_change=False,
        blockers=['Missing QA approval - Maria Rodriguez', 'No maintenance window scheduled']
    ),
    'PROJ-1236': ChangeUpdate(
        id='PROJ-1236',
        title='Frontend UI updates',
        services=['FrontendApp'],
        risk=RiskLevel.LOW,
        status=ChangeStatus.READY,
        assignee='Alex Wong',
        maintenance_window='MW-2024-002',
        eta='Tonight',
        is_my_change=False
    ),
    'PROJ-1237': ChangeUpdate(
        id='PROJ-1237',
        title='Payment gateway rollback',
        services=['PaymentGateway'],
        risk=RiskLevel.CRITICAL,
        status=ChangeStatus.ROLLBACK,
        assignee='Sarah Chen',
        eta='5 minutes',
        is_my_change=True,
        urgent=True,
        current_step=2,
        total_steps=3,
        progress=67
    )
}

async def add_client(client_id: str, request: Request) -> asyncio.Queue:
    """Add a new client and return their message queue"""
    queue = asyncio.Queue()
    active_clients[client_id] = queue
    
    # Store client info
    client_info[client_id] = Client(
        id=client_id,
        connected_at=datetime.now(),
        user_agent=request.headers.get("user-agent", "")
    )
    
    logger.info("Client connected", client_id=client_id, total_clients=len(active_clients))
    
    # Send initial data to new client
    await send_to_client(client_id, {
        "type": "initial_data",
        "changes": [change.dict() for change in mock_changes.values()],
        "timestamp": datetime.now().isoformat()
    })
    
    # Notify all clients about new connection
    await broadcast_message({
        "type": "client_connected",
        "client_id": client_id,
        "total_clients": len(active_clients),
        "timestamp": datetime.now().isoformat()
    })
    
    return queue

async def remove_client(client_id: str):
    """Remove a client and clean up resources"""
    if client_id in active_clients:
        del active_clients[client_id]
    if client_id in client_info:
        del client_info[client_id]
    
    logger.info("Client disconnected", client_id=client_id, total_clients=len(active_clients))
    
    # Notify remaining clients about disconnection
    await broadcast_message({
        "type": "client_disconnected",
        "client_id": client_id,
        "total_clients": len(active_clients),
        "timestamp": datetime.now().isoformat()
    })

async def broadcast_message(message: dict):
    """Send a message to all connected clients"""
    if not active_clients:
        return
        
    # Add to all client queues
    dead_clients = []
    for client_id, queue in list(active_clients.items()):
        try:
            await queue.put(message)
        except Exception as e:
            logger.error("Error sending to client", client_id=client_id, error=str(e))
            dead_clients.append(client_id)
    
    # Clean up dead clients
    for client_id in dead_clients:
        await remove_client(client_id)

async def send_to_client(client_id: str, message: dict):
    """Send a message to a specific client"""
    if client_id in active_clients:
        try:
            await active_clients[client_id].put(message)
        except Exception as e:
            logger.error("Error sending to client", client_id=client_id, error=str(e))
            await remove_client(client_id)

async def client_stream(request: Request, client_id: str) -> AsyncIterator[str]:
    """Generate SSE stream for a specific client"""
    queue = await add_client(client_id, request)
    
    try:
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break
            
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"data: {json.dumps(message, default=str)}\n\n"
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
                
    except Exception as e:
        logger.error("Stream error", client_id=client_id, error=str(e))
    finally:
        await remove_client(client_id)

async def periodic_updates():
    """Background task to send periodic change updates"""
    counter = 0
    while True:
        if active_clients:
            # Simulate change updates
            for change_id, change in mock_changes.items():
                if change.status in [ChangeStatus.DEPLOYING, ChangeStatus.ROLLBACK]:
                    # Simulate progress
                    if change.progress and change.progress < 100:
                        change.progress = min(100, change.progress + 5)
                        if change.progress >= 100:
                            change.status = ChangeStatus.COMPLETE
                        
                        # Send update to all clients
                        await broadcast_message({
                            "type": "change_update",
                            "change": change.dict(),
                            "timestamp": datetime.now().isoformat()
                        })
            
            # Send periodic status
            await broadcast_message({
                "type": "periodic_update",
                "counter": counter,
                "timestamp": datetime.now().isoformat(),
                "active_clients": len(active_clients),
                "total_changes": len(mock_changes)
            })
            
        counter += 1
        await asyncio.sleep(10)  # Update every 10 seconds

# Lifespan management
@asyncio.coroutine
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Release Dashboard API")
    update_task = asyncio.create_task(periodic_updates())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Release Dashboard API")
    update_task.cancel()
    try:
        await update_task
    except asyncio.CancelledError:
        pass

app.router.lifespan_context = lifespan

# API Routes

@app.get("/", response_class=HTMLResponse, summary="Test SSE Client")
async def root(request: Request):
    """Enhanced HTML client to test multi-client SSE"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse, summary="Release Dashboard")
@app.get("/dashboard.html", response_class=HTMLResponse, summary="Release Dashboard")
async def dashboard(request: Request):
    """Main release workflow dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/stream", summary="SSE Event Stream")
async def stream_events(request: Request):
    """
    SSE endpoint that supports multiple clients.
    Each client gets a unique ID and their own stream.
    """
    client_id = str(uuid.uuid4())
    
    return StreamingResponse(
        client_stream(request, client_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Client-ID": client_id,
        }
    )

@app.post("/broadcast", summary="Broadcast Message")
async def broadcast_to_all(message: BroadcastMessage):
    """API endpoint to broadcast a message to all connected clients"""
    broadcast_data = {
        "type": "broadcast",
        "message": message.message,
        "timestamp": datetime.now().isoformat()
    }
    await broadcast_message(broadcast_data)
    
    return {
        "status": "success",
        "message": "Broadcasted to all clients",
        "client_count": len(active_clients)
    }

@app.post("/send/{client_id}", summary="Send Direct Message")
async def send_to_specific_client(client_id: str, message: DirectMessage):
    """API endpoint to send a message to a specific client"""
    if client_id not in active_clients:
        raise HTTPException(status_code=404, detail="Client not found")
    
    direct_data = {
        "type": "direct_message",
        "message": message.message,
        "timestamp": datetime.now().isoformat()
    }
    await send_to_client(client_id, direct_data)
    
    return {"status": "success", "message": f"Sent to client {client_id}"}

@app.get("/clients", response_model=Dict, summary="Get Active Clients")
async def get_active_clients():
    """Get information about all active clients"""
    return {
        "total_clients": len(active_clients),
        "clients": [asdict(client) for client in client_info.values()]
    }

@app.get("/changes", response_model=List[ChangeUpdate], summary="Get All Changes")
async def get_all_changes():
    """Get current state of all changes"""
    return list(mock_changes.values())

@app.get("/changes/{change_id}", response_model=ChangeUpdate, summary="Get Specific Change")
async def get_change(change_id: str):
    """Get details for a specific change"""
    if change_id not in mock_changes:
        raise HTTPException(status_code=404, detail="Change not found")
    return mock_changes[change_id]

@app.post("/changes/{change_id}/update", summary="Update Change Status")
async def update_change(change_id: str, update_data: Dict):
    """Update a change and broadcast to all clients"""
    if change_id not in mock_changes:
        raise HTTPException(status_code=404, detail="Change not found")
    
    # Update change data
    change = mock_changes[change_id]
    for key, value in update_data.items():
        if hasattr(change, key):
            setattr(change, key, value)
    
    # Broadcast update
    await broadcast_message({
        "type": "change_update",
        "change": change.dict(),
        "timestamp": datetime.now().isoformat()
    })
    
    return {"status": "success", "change": change}

# Health check
@app.get("/health", summary="Health Check")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_clients": len(active_clients),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )