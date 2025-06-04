from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import AsyncIterator, Dict, Set
from dataclasses import dataclass, asdict

app = FastAPI(title="Multi-Client SSE Streaming API")

# Add CORS middleware for browser compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global client tracking
@dataclass
class Client:
    id: str
    connected_at: datetime
    user_agent: str = ""

# Store active clients and their queues
active_clients: Dict[str, asyncio.Queue] = {}
client_info: Dict[str, Client] = {}

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
    
    print(f"Client {client_id} connected. Total clients: {len(active_clients)}")
    
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
    
    print(f"Client {client_id} disconnected. Total clients: {len(active_clients)}")
    
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
    for client_id, queue in list(active_clients.items()):
        try:
            await queue.put(message)
        except Exception as e:
            print(f"Error sending to client {client_id}: {e}")
            # Remove problematic client
            await remove_client(client_id)

async def send_to_client(client_id: str, message: dict):
    """Send a message to a specific client"""
    if client_id in active_clients:
        try:
            await active_clients[client_id].put(message)
        except Exception as e:
            print(f"Error sending to client {client_id}: {e}")
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
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield f"data: {json.dumps(message)}\n\n"
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
                
    except Exception as e:
        print(f"Stream error for client {client_id}: {e}")
    finally:
        await remove_client(client_id)

async def periodic_broadcast():
    """Background task to send periodic updates to all clients"""
    counter = 0
    while True:
        if active_clients:
            await broadcast_message({
                "type": "periodic_update",
                "counter": counter,
                "timestamp": datetime.now().isoformat(),
                "active_clients": len(active_clients)
            })
        counter += 1
        await asyncio.sleep(5)  # Broadcast every 5 seconds

# Start background task when app starts
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_broadcast())

@app.get("/stream")
async def stream_events(request: Request):
    """
    SSE endpoint that supports multiple clients
    Each client gets a unique ID and their own stream
    """
    client_id = str(uuid.uuid4())
    
    return StreamingResponse(
        client_stream(request, client_id),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Client-ID": client_id,
        }
    )

@app.post("/broadcast")
async def broadcast_to_all(message: dict):
    """
    API endpoint to broadcast a message to all connected clients
    """
    message["type"] = "broadcast"
    message["timestamp"] = datetime.now().isoformat()
    await broadcast_message(message)
    
    return {
        "status": "success",
        "message": "Broadcasted to all clients",
        "client_count": len(active_clients)
    }

@app.post("/send/{client_id}")
async def send_to_specific_client(client_id: str, message: dict):
    """
    API endpoint to send a message to a specific client
    """
    if client_id not in active_clients:
        return {"status": "error", "message": "Client not found"}
    
    message["type"] = "direct_message"
    message["timestamp"] = datetime.now().isoformat()
    await send_to_client(client_id, message)
    
    return {"status": "success", "message": f"Sent to client {client_id}"}

@app.get("/clients")
async def get_active_clients():
    """
    Get information about all active clients
    """
    return {
        "total_clients": len(active_clients),
        "clients": [asdict(client) for client in client_info.values()]
    }

@app.get("/")
async def root():
    """
    Enhanced HTML client to test multi-client SSE
    """
    html_content = """
        TODO return templates/index.html
    """
    return StreamingResponse(
        iter([html_content]),
        media_type="text/html"
    )

@app.get("/dashboard.html")
async def dashboard():
    #TODO return templates/dashboard.html
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)