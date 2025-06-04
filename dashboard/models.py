# Pydantic Models for API
from pydantic import BaseModel, Field
from typing import AsyncIterator, Dict, Set, List, Optional, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class ChangeStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    WAITING_WINDOW = "waiting_window"
    DEPLOYING = "deploying"
    COMPLETE = "complete"
    FAILED = "failed"
    BLOCKED = "blocked"
    ROLLBACK = "rollback"
    READY = "ready"

class StepStatus(str, Enum):
    NOTSTARTED = "notstarted"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"

class WorkflowStep(BaseModel):
    name: str = Field(..., description="Step name")
    status: StepStatus = Field(..., description="Current step status")
    completed_at: Optional[str] = Field(None, description="When step was completed")
    details: Optional[str] = Field(None, description="Additional step details")
    eta_minutes: Optional[int] = Field(None, description="Estimated time to completion")

class ChangeUpdate(BaseModel):
    """Pydantic model for streaming change updates to dashboard"""
    id: str = Field(..., description="Change ID (e.g., PROJ-1234)")
    title: str = Field(..., description="Change title/description")
    services: List[str] = Field(..., description="Affected services")
    risk: RiskLevel = Field(..., description="Risk level")
    status: ChangeStatus = Field(..., description="Current status")
    assignee: str = Field(..., description="Person responsible")
    maintenance_window: Optional[str] = Field(None, description="Maintenance window ID")
    eta: Optional[str] = Field(None, description="Estimated completion time")
    is_my_change: bool = Field(False, description="Whether this is user's change")
    current_step: Optional[int] = Field(None, description="Current workflow step")
    total_steps: Optional[int] = Field(None, description="Total workflow steps")
    progress: Optional[int] = Field(None, description="Progress percentage")
    blockers: Optional[List[str]] = Field(None, description="Current blockers")
    steps: Optional[List[WorkflowStep]] = Field(None, description="Workflow steps")
    urgent: bool = Field(False, description="Urgent flag")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update time")

class BroadcastMessage(BaseModel):
    """Message for broadcasting to all clients"""
    message: str = Field(..., description="Message content")
    type: Optional[str] = Field("broadcast", description="Message type")

class DirectMessage(BaseModel):
    """Message for sending to specific client"""
    message: str = Field(..., description="Message content")
    type: Optional[str] = Field("direct_message", description="Message type")

class SSEMessage(BaseModel):
    """Base SSE message structure"""
    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Optional[Dict] = Field(None, description="Message payload")

class ClientInfo(BaseModel):
    """Client connection information"""
    id: str = Field(..., description="Unique client ID")
    connected_at: datetime = Field(..., description="Connection timestamp")
    user_agent: str = Field("", description="Client user agent")
