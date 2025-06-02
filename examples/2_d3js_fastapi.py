from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio
from datetime import datetime

# Router instance
router = APIRouter(prefix="/api/graph", tags=["graph"])

# Enums for node types
class NodeType(str, Enum):
    SERVER = "server"
    DATABASE = "database"
    SERVICE = "service"
    NETWORK = "network"
    STORAGE = "storage"
    APPLICATION = "application"

class OSType(str, Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"

# Pydantic models
class NodeTemplate(BaseModel):
    icon: str
    color: str
    shape: str = "rect"
    default_size: Dict[str, int] = Field(default={"width": 100, "height": 60})
    properties: List[str] = Field(default=[])

class NodeData(BaseModel):
    id: str
    name: str
    display_name: str
    node_type: NodeType
    x: Optional[float] = None
    y: Optional[float] = None
    properties: Dict[str, Any] = Field(default={})
    connections: List[str] = Field(default=[])
    metadata: Dict[str, Any] = Field(default={})

class EdgeData(BaseModel):
    id: str
    source: str
    target: str
    label: Optional[str] = None
    edge_type: str = "default"
    properties: Dict[str, Any] = Field(default={})

class GraphData(BaseModel):
    nodes: List[NodeData]
    edges: List[EdgeData]
    metadata: Dict[str, Any] = Field(default={})

class GraphFilter(BaseModel):
    search_term: Optional[str] = None
    node_types: Optional[List[NodeType]] = None
    properties: Optional[Dict[str, Any]] = None

# Mock data store (replace with actual database)
class GraphDataStore:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.templates = self._init_templates()
        self._populate_sample_data()
    
    def _init_templates(self) -> Dict[NodeType, NodeTemplate]:
        return {
            NodeType.SERVER: NodeTemplate(
                icon="server",
                color="#3b82f6",
                properties=["hostname", "os", "cpu_cores", "memory_gb", "uptime"]
            ),
            NodeType.DATABASE: NodeTemplate(
                icon="database",
                color="#059669",
                properties=["db_type", "version", "size_gb", "connections"]
            ),
            NodeType.SERVICE: NodeTemplate(
                icon="service",
                color="#7c3aed",
                properties=["service_type", "port", "status", "version"]
            ),
            NodeType.NETWORK: NodeTemplate(
                icon="network",
                color="#ea580c",
                properties=["device_type", "vlan", "bandwidth", "protocol"]
            ),
            NodeType.STORAGE: NodeTemplate(
                icon="storage",
                color="#dc2626",
                properties=["capacity_tb", "used_percent", "raid_level", "mount_point"]
            ),
            NodeType.APPLICATION: NodeTemplate(
                icon="application",
                color="#9333ea",
                properties=["app_type", "version", "users", "deployment"]
            )
        }
    
    def _populate_sample_data(self):
        # Sample nodes
        sample_nodes = [
            NodeData(
                id="web-01",
                name="web-01.company.com",
                display_name="Web Server 01",
                node_type=NodeType.SERVER,
                x=100, y=100,
                properties={
                    "hostname": "web-01.company.com",
                    "os": "linux",
                    "cpu_cores": 8,
                    "memory_gb": 32,
                    "uptime": "45 days"
                },
                connections=["db-01", "cache-01"],
                metadata={"environment": "production", "team": "platform"}
            ),
            NodeData(
                id="db-01",
                name="postgres-primary",
                display_name="PostgreSQL Primary",
                node_type=NodeType.DATABASE,
                x=300, y=150,
                properties={
                    "db_type": "postgresql",
                    "version": "15.2",
                    "size_gb": 500,
                    "connections": 45
                },
                connections=["web-01", "backup-storage"],
                metadata={"environment": "production", "team": "data"}
            ),
            NodeData(
                id="cache-01",
                name="redis-cluster",
                display_name="Redis Cache",
                node_type=NodeType.SERVICE,
                x=200, y=250,
                properties={
                    "service_type": "redis",
                    "port": 6379,
                    "status": "running",
                    "version": "7.0"
                },
                connections=["web-01"],
                metadata={"environment": "production", "team": "platform"}
            ),
            NodeData(
                id="backup-storage",
                name="backup-nas",
                display_name="Backup Storage",
                node_type=NodeType.STORAGE,
                x=400, y=200,
                properties={
                    "capacity_tb": 10,
                    "used_percent": 65,
                    "raid_level": "RAID-6",
                    "mount_point": "/backup"
                },
                connections=["db-01"],
                metadata={"environment": "production", "team": "data"}
            )
        ]
        
        # Sample edges
        sample_edges = [
            EdgeData(
                id="web-db",
                source="web-01",
                target="db-01",
                label="Database Connection",
                edge_type="database",
                properties={"protocol": "TCP", "port": 5432}
            ),
            EdgeData(
                id="web-cache",
                source="web-01",
                target="cache-01",
                label="Cache Connection",
                edge_type="cache",
                properties={"protocol": "TCP", "port": 6379}
            ),
            EdgeData(
                id="db-backup",
                source="db-01",
                target="backup-storage",
                label="Backup",
                edge_type="backup",
                properties={"schedule": "daily", "retention": "30 days"}
            )
        ]
        
        # Store in mock database
        for node in sample_nodes:
            self.nodes[node.id] = node
        
        for edge in sample_edges:
            self.edges[edge.id] = edge

# Global data store instance
data_store = GraphDataStore()

@router.get("/templates", response_model=Dict[NodeType, NodeTemplate])
async def get_node_templates():
    """Get all node type templates for rendering"""
    return data_store.templates

@router.get("/nodes", response_model=List[NodeData])
async def get_nodes(
    node_type: Optional[NodeType] = None,
    search: Optional[str] = Query(None, description="Search term for node names"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get nodes with optional filtering"""
    nodes = list(data_store.nodes.values())
    
    # Filter by node type
    if node_type:
        nodes = [n for n in nodes if n.node_type == node_type]
    
    # Filter by search term
    if search:
        search_lower = search.lower()
        nodes = [n for n in nodes if 
                search_lower in n.display_name.lower() or 
                search_lower in n.name.lower()]
    
    # Apply pagination
    total = len(nodes)
    nodes = nodes[offset:offset + limit]
    
    return nodes

@router.get("/nodes/{node_id}", response_model=NodeData)
async def get_node(node_id: str):
    """Get specific node details"""
    if node_id not in data_store.nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    return data_store.nodes[node_id]

@router.get("/nodes/{node_id}/details")
async def get_node_details(node_id: str):
    """Get detailed node information for context pane"""
    if node_id not in data_store.nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = data_store.nodes[node_id]
    
    # Simulate fetching additional details (could be expensive queries)
    await asyncio.sleep(0.1)  # Simulate async database call
    
    # Get connected nodes
    connected_nodes = []
    for conn_id in node.connections:
        if conn_id in data_store.nodes:
            conn_node = data_store.nodes[conn_id]
            connected_nodes.append({
                "id": conn_node.id,
                "name": conn_node.display_name,
                "type": conn_node.node_type
            })
    
    # Additional computed metrics
    details = {
        "basic_info": {
            "id": node.id,
            "name": node.display_name,
            "type": node.node_type.value,
            "environment": node.metadata.get("environment", "unknown"),
            "team": node.metadata.get("team", "unknown")
        },
        "properties": node.properties,
        "connections": {
            "count": len(connected_nodes),
            "nodes": connected_nodes
        },
        "metrics": {
            "last_updated": datetime.now().isoformat(),
            "health_score": 95,  # Mock health score
            "alert_count": 2     # Mock alert count
        },
        "metadata": node.metadata
    }
    
    return details

@router.get("/edges", response_model=List[EdgeData])
async def get_edges(
    source: Optional[str] = None,
    target: Optional[str] = None,
    edge_type: Optional[str] = None
):
    """Get edges with optional filtering"""
    edges = list(data_store.edges.values())
    
    if source:
        edges = [e for e in edges if e.source == source]
    
    if target:
        edges = [e for e in edges if e.target == target]
    
    if edge_type:
        edges = [e for e in edges if e.edge_type == edge_type]
    
    return edges

@router.get("/graph", response_model=GraphData)
async def get_graph(
    center_node: Optional[str] = Query(None, description="Center node for LOD"),
    depth: int = Query(2, ge=1, le=5, description="Depth for LOD expansion"),
    filter_search: Optional[str] = Query(None, description="Filter nodes by search term")
):
    """Get complete graph or level-of-detail subgraph"""
    
    if center_node and center_node in data_store.nodes:
        # Level-of-detail: get nodes within specified depth
        included_nodes = set()
        queue = [(center_node, 0)]
        
        while queue:
            node_id, current_depth = queue.pop(0)
            if current_depth > depth or node_id in included_nodes:
                continue
                
            included_nodes.add(node_id)
            
            # Add connected nodes to queue
            if node_id in data_store.nodes:
                for conn_id in data_store.nodes[node_id].connections:
                    if conn_id not in included_nodes:
                        queue.append((conn_id, current_depth + 1))
        
        nodes = [data_store.nodes[nid] for nid in included_nodes if nid in data_store.nodes]
        edges = [e for e in data_store.edges.values() 
                if e.source in included_nodes and e.target in included_nodes]
    else:
        # Full graph
        nodes = list(data_store.nodes.values())
        edges = list(data_store.edges.values())
    
    # Apply search filter
    if filter_search:
        search_lower = filter_search.lower()
        filtered_node_ids = set()
        for node in nodes:
            if (search_lower in node.display_name.lower() or 
                search_lower in node.name.lower()):
                filtered_node_ids.add(node.id)
        
        nodes = [n for n in nodes if n.id in filtered_node_ids]
        edges = [e for e in edges if e.source in filtered_node_ids and e.target in filtered_node_ids]
    
    return GraphData(
        nodes=nodes,
        edges=edges,
        metadata={
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "generated_at": datetime.now().isoformat(),
            "center_node": center_node,
            "depth": depth if center_node else None
        }
    )

@router.post("/nodes/{node_id}/expand")
async def expand_node(node_id: str, depth: int = Query(1, ge=1, le=3)):
    """Lazy load expansion of a specific node (for double-click)"""
    if node_id not in data_store.nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Simulate expensive data loading
    await asyncio.sleep(0.2)
    
    # Get additional connected nodes that weren't loaded before
    base_node = data_store.nodes[node_id]
    expanded_nodes = []
    expanded_edges = []
    
    # In a real implementation, this would query for additional
    # related nodes from the database
    for conn_id in base_node.connections:
        if conn_id in data_store.nodes:
            conn_node = data_store.nodes[conn_id]
            expanded_nodes.append(conn_node)
            
            # Find edges
            for edge in data_store.edges.values():
                if ((edge.source == node_id and edge.target == conn_id) or
                    (edge.source == conn_id and edge.target == node_id)):
                    expanded_edges.append(edge)
    
    return {
        "expanded_from": node_id,
        "nodes": expanded_nodes,
        "edges": expanded_edges,
        "metadata": {
            "expansion_depth": depth,
            "loaded_at": datetime.now().isoformat()
        }
    }

@router.get("/search")
async def search_nodes(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100)
):
    """Search nodes for autocomplete/filtering"""
    query_lower = q.lower()
    
    results = []
    for node in data_store.nodes.values():
        score = 0
        
        # Exact match gets highest score
        if query_lower == node.display_name.lower():
            score = 100
        elif query_lower == node.name.lower():
            score = 95
        # Starts with match
        elif node.display_name.lower().startswith(query_lower):
            score = 80
        elif node.name.lower().startswith(query_lower):
            score = 75
        # Contains match
        elif query_lower in node.display_name.lower():
            score = 60
        elif query_lower in node.name.lower():
            score = 55
        # Property matches
        else:
            for key, value in node.properties.items():
                if query_lower in str(value).lower():
                    score = 30
                    break
        
        if score > 0:
            results.append({
                "id": node.id,
                "display_name": node.display_name,
                "type": node.node_type.value,
                "score": score
            })
    
    # Sort by score and limit
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]