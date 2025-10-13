"""
Dynamic Knowledge Graph Service

Builds and maintains a knowledge graph from conversational data.
Structures entities, relationships, and emotional context into graph format.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
import asyncio
import json

logger = logging.getLogger(__name__)


class GraphNode:
    """Represents a node in the knowledge graph."""
    
    def __init__(
        self,
        node_id: str,
        node_type: str,
        label: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.label = label
        self.properties = properties or {}
        self.created_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "id": self.node_id,
            "type": self.node_type,
            "label": self.label,
            "properties": self.properties,
            "created_at": self.created_at
        }


class GraphRelationship:
    """Represents a relationship (edge) in the knowledge graph."""
    
    def __init__(
        self,
        rel_id: str,
        rel_type: str,
        from_node: str,
        to_node: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        self.rel_id = rel_id
        self.rel_type = rel_type
        self.from_node = from_node
        self.to_node = to_node
        self.properties = properties or {}
        self.created_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation."""
        return {
            "id": self.rel_id,
            "type": self.rel_type,
            "from": self.from_node,
            "to": self.to_node,
            "properties": self.properties,
            "created_at": self.created_at
        }


class KnowledgeGraphService:
    """
    Service for building and maintaining a dynamic knowledge graph.
    Structures conversational data into nodes and relationships.
    """
    
    def __init__(self):
        """Initialize knowledge graph service."""
        # In-memory storage (would use Neo4j or similar in production)
        self.nodes: Dict[str, GraphNode] = {}
        self.relationships: Dict[str, GraphRelationship] = {}
        self.node_counter = 0
        self.rel_counter = 0
        
        logger.info("Initialized KnowledgeGraphService")
    
    def _generate_node_id(self, node_type: str, label: str) -> str:
        """Generate unique node ID."""
        # Use label-based ID for deduplication
        safe_label = label.lower().replace(" ", "_")
        return f"{node_type}_{safe_label}"
    
    def _generate_rel_id(self) -> str:
        """Generate unique relationship ID."""
        self.rel_counter += 1
        return f"rel_{self.rel_counter}"
    
    async def add_entity_nodes(
        self,
        entities: Dict[str, List[Dict[str, Any]]],
        conversation_id: str
    ) -> List[GraphNode]:
        """
        Add entity nodes from NER results to the graph.
        
        Args:
            entities: Entity dictionary from NER service
            conversation_id: ID of the conversation
            
        Returns:
            List of created/updated nodes
        """
        created_nodes = []
        
        # Map entity categories to node types
        category_mapping = {
            "people": "PERSON",
            "places": "PLACE",
            "organizations": "ORGANIZATION",
            "concepts": "CONCEPT",
            "dates": "DATE"
        }
        
        for category, entity_list in entities.items():
            if category not in category_mapping:
                continue
            
            node_type = category_mapping[category]
            
            for entity in entity_list:
                label = entity["text"]
                node_id = self._generate_node_id(node_type, label)
                
                # Check if node already exists
                if node_id in self.nodes:
                    # Update properties
                    node = self.nodes[node_id]
                    if "occurrences" not in node.properties:
                        node.properties["occurrences"] = []
                    node.properties["occurrences"].append({
                        "conversation_id": conversation_id,
                        "position": {"start": entity["start"], "end": entity["end"]},
                        "timestamp": datetime.utcnow().isoformat()
                    })
                else:
                    # Create new node
                    node = GraphNode(
                        node_id=node_id,
                        node_type=node_type,
                        label=label,
                        properties={
                            "original_label": entity.get("label"),
                            "occurrences": [{
                                "conversation_id": conversation_id,
                                "position": {"start": entity["start"], "end": entity["end"]},
                                "timestamp": datetime.utcnow().isoformat()
                            }]
                        }
                    )
                    self.nodes[node_id] = node
                
                created_nodes.append(node)
        
        logger.info(f"Added/updated {len(created_nodes)} entity nodes")
        return created_nodes
    
    async def add_emotional_relationships(
        self,
        emotional_analysis: Dict[str, Any],
        subject_node_id: str,
        conversation_id: str
    ) -> List[GraphRelationship]:
        """
        Add emotional relationships from COMET analysis to the graph.
        
        Args:
            emotional_analysis: Output from COMET service
            subject_node_id: ID of the subject (speaker) node
            conversation_id: ID of the conversation
            
        Returns:
            List of created relationships
        """
        created_relationships = []
        
        # Create emotion nodes and relationships
        for emotion in emotional_analysis.get("subject_emotions", []):
            emotion_node_id = self._generate_node_id("EMOTION", emotion)
            
            # Create emotion node if it doesn't exist
            if emotion_node_id not in self.nodes:
                emotion_node = GraphNode(
                    node_id=emotion_node_id,
                    node_type="EMOTION",
                    label=emotion,
                    properties={"emotion_type": "feeling"}
                )
                self.nodes[emotion_node_id] = emotion_node
            
            # Create FEELS relationship
            rel = GraphRelationship(
                rel_id=self._generate_rel_id(),
                rel_type="FEELS",
                from_node=subject_node_id,
                to_node=emotion_node_id,
                properties={
                    "conversation_id": conversation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            self.relationships[rel.rel_id] = rel
            created_relationships.append(rel)
        
        # Create want/desire relationships
        for want in emotional_analysis.get("subject_wants", []):
            want_node_id = self._generate_node_id("DESIRE", want)
            
            if want_node_id not in self.nodes:
                want_node = GraphNode(
                    node_id=want_node_id,
                    node_type="DESIRE",
                    label=want,
                    properties={"desire_type": "want"}
                )
                self.nodes[want_node_id] = want_node
            
            # Create WANTS relationship
            rel = GraphRelationship(
                rel_id=self._generate_rel_id(),
                rel_type="WANTS",
                from_node=subject_node_id,
                to_node=want_node_id,
                properties={
                    "conversation_id": conversation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            self.relationships[rel.rel_id] = rel
            created_relationships.append(rel)
        
        logger.info(f"Added {len(created_relationships)} emotional relationships")
        return created_relationships
    
    async def add_conversation_context(
        self,
        conversation_id: str,
        text: str,
        speaker_id: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> GraphNode:
        """
        Add conversation context as a node.
        
        Args:
            conversation_id: ID of the conversation
            text: Conversation text
            speaker_id: Optional speaker identifier
            timestamp: Optional timestamp
            
        Returns:
            Created conversation node
        """
        node_id = f"CONV_{conversation_id}"
        
        if node_id not in self.nodes:
            node = GraphNode(
                node_id=node_id,
                node_type="CONVERSATION",
                label=f"Conversation {conversation_id}",
                properties={
                    "conversation_id": conversation_id,
                    "text": text[:500],  # Store first 500 chars
                    "speaker_id": speaker_id,
                    "timestamp": timestamp or datetime.utcnow().isoformat()
                }
            )
            self.nodes[node_id] = node
        else:
            node = self.nodes[node_id]
            # Update with new information
            if "messages" not in node.properties:
                node.properties["messages"] = []
            node.properties["messages"].append({
                "text": text[:200],
                "speaker_id": speaker_id,
                "timestamp": timestamp or datetime.utcnow().isoformat()
            })
        
        return node
    
    async def query_related_entities(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 1
    ) -> Dict[str, Any]:
        """
        Query entities related to a given node.
        
        Args:
            node_id: Starting node ID
            relationship_types: Filter by relationship types
            max_depth: Maximum traversal depth
            
        Returns:
            Dictionary with related nodes and relationships
        """
        if node_id not in self.nodes:
            return {"nodes": [], "relationships": []}
        
        related_nodes = []
        related_rels = []
        
        # Find directly connected relationships
        for rel_id, rel in self.relationships.items():
            if relationship_types and rel.rel_type not in relationship_types:
                continue
            
            if rel.from_node == node_id:
                related_rels.append(rel)
                if rel.to_node in self.nodes:
                    related_nodes.append(self.nodes[rel.to_node])
            elif rel.to_node == node_id:
                related_rels.append(rel)
                if rel.from_node in self.nodes:
                    related_nodes.append(self.nodes[rel.from_node])
        
        return {
            "nodes": [node.to_dict() for node in related_nodes],
            "relationships": [rel.to_dict() for rel in related_rels]
        }
    
    async def get_graph_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the knowledge graph.
        
        Returns:
            Summary with node/relationship counts by type
        """
        node_types = {}
        rel_types = {}
        
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        
        for rel in self.relationships.values():
            rel_types[rel.rel_type] = rel_types.get(rel.rel_type, 0) + 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_relationships": len(self.relationships),
            "nodes_by_type": node_types,
            "relationships_by_type": rel_types
        }
    
    async def export_graph(self, format: str = "json") -> str:
        """
        Export the knowledge graph in various formats.
        
        Args:
            format: Export format (json, cypher, graphml)
            
        Returns:
            Serialized graph data
        """
        if format == "json":
            graph_data = {
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "relationships": [rel.to_dict() for rel in self.relationships.values()]
            }
            return json.dumps(graph_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the graph."""
        self.nodes.clear()
        self.relationships.clear()
        self.node_counter = 0
        self.rel_counter = 0
        logger.info("Cleared knowledge graph")


# Global singleton instance
knowledge_graph_service = KnowledgeGraphService()
