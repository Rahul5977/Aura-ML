"""
Neo4j Graph Service (Week 7)

Persistent graph database service for knowledge storage and retrieval.
Replaces in-memory graph with production-ready Neo4j database.
"""

from neo4j import AsyncGraphDatabase, AsyncDriver
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Neo4jGraphService:
    """
    Neo4j-backed knowledge graph service.
    Provides persistent storage and advanced graph operations.
    """
    
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            username: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.driver: Optional[AsyncDriver] = None
        self._username = username
        self._password = password
        logger.info(f"Initialized Neo4jGraphService (not yet connected)")
    
    async def connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self._username, self._password)
            )
            # Test connection
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            logger.info(f"✅ Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def close(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")
    
    async def initialize_schema(self):
        """Create indexes and constraints for better performance."""
        async with self.driver.session() as session:
            # Create constraints (unique labels)
            constraints = [
                "CREATE CONSTRAINT person_label IF NOT EXISTS FOR (p:PERSON) REQUIRE p.label IS UNIQUE",
                "CREATE CONSTRAINT place_label IF NOT EXISTS FOR (pl:PLACE) REQUIRE pl.label IS UNIQUE",
                "CREATE CONSTRAINT org_label IF NOT EXISTS FOR (o:ORGANIZATION) REQUIRE o.label IS UNIQUE",
                "CREATE CONSTRAINT concept_label IF NOT EXISTS FOR (c:CONCEPT) REQUIRE c.label IS UNIQUE",
                "CREATE CONSTRAINT date_label IF NOT EXISTS FOR (d:DATE) REQUIRE d.label IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    await session.run(constraint)
                    logger.info(f"Created constraint: {constraint.split()[2]}")
                except Exception as e:
                    logger.debug(f"Constraint already exists or failed: {e}")
            
            # Create indexes for faster lookups
            indexes = [
                "CREATE INDEX conversation_idx IF NOT EXISTS FOR (n) ON (n.conversation_id)",
                "CREATE INDEX timestamp_idx IF NOT EXISTS FOR (n) ON (n.created_at)"
            ]
            
            for index in indexes:
                try:
                    await session.run(index)
                    logger.info(f"Created index: {index.split()[2]}")
                except Exception as e:
                    logger.debug(f"Index already exists or failed: {e}")
    
    async def add_entity_node(
        self,
        node_type: str,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Add or update an entity node in Neo4j.
        
        Args:
            node_type: Type of entity (PERSON, PLACE, ORGANIZATION, etc.)
            label: Entity label/name
            properties: Additional properties
            conversation_id: Optional conversation ID
            
        Returns:
            Neo4j internal node ID
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected")
        
        async with self.driver.session() as session:
            props = properties or {}
            props["conversation_id"] = conversation_id
            
            query = f"""
            MERGE (n:{node_type} {{label: $label}})
            ON CREATE SET n += $properties, 
                         n.created_at = timestamp(),
                         n.occurrence_count = 1
            ON MATCH SET n.last_seen = timestamp(), 
                        n.occurrence_count = coalesce(n.occurrence_count, 0) + 1
            RETURN elementId(n) as node_id
            """
            
            result = await session.run(
                query,
                label=label,
                properties=props
            )
            record = await result.single()
            return record["node_id"]
    
    async def add_relationship(
        self,
        from_label: str,
        from_type: str,
        to_label: str,
        to_type: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create or update relationship between nodes.
        
        Args:
            from_label: Source node label
            from_type: Source node type
            to_label: Target node label
            to_type: Target node type
            rel_type: Relationship type (KNOWS, LOCATED_AT, FEELS, etc.)
            properties: Relationship properties
            
        Returns:
            Neo4j internal relationship ID
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected")
        
        async with self.driver.session() as session:
            props = properties or {}
            
            query = f"""
            MATCH (a:{from_type} {{label: $from_label}})
            MATCH (b:{to_type} {{label: $to_label}})
            MERGE (a)-[r:{rel_type}]->(b)
            ON CREATE SET r += $properties, 
                         r.created_at = timestamp(),
                         r.strength = 1
            ON MATCH SET r.last_updated = timestamp(),
                        r.strength = coalesce(r.strength, 0) + 1
            RETURN elementId(r) as rel_id
            """
            
            result = await session.run(
                query,
                from_label=from_label,
                to_label=to_label,
                properties=props
            )
            record = await result.single()
            return record["rel_id"]
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        depth: int = 2,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Retrieve graph context for a conversation.
        
        Args:
            conversation_id: Conversation ID to query
            depth: Graph traversal depth
            limit: Maximum nodes to return
            
        Returns:
            Dictionary with nodes and relationships
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected")
        
        async with self.driver.session() as session:
            query = """
            MATCH (n)
            WHERE n.conversation_id = $conv_id
            OPTIONAL MATCH (n)-[r*1..%d]-(m)
            WITH collect(DISTINCT n) + collect(DISTINCT m) as all_nodes,
                 collect(DISTINCT r) as all_rels
            RETURN [node IN all_nodes | properties(node)] as nodes,
                   [rel IN all_rels | properties(rel)] as relationships
            LIMIT $limit
            """ % depth
            
            result = await session.run(
                query,
                conv_id=conversation_id,
                limit=limit
            )
            record = await result.single()
            
            if not record:
                return {"nodes": [], "relationships": []}
            
            return {
                "nodes": record["nodes"] or [],
                "relationships": record["relationships"] or [],
                "conversation_id": conversation_id
            }
    
    async def get_entity_connections(
        self,
        entity_label: str,
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get all connections for an entity.
        
        Args:
            entity_label: Entity to query
            entity_type: Optional type filter
            limit: Maximum results
            
        Returns:
            List of connected entities with relationships
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected")
        
        async with self.driver.session() as session:
            type_clause = f":{entity_type}" if entity_type else ""
            
            query = f"""
            MATCH (a{type_clause} {{label: $label}})-[r]-(b)
            RETURN type(r) as relationship,
                   b.label as connected_entity,
                   labels(b)[0] as entity_type,
                   r.strength as strength,
                   r.created_at as created_at
            ORDER BY r.strength DESC
            LIMIT $limit
            """
            
            result = await session.run(
                query,
                label=entity_label,
                limit=limit
            )
            
            connections = []
            async for record in result:
                connections.append({
                    "relationship": record["relationship"],
                    "connected_entity": record["connected_entity"],
                    "entity_type": record["entity_type"],
                    "strength": record["strength"],
                    "created_at": record["created_at"]
                })
            
            return connections
    
    async def get_graph_summary(self) -> Dict[str, Any]:
        """
        Get overall graph statistics.
        
        Returns:
            Summary with counts and statistics
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected")
        
        async with self.driver.session() as session:
            query = """
            MATCH (n)
            WITH labels(n)[0] as node_type, count(n) as count
            RETURN collect({type: node_type, count: count}) as node_counts
            """
            result = await session.run(query)
            record = await result.single()
            node_counts = record["node_counts"] if record else []
            
            # Count relationships
            query = """
            MATCH ()-[r]->()
            WITH type(r) as rel_type, count(r) as count
            RETURN collect({type: rel_type, count: count}) as rel_counts
            """
            result = await session.run(query)
            record = await result.single()
            rel_counts = record["rel_counts"] if record else []
            
            # Total counts
            total_nodes = sum(item["count"] for item in node_counts)
            total_rels = sum(item["count"] for item in rel_counts)
            
            return {
                "total_nodes": total_nodes,
                "total_relationships": total_rels,
                "node_types": node_counts,
                "relationship_types": rel_counts,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def clear_conversation(self, conversation_id: str):
        """
        Delete all nodes and relationships for a conversation.
        
        Args:
            conversation_id: Conversation to clear
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected")
        
        async with self.driver.session() as session:
            query = """
            MATCH (n {conversation_id: $conv_id})
            DETACH DELETE n
            """
            await session.run(query, conv_id=conversation_id)
            logger.info(f"Cleared conversation: {conversation_id}")
    
    async def export_graph(self, format: str = "cypher") -> str:
        """
        Export entire graph in specified format.
        
        Args:
            format: Export format (cypher, json, graphml)
            
        Returns:
            Exported graph data
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected")
        
        if format == "cypher":
            # Export as Cypher statements
            async with self.driver.session() as session:
                query = """
                MATCH (n)
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN n, collect(r) as rels, collect(m) as targets
                """
                result = await session.run(query)
                
                cypher_statements = []
                async for record in result:
                    node = record["n"]
                    # Generate CREATE statement for node
                    labels = ":".join(node.labels)
                    props = dict(node.items())
                    cypher_statements.append(
                        f"CREATE (:{labels} {props});"
                    )
                
                return "\n".join(cypher_statements)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global singleton (initialized in main.py)
neo4j_graph_service: Optional[Neo4jGraphService] = None


async def initialize_neo4j_service(uri: str, username: str, password: str):
    """
    Initialize and connect to Neo4j.
    Should be called during application startup.
    """
    global neo4j_graph_service
    
    neo4j_graph_service = Neo4jGraphService(uri, username, password)
    await neo4j_graph_service.connect()
    await neo4j_graph_service.initialize_schema()
    
    logger.info("✅ Neo4j graph service initialized")


async def shutdown_neo4j_service():
    """
    Close Neo4j connection.
    Should be called during application shutdown.
    """
    global neo4j_graph_service
    
    if neo4j_graph_service:
        await neo4j_graph_service.close()
        logger.info("Neo4j connection closed")
