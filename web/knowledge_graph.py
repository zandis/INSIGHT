"""
Knowledge Graph Visualization Module.

Provides graph-based visualization of research relationships
including genes, variants, publications, and concepts.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    type: str  # 'gene', 'variant', 'publication', 'concept', 'disease'
    data: Dict[str, Any] = field(default_factory=dict)
    size: float = 1.0
    color: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'label': self.label,
            'type': self.type,
            'data': self.data,
            'size': self.size,
            'color': self.color or self._default_color()
        }

    def _default_color(self) -> str:
        """Get default color based on node type."""
        colors = {
            'gene': '#4CAF50',
            'variant': '#FF9800',
            'publication': '#2196F3',
            'concept': '#9C27B0',
            'disease': '#F44336',
            'pathway': '#00BCD4',
            'drug': '#E91E63'
        }
        return colors.get(self.type, '#607D8B')


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source: str
    target: str
    relationship: str
    weight: float = 1.0
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'source': self.source,
            'target': self.target,
            'relationship': self.relationship,
            'weight': self.weight,
            'data': self.data
        }


class KnowledgeGraph:
    """
    Knowledge graph for representing and visualizing research relationships.
    """

    def __init__(self):
        """Initialize the knowledge graph."""
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)

    def add_node(
        self,
        node_id: str,
        label: str,
        node_type: str,
        data: Optional[Dict] = None,
        size: float = 1.0,
        color: Optional[str] = None
    ) -> GraphNode:
        """
        Add a node to the graph.

        Args:
            node_id: Unique identifier for the node
            label: Display label
            node_type: Type of node (gene, variant, etc.)
            data: Additional node data
            size: Node size for visualization
            color: Node color (optional)

        Returns:
            The created or existing GraphNode
        """
        if node_id not in self.nodes:
            self.nodes[node_id] = GraphNode(
                id=node_id,
                label=label,
                type=node_type,
                data=data or {},
                size=size,
                color=color
            )
        return self.nodes[node_id]

    def add_edge(
        self,
        source: str,
        target: str,
        relationship: str,
        weight: float = 1.0,
        data: Optional[Dict] = None
    ) -> GraphEdge:
        """
        Add an edge to the graph.

        Args:
            source: Source node ID
            target: Target node ID
            relationship: Type of relationship
            weight: Edge weight
            data: Additional edge data

        Returns:
            The created GraphEdge
        """
        edge = GraphEdge(
            source=source,
            target=target,
            relationship=relationship,
            weight=weight,
            data=data or {}
        )
        self.edges.append(edge)
        self._adjacency[source].add(target)
        self._adjacency[target].add(source)
        return edge

    def get_neighbors(self, node_id: str) -> Set[str]:
        """Get all neighbors of a node."""
        return self._adjacency.get(node_id, set())

    def get_subgraph(self, node_id: str, depth: int = 1) -> 'KnowledgeGraph':
        """
        Get a subgraph centered on a node.

        Args:
            node_id: Center node ID
            depth: How many hops to include

        Returns:
            New KnowledgeGraph with the subgraph
        """
        if node_id not in self.nodes:
            return KnowledgeGraph()

        visited = {node_id}
        current_level = {node_id}

        for _ in range(depth):
            next_level = set()
            for nid in current_level:
                neighbors = self.get_neighbors(nid)
                next_level.update(neighbors - visited)
            visited.update(next_level)
            current_level = next_level

        subgraph = KnowledgeGraph()

        for nid in visited:
            if nid in self.nodes:
                node = self.nodes[nid]
                subgraph.add_node(
                    node.id, node.label, node.type,
                    node.data, node.size, node.color
                )

        for edge in self.edges:
            if edge.source in visited and edge.target in visited:
                subgraph.add_edge(
                    edge.source, edge.target, edge.relationship,
                    edge.weight, edge.data
                )

        return subgraph

    def to_d3_format(self) -> Dict:
        """
        Convert graph to D3.js compatible format.

        Returns:
            Dictionary with 'nodes' and 'links' for D3 visualization
        """
        return {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'links': [edge.to_dict() for edge in self.edges]
        }

    def to_cytoscape_format(self) -> List[Dict]:
        """
        Convert graph to Cytoscape.js compatible format.

        Returns:
            List of elements for Cytoscape visualization
        """
        elements = []

        for node in self.nodes.values():
            elements.append({
                'data': {
                    'id': node.id,
                    'label': node.label,
                    'type': node.type,
                    **node.data
                },
                'classes': node.type
            })

        for edge in self.edges:
            elements.append({
                'data': {
                    'source': edge.source,
                    'target': edge.target,
                    'relationship': edge.relationship,
                    'weight': edge.weight
                }
            })

        return elements

    def to_json(self) -> str:
        """Serialize graph to JSON string."""
        return json.dumps(self.to_d3_format(), indent=2)

    @classmethod
    def from_research_results(cls, results: Dict) -> 'KnowledgeGraph':
        """
        Build a knowledge graph from research results.

        Args:
            results: Research results dictionary

        Returns:
            KnowledgeGraph populated from results
        """
        graph = cls()

        # Process gene data
        for gene in results.get('genes', []):
            gene_id = f"gene_{gene.get('symbol', gene.get('_id', 'unknown'))}"
            graph.add_node(
                node_id=gene_id,
                label=gene.get('symbol', 'Unknown'),
                node_type='gene',
                data={
                    'name': gene.get('name', ''),
                    'entrez_id': gene.get('entrezgene', ''),
                    'summary': gene.get('summary', '')[:200] if gene.get('summary') else ''
                },
                size=1.5
            )

            # Add pathway relationships
            pathways = gene.get('pathway', {})
            if isinstance(pathways, dict):
                for pathway_source, pathway_list in pathways.items():
                    if isinstance(pathway_list, list):
                        for pathway in pathway_list[:5]:
                            pathway_id = f"pathway_{pathway.get('id', pathway.get('name', ''))}"
                            graph.add_node(
                                node_id=pathway_id,
                                label=pathway.get('name', pathway_id),
                                node_type='pathway',
                                size=1.0
                            )
                            graph.add_edge(gene_id, pathway_id, 'involved_in')

        # Process variant data
        for variant in results.get('variants', []):
            variant_id = f"variant_{variant.get('_id', variant.get('rsid', 'unknown'))}"
            graph.add_node(
                node_id=variant_id,
                label=variant.get('rsid', variant.get('_id', 'Unknown')),
                node_type='variant',
                data={
                    'chrom': variant.get('chrom', ''),
                    'pos': variant.get('pos', ''),
                    'ref': variant.get('ref', ''),
                    'alt': variant.get('alt', '')
                },
                size=1.2
            )

            # Link to associated genes
            if variant.get('cadd', {}).get('gene'):
                gene_symbol = variant['cadd']['gene'].get('genename', '')
                if gene_symbol:
                    gene_node_id = f"gene_{gene_symbol}"
                    if gene_node_id in graph.nodes:
                        graph.add_edge(variant_id, gene_node_id, 'located_in')

        # Process publication data
        for i, pub in enumerate(results.get('publications', [])[:50]):
            pub_id = f"pub_{pub.get('pmid', i)}"
            graph.add_node(
                node_id=pub_id,
                label=pub.get('title', 'Untitled')[:50] + '...',
                node_type='publication',
                data={
                    'pmid': pub.get('pmid', ''),
                    'title': pub.get('title', ''),
                    'authors': pub.get('authors', ''),
                    'journal': pub.get('journal', '')
                },
                size=0.8
            )

            # Link to mentioned genes
            for gene_mention in pub.get('genes_mentioned', []):
                gene_node_id = f"gene_{gene_mention}"
                if gene_node_id in graph.nodes:
                    graph.add_edge(pub_id, gene_node_id, 'mentions')

        # Process disease associations
        for disease in results.get('diseases', []):
            disease_id = f"disease_{disease.get('id', disease.get('name', 'unknown'))}"
            graph.add_node(
                node_id=disease_id,
                label=disease.get('name', 'Unknown Disease'),
                node_type='disease',
                data=disease,
                size=1.3
            )

            # Link to associated genes
            for gene_symbol in disease.get('associated_genes', []):
                gene_node_id = f"gene_{gene_symbol}"
                if gene_node_id in graph.nodes:
                    graph.add_edge(disease_id, gene_node_id, 'associated_with')

        return graph

    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        node_types = defaultdict(int)
        for node in self.nodes.values():
            node_types[node.type] += 1

        edge_types = defaultdict(int)
        for edge in self.edges:
            edge_types[edge.relationship] += 1

        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': dict(node_types),
            'edge_types': dict(edge_types),
            'avg_degree': (2 * len(self.edges) / len(self.nodes)) if self.nodes else 0
        }


class KnowledgeGraphBuilder:
    """
    Builder class for constructing knowledge graphs from various data sources.
    """

    def __init__(self):
        """Initialize the builder."""
        self.graph = KnowledgeGraph()

    def add_gene(
        self,
        symbol: str,
        name: str = '',
        entrez_id: str = '',
        **kwargs
    ) -> 'KnowledgeGraphBuilder':
        """Add a gene node."""
        self.graph.add_node(
            node_id=f"gene_{symbol}",
            label=symbol,
            node_type='gene',
            data={'name': name, 'entrez_id': entrez_id, **kwargs},
            size=1.5
        )
        return self

    def add_variant(
        self,
        rsid: str,
        chrom: str = '',
        pos: str = '',
        **kwargs
    ) -> 'KnowledgeGraphBuilder':
        """Add a variant node."""
        self.graph.add_node(
            node_id=f"variant_{rsid}",
            label=rsid,
            node_type='variant',
            data={'chrom': chrom, 'pos': pos, **kwargs},
            size=1.2
        )
        return self

    def add_publication(
        self,
        pmid: str,
        title: str,
        **kwargs
    ) -> 'KnowledgeGraphBuilder':
        """Add a publication node."""
        self.graph.add_node(
            node_id=f"pub_{pmid}",
            label=title[:50] + '...' if len(title) > 50 else title,
            node_type='publication',
            data={'pmid': pmid, 'title': title, **kwargs},
            size=0.8
        )
        return self

    def add_disease(
        self,
        disease_id: str,
        name: str,
        **kwargs
    ) -> 'KnowledgeGraphBuilder':
        """Add a disease node."""
        self.graph.add_node(
            node_id=f"disease_{disease_id}",
            label=name,
            node_type='disease',
            data={'id': disease_id, 'name': name, **kwargs},
            size=1.3
        )
        return self

    def link_gene_variant(
        self,
        gene_symbol: str,
        variant_rsid: str
    ) -> 'KnowledgeGraphBuilder':
        """Link a gene to a variant."""
        self.graph.add_edge(
            f"variant_{variant_rsid}",
            f"gene_{gene_symbol}",
            'located_in'
        )
        return self

    def link_gene_disease(
        self,
        gene_symbol: str,
        disease_id: str
    ) -> 'KnowledgeGraphBuilder':
        """Link a gene to a disease."""
        self.graph.add_edge(
            f"gene_{gene_symbol}",
            f"disease_{disease_id}",
            'associated_with'
        )
        return self

    def link_publication_gene(
        self,
        pmid: str,
        gene_symbol: str
    ) -> 'KnowledgeGraphBuilder':
        """Link a publication to a gene."""
        self.graph.add_edge(
            f"pub_{pmid}",
            f"gene_{gene_symbol}",
            'mentions'
        )
        return self

    def build(self) -> KnowledgeGraph:
        """Build and return the knowledge graph."""
        return self.graph


def generate_graph_html(graph: KnowledgeGraph) -> str:
    """
    Generate standalone HTML for graph visualization.

    Args:
        graph: KnowledgeGraph to visualize

    Returns:
        HTML string with embedded D3.js visualization
    """
    graph_data = graph.to_json()

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Knowledge Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin: 0; overflow: hidden; font-family: Arial, sans-serif; }}
        svg {{ width: 100%; height: 100vh; }}
        .node circle {{ stroke: #fff; stroke-width: 2px; cursor: pointer; }}
        .node text {{ font-size: 10px; pointer-events: none; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; }}
        .tooltip {{
            position: absolute;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 8px;
            font-size: 12px;
            pointer-events: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .legend {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 4px 0;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <svg></svg>
    <div class="tooltip" style="display: none;"></div>
    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background: #4CAF50;"></div>Gene</div>
        <div class="legend-item"><div class="legend-color" style="background: #FF9800;"></div>Variant</div>
        <div class="legend-item"><div class="legend-color" style="background: #2196F3;"></div>Publication</div>
        <div class="legend-item"><div class="legend-color" style="background: #F44336;"></div>Disease</div>
        <div class="legend-item"><div class="legend-color" style="background: #00BCD4;"></div>Pathway</div>
    </div>
    <script>
        const data = {graph_data};

        const width = window.innerWidth;
        const height = window.innerHeight;

        const svg = d3.select("svg")
            .attr("viewBox", [0, 0, width, height]);

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));

        const link = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .join("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.weight));

        const node = svg.append("g")
            .selectAll("g")
            .data(data.nodes)
            .join("g")
            .attr("class", "node")
            .call(drag(simulation));

        node.append("circle")
            .attr("r", d => d.size * 10)
            .attr("fill", d => d.color);

        node.append("text")
            .text(d => d.label)
            .attr("x", d => d.size * 10 + 5)
            .attr("y", 3);

        const tooltip = d3.select(".tooltip");

        node.on("mouseover", (event, d) => {{
            tooltip.style("display", "block")
                .html(`<strong>${{d.label}}</strong><br>Type: ${{d.type}}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }})
        .on("mouseout", () => tooltip.style("display", "none"));

        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});

        function drag(simulation) {{
            return d3.drag()
                .on("start", (event, d) => {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }})
                .on("drag", (event, d) => {{
                    d.fx = event.x;
                    d.fy = event.y;
                }})
                .on("end", (event, d) => {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }});
        }}
    </script>
</body>
</html>
"""
