import { useMemo } from "react";
import {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from "@dagrejs/dagre";
import type { EvidenceEdge, EvidenceNode } from "../../api/types";

const NODE_W = 200;
const NODE_H = 64;

function layout(nodes: EvidenceNode[], edges: EvidenceEdge[]) {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", nodesep: 32, ranksep: 80 });
  g.setDefaultEdgeLabel(() => ({}));
  nodes.forEach((n) => g.setNode(n.node_id, { width: NODE_W, height: NODE_H }));
  edges.forEach((e) => g.setEdge(e.source_node_id, e.target_node_id));
  dagre.layout(g);
  return new Map(nodes.map((n) => [n.node_id, g.node(n.node_id)]));
}

// One graph "card" node component -- confidence shown as a small inline
// bar rather than color alone (color still carries the root highlight,
// but the number is always readable text too).
function GraphNode({ data }: { data: { node: EvidenceNode; isRoot: boolean } }) {
  const { node, isRoot } = data;
  return (
    <div
      className="rounded-lg border px-3 py-2 shadow-sm"
      style={{
        width: NODE_W,
        background: isRoot ? "color-mix(in srgb, var(--series-structured) 12%, var(--surface-1))" : "var(--surface-1)",
        borderColor: isRoot ? "var(--series-structured)" : "var(--border)",
        borderWidth: isRoot ? 2 : 1,
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: "var(--baseline)" }} />
      <Handle type="source" position={Position.Right} style={{ background: "var(--baseline)" }} />
      <div className="truncate text-xs font-semibold" title={node.node_type}>
        {node.node_type.replace(":", " · ")}
      </div>
      <div className="mt-1 flex items-center gap-2">
        <div className="h-1.5 flex-1 rounded-full" style={{ background: "var(--gridline)" }}>
          <div
            className="h-1.5 rounded-full"
            style={{
              width: `${Math.round(node.confidence * 100)}%`,
              background: isRoot ? "var(--series-structured)" : "var(--baseline)",
            }}
          />
        </div>
        <span className="text-xs text-muted" style={{ fontVariantNumeric: "tabular-nums" }}>
          {node.confidence.toFixed(2)}
        </span>
      </div>
      {isRoot && (
        <div className="mt-1 text-xs font-medium" style={{ color: "var(--series-structured)" }}>
          Identified root cause
        </div>
      )}
    </div>
  );
}

const nodeTypes = { evidence: GraphNode };

export default function EvidenceGraph({
  nodes,
  edges,
  highlightNodeId,
}: {
  nodes: EvidenceNode[];
  edges: EvidenceEdge[];
  highlightNodeId: string | null;
}) {
  const { flowNodes, flowEdges } = useMemo(() => {
    if (nodes.length === 0) return { flowNodes: [], flowEdges: [] };
    const positions = layout(nodes, edges);
    const flowNodes: Node[] = nodes.map((n) => {
      const pos = positions.get(n.node_id) ?? { x: 0, y: 0 };
      return {
        id: n.node_id,
        type: "evidence",
        position: { x: pos.x - NODE_W / 2, y: pos.y - NODE_H / 2 },
        data: { node: n, isRoot: n.node_id === highlightNodeId },
      };
    });
    const flowEdges: Edge[] = edges.map((e) => ({
      id: `${e.source_node_id}-${e.target_node_id}`,
      source: e.source_node_id,
      target: e.target_node_id,
      label: e.confidence.toFixed(2),
      labelStyle: { fill: "var(--text-muted)", fontSize: 11 },
      style: { stroke: "var(--baseline)", strokeWidth: 1.5, opacity: Math.max(0.35, e.confidence) },
      animated: false,
    }));
    return { flowNodes, flowEdges };
  }, [nodes, edges, highlightNodeId]);

  if (nodes.length === 0) {
    return <p className="text-secondary">No evidence nodes.</p>;
  }

  const height = Math.max(280, Math.min(520, 140 + nodes.length * 40));

  return (
    <div className="surface overflow-hidden rounded-lg" style={{ height }}>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        nodeTypes={nodeTypes}
        fitView
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={true}
      >
        <Background color="var(--gridline)" gap={20} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
