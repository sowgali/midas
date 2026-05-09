import { useCallback, useMemo, type JSX } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Background,
  Controls,
  ReactFlow,
  type Edge,
  type NodeMouseHandler,
  type EdgeMouseHandler,
} from "@xyflow/react";
import dagre from "dagre";

import "@xyflow/react/dist/style.css";

import { fetchGraph, type GraphFilters } from "../api";
import type { EntityDto, GraphEdgeDto, GraphResponse } from "../types";
import { formatUsd } from "../format";
import EntityNode, { type EntityFlowNode } from "./EntityNode";
import styles from "./GraphView.module.css";
import type { Selection } from "../App";

const NODE_WIDTH = 200;
const NODE_HEIGHT = 44;

const NODE_TYPES = { entity: EntityNode };

interface GraphViewProps {
  filters: GraphFilters;
  onSelect: (selection: Selection) => void;
}

interface LayoutResult {
  nodes: EntityFlowNode[];
  edges: Edge[];
}

function edgeKey(from_id: string, to_id: string): string {
  return `${from_id}->${to_id}`;
}

function edgeWidth(amount: number | null): number {
  if (amount === null || amount <= 0) return 1;
  // log10($1) -> 0, log10($1T) -> 12; clamp to 1..6 px.
  const ratio = Math.log10(amount + 1) / Math.log10(1e12);
  const clamped = Math.min(1, Math.max(0, ratio));
  return 1 + clamped * 5;
}

function buildLayout(graph: GraphResponse): LayoutResult {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "LR", nodesep: 40, ranksep: 80 });

  for (const entity of graph.nodes) {
    g.setNode(entity.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const edge of graph.edges) {
    if (g.hasNode(edge.from_id) && g.hasNode(edge.to_id)) {
      g.setEdge(edge.from_id, edge.to_id);
    }
  }

  dagre.layout(g);

  const flowNodes: EntityFlowNode[] = graph.nodes.map((entity) => {
    const pos = g.node(entity.id);
    return {
      id: entity.id,
      type: "entity",
      // dagre returns the node's center; xyflow expects the top-left corner.
      position: {
        x: (pos?.x ?? 0) - NODE_WIDTH / 2,
        y: (pos?.y ?? 0) - NODE_HEIGHT / 2,
      },
      data: { entity },
    };
  });

  const flowEdges: Edge[] = graph.edges.map((edge) => {
    const amountLabel = formatUsd(edge.total_amount_usd);
    const label =
      edge.deal_count > 1 ? `${amountLabel} · ${edge.deal_count}` : amountLabel;
    const dashed = edge.total_amount_usd === null;
    return {
      id: edgeKey(edge.from_id, edge.to_id),
      source: edge.from_id,
      target: edge.to_id,
      label,
      labelStyle: { fontSize: 11, fill: "#374151" },
      labelBgPadding: [4, 2],
      labelBgBorderRadius: 3,
      labelBgStyle: { fill: "#ffffff", fillOpacity: 0.92 },
      style: {
        stroke: dashed ? "#94a3b8" : "#475569",
        strokeWidth: edgeWidth(edge.total_amount_usd),
        strokeDasharray: dashed ? "4 3" : undefined,
      },
      data: { edge },
    };
  });

  return { nodes: flowNodes, edges: flowEdges };
}

export default function GraphView({
  filters,
  onSelect,
}: GraphViewProps): JSX.Element {
  const { data, isLoading, isError, error } = useQuery<GraphResponse, Error>({
    queryKey: ["graph", filters.sector ?? "", filters.as_of ?? ""],
    queryFn: () => fetchGraph(filters),
  });

  const layout = useMemo<LayoutResult>(() => {
    if (!data) return { nodes: [], edges: [] };
    return buildLayout(data);
  }, [data]);

  const entityById = useMemo<Map<string, EntityDto>>(() => {
    const m = new Map<string, EntityDto>();
    for (const entity of data?.nodes ?? []) m.set(entity.id, entity);
    return m;
  }, [data]);

  const edgeByKey = useMemo<Map<string, GraphEdgeDto>>(() => {
    const m = new Map<string, GraphEdgeDto>();
    for (const e of data?.edges ?? []) m.set(edgeKey(e.from_id, e.to_id), e);
    return m;
  }, [data]);

  const handleNodeClick = useCallback<NodeMouseHandler>(
    (_event, node) => {
      const entity = entityById.get(node.id);
      if (entity) onSelect({ kind: "node", entity });
    },
    [entityById, onSelect],
  );

  const handleEdgeClick = useCallback<EdgeMouseHandler<Edge>>(
    (_event, edge) => {
      const graphEdge = edgeByKey.get(edge.id);
      if (!graphEdge) return;
      const from = entityById.get(graphEdge.from_id);
      const to = entityById.get(graphEdge.to_id);
      if (!from || !to) return;
      onSelect({ kind: "edge", edge: graphEdge, from, to });
    },
    [edgeByKey, entityById, onSelect],
  );

  const handlePaneClick = useCallback(() => {
    onSelect(null);
  }, [onSelect]);

  if (isLoading) {
    return <div className={styles.status}>Loading graph…</div>;
  }
  if (isError) {
    return (
      <div className={styles.statusError}>
        Failed to load graph: {error.message}
      </div>
    );
  }
  if (!data || data.nodes.length === 0) {
    return <div className={styles.status}>No entities match these filters.</div>;
  }

  return (
    <div className={styles.canvas}>
      <ReactFlow
        nodes={layout.nodes}
        edges={layout.edges}
        nodeTypes={NODE_TYPES}
        onNodeClick={handleNodeClick}
        onEdgeClick={handleEdgeClick}
        onPaneClick={handlePaneClick}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={24} size={1} color="#e5e7eb" />
        <Controls position="bottom-right" showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
