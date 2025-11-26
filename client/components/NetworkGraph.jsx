// client/components/NetworkGraph.jsx
import React, { useMemo, useState, useCallback } from "react";
import dynamic from "next/dynamic";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
});

/**
 * Props:
 *  - nodes: [{ id, label, type, credScore? }]
 *  - links: [{ source, target, rel }]
 *  - onNodeClick?: (node) => void
 */
export default function NetworkGraph({ nodes = [], links = [], onNodeClick }) {
  const [hoverNode, setHoverNode] = useState(null);

  // --------- helpers ---------
  const getNodeColor = (node) => {
    // Domain nodes: color by discrete credibility score
    if (node.type === "domain") {
      let s = node.credScore;

      // allow score as string or number
      if (typeof s === "string") {
        const parsed = parseFloat(s);
        s = Number.isFinite(parsed) ? parsed : null;
      }

      if (!Number.isFinite(s)) {
        // no score -> treat as neutral
        return "#f97316"; // orange
      }

      if (s < 0) {
        return "#ac0808ff"; // low credibility -> red
      }
      if (s === 0) {
        return "#f97316"; // neutral / unknown -> orange
      }
      if (s > 0) {
        return "#16a34a"; // trusted / highly trusted -> green
      }

      // fallback in case some other value sneaks in
      return "#f97316";
    }

    // Non-domain nodes: fixed colors by type
    switch (node.type) {
      case "author":
        return "#4f46e5"; // indigo
      case "subreddit":
        return "#0ea5e9"; // blue-ish
      default:
        return "#6b7280"; // gray
    }
  };

  // Precompute neighbor sets for hover highlighting
  const { neighbors, nodeById } = useMemo(() => {
    const nb = {};
    const byId = {};
    nodes.forEach((n) => {
      byId[n.id] = n;
      nb[n.id] = new Set();
    });
    links.forEach((l) => {
      const src = typeof l.source === "object" ? l.source.id : l.source;
      const tgt = typeof l.target === "object" ? l.target.id : l.target;
      if (!nb[src]) nb[src] = new Set();
      if (!nb[tgt]) nb[tgt] = new Set();
      nb[src].add(tgt);
      nb[tgt].add(src);
    });
    return { neighbors: nb, nodeById: byId };
  }, [nodes, links]);

  const handleNodeClick = useCallback(
    (node) => {
      if (onNodeClick) onNodeClick(node);
    },
    [onNodeClick]
  );

  const handleNodeHover = useCallback(
    (node) => {
      setHoverNode(node || null);
    },
    [setHoverNode]
  );

  const isHighlighted = (node) => {
    if (!hoverNode) return false;
    if (hoverNode.id === node.id) return true;
    const nbSet = neighbors[hoverNode.id];
    return nbSet && nbSet.has(node.id);
  };

  const isDimmed = (node) => {
    if (!hoverNode) return false;
    return !isHighlighted(node);
  };

  return (
    <div className="bg-white rounded-xl shadow-sm p-3 w-full h-full">
      <div className="flex items-center justify-between mb-2">
        <h2 className="font-semibold text-sm">Author–Domain network</h2>
        <span className="text-[11px] text-gray-500">
          {nodes.length} nodes · {links.length} edges
        </span>
      </div>
      <div className="border rounded-lg overflow-hidden h-[480px]">
        <ForceGraph2D
          graphData={{ nodes, links }}
          nodeLabel={(node) =>
            `${node.type || "node"}: ${node.label || node.id}${
              node.type === "domain" && node.credScore !== undefined
                ? ` (credScore: ${node.credScore})`
                : ""
            }`
          }
          nodeCanvasObject={(node, ctx, globalScale) => {
            const label = node.label || node.id;
            const fontSize = 12 / globalScale;
            const color = getNodeColor(node);

            const highlighted = isHighlighted(node);
            const dimmed = isDimmed(node);

            // node circle
            ctx.beginPath();
            ctx.arc(node.x, node.y, highlighted ? 6 : 4, 0, 2 * Math.PI, false);
            ctx.fillStyle = dimmed ? "rgba(209, 213, 219, 0.5)" : color;
            ctx.fill();

            // small white border
            if (highlighted) {
              ctx.lineWidth = 1;
              ctx.strokeStyle = "#ffffff";
              ctx.stroke();
            }

            // label
            ctx.font = `${fontSize}px system-ui`;
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            ctx.fillStyle = dimmed ? "rgba(107,114,128,0.4)" : "#111827";
            ctx.fillText(label, node.x, node.y + 6);
          }}
          linkColor={(link) => {
            if (!hoverNode) return "rgba(148,163,184,0.7)";
            const src =
              typeof link.source === "object" ? link.source.id : link.source;
            const tgt =
              typeof link.target === "object" ? link.target.id : link.target;
            const nbSet = neighbors[hoverNode.id];
            const connected =
              hoverNode.id === src ||
              hoverNode.id === tgt ||
              (nbSet && (nbSet.has(src) || nbSet.has(tgt)));
            return connected
              ? "rgba(148,163,184,0.9)"
              : "rgba(229,231,235,0.5)";
          }}
          linkWidth={(link) => {
            if (!hoverNode) return 1;
            const src =
              typeof link.source === "object" ? link.source.id : link.source;
            const tgt =
              typeof link.target === "object" ? link.target.id : link.target;
            const nbSet = neighbors[hoverNode.id];
            const connected =
              hoverNode.id === src ||
              hoverNode.id === tgt ||
              (nbSet && (nbSet.has(src) || nbSet.has(tgt)));
            return connected ? 1.8 : 0.6;
          }}
          linkDirectionalParticles={0}
          backgroundColor="#f9fafb"
          onNodeClick={handleNodeClick}
          onNodeHover={handleNodeHover}
        />
      </div>
      <div className="mt-2 flex items-center justify-between text-[11px] text-gray-500">
        <div className="flex flex-wrap items-center gap-3">
          <LegendDot color="#4f46e5" label="Authors" />
          <LegendDot color="#16a34a" label="Trusted domains " />
          <LegendDot color="#f97316" label="Neutral domains " />
          <LegendDot color="#ef4444" label="Low-cred domains " />
        </div>
        <div>Hover to highlight ego-network · Click for details</div>
      </div>
    </div>
  );
}

function LegendDot({ color, label }) {
  return (
    <div className="flex items-center gap-1">
      <span
        className="inline-block rounded-full"
        style={{ width: 8, height: 8, backgroundColor: color }}
      />
      <span>{label}</span>
    </div>
  );
}
