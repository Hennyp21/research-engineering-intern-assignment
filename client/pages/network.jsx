// client/pages/network.jsx
import { useState, useEffect } from "react";
import FiltersBar from "../components/FiltersBar";
import NetworkGraph from "../components/NetworkGraph";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function NetworkPage() {
  const [filters, setFilters] = useState({
    subreddit: "",
    keyword: "",
    startDate: "",
    endDate: "",
    minUps: "",
  });

  const [nodes, setNodes] = useState([]);
  const [links, setLinks] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const loadNetwork = async () => {
    setLoading(true);
    setErrorMsg("");
    try {
      const params = new URLSearchParams();
      if (filters.subreddit) params.set("subreddit", filters.subreddit);
      if (filters.keyword) params.set("keyword", filters.keyword);
      params.set("limit", "300");

      const res = await fetch(`${API_URL}/api/network?${params.toString()}`);
      if (!res.ok) {
        throw new Error(`Network API error: ${res.status}`);
      }
      const json = await res.json();
      setNodes(json.nodes || []);
      setLinks(json.edges || json.links || []);
    } catch (err) {
      console.error("Error loading network:", err);
      setErrorMsg("Could not load network graph. Try adjusting filters.");
      setNodes([]);
      setLinks([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadNetwork();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleNodeClick = (node) => {
    setSelectedNode(node);
  };

  const handleApplyFilters = () => {
    setSelectedNode(null);
    loadNetwork();
  };

  return (
    <div className="app-shell">
      <main className="app-inner">
        <header className="flex items-center justify-between mb-2">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">
              Network view
            </h1>
            <p className="text-sm text-gray-600">
              Explore how authors and domains connect for the selected
              narrative. 
            </p>
          </div>
          <a
            href="/"
            className="hidden sm:inline-flex text-sm px-3 py-1 rounded-full bg-gray-200 text-gray-700 hover:bg-gray-300"
          >
            ← Back to overview
          </a>
        </header>

        <FiltersBar
          values={filters}
          onChange={setFilters}
          onApply={handleApplyFilters}
        />

        {loading && (
          <div className="mt-3 text-xs text-gray-500">
            Building network from filtered posts…
          </div>
        )}
        {errorMsg && (
          <div className="mt-3 text-xs text-red-500">{errorMsg}</div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-4">
          <div className="lg:col-span-2 min-h-[520px]">
            <NetworkGraph
              nodes={nodes}
              links={links}
              onNodeClick={handleNodeClick}
            />
          </div>

          <div className="lg:col-span-1 bg-white rounded-xl shadow-sm p-4 text-sm flex flex-col">
            <h2 className="font-semibold mb-2">Node details</h2>
            {!selectedNode ? (
              <div className="text-xs text-gray-500">
                Click a node in the graph to inspect an author or domain.
              </div>
            ) : (
              <div className="space-y-3">
                <div>
                  <div className="text-[11px] uppercase text-gray-400">
                    Type
                  </div>
                  <div className="font-semibold">
                    {selectedNode.type || "Unknown"}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] uppercase text-gray-400">
                    Label
                  </div>
                  <div className="break-all">
                    {selectedNode.label || selectedNode.id}
                  </div>
                </div>
                <div className="text-xs text-gray-500">
                  This node participates in the author–domain network for
                  the current filters. In the report, you can describe
                  how clusters of authors around certain domains might
                  indicate aligned media ecosystems or coordinated sharing.
                </div>
              </div>
            )}

            <div className="mt-4 pt-3 border-t border-gray-100 text-[11px] text-gray-500">
              Tip: Use the filters above to focus on a single subreddit or
              keyword, then inspect which domains different author clusters
              are amplifying.
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
