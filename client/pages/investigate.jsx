// client/pages/investigate.jsx
import { useState, useEffect } from "react";
import Link from "next/link";
import PostsTable from "../components/PostsTable";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function InvestigatePage() {
  // Added 'author' to state
  const [filters, setFilters] = useState({
    subreddit: "",
    author: "", 
    keyword: "",
    startDate: "",
    endDate: "",
    minUps: "",
  });

  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadPosts = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.set("limit", "100");
      
      // Send all filters including Author
      if (filters.subreddit) params.set("subreddit", filters.subreddit);
      if (filters.author) params.set("author", filters.author); // <--- NEW
      if (filters.keyword) params.set("keyword", filters.keyword);
      if (filters.startDate) params.set("start", filters.startDate);
      if (filters.endDate) params.set("end", filters.endDate);
      if (filters.minUps) params.set("min_ups", filters.minUps);

      const res = await fetch(`${API_URL}/api/posts?${params}`);
      const json = await res.json();
      
      // Safety check
      if (Array.isArray(json)) {
        setPosts(json);
      } else {
        setPosts([]);
      }
    } catch (err) {
      console.error("Error loading posts:", err);
      setPosts([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPosts();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <main className="max-w-7xl mx-auto py-8 px-4 space-y-6">
        
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-800">Investigate Posts</h1>
            <p className="text-sm text-slate-500">Deep dive into author behavior and sentiment.</p>
          </div>
          <Link href="/" className="text-sm text-indigo-600 font-medium hover:underline">
            ← Back to Dashboard
          </Link>
        </div>

        {/* Custom Filters Bar */}
        <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 grid grid-cols-1 md:grid-cols-5 gap-4">
          <input
            type="text"
            placeholder="Subreddit (e.g. politics)"
            className="border p-2 rounded text-sm outline-none focus:ring-2 focus:ring-indigo-500"
            value={filters.subreddit}
            onChange={(e) => setFilters({...filters, subreddit: e.target.value})}
          />
          
          {/* NEW AUTHOR INPUT */}
          <input
            type="text"
            placeholder="Author (e.g. AutoModerator)"
            className="border p-2 rounded text-sm outline-none focus:ring-2 focus:ring-indigo-500"
            value={filters.author}
            onChange={(e) => setFilters({...filters, author: e.target.value})}
          />
          
          <input
            type="text"
            placeholder="Keyword (e.g. election)"
            className="border p-2 rounded text-sm outline-none focus:ring-2 focus:ring-indigo-500"
            value={filters.keyword}
            onChange={(e) => setFilters({...filters, keyword: e.target.value})}
          />
          
          <input
            type="number"
            placeholder="Min Upvotes"
            className="border p-2 rounded text-sm outline-none focus:ring-2 focus:ring-indigo-500"
            value={filters.minUps}
            onChange={(e) => setFilters({...filters, minUps: e.target.value})}
          />
          
          <button 
            onClick={loadPosts}
            className="bg-indigo-600 text-white rounded px-4 py-2 text-sm font-medium hover:bg-indigo-700 transition shadow-sm"
          >
            Apply Filters
          </button>
        </div>

        {/* Results Table */}
        {loading ? (
          <div className="text-center py-10 text-slate-500 flex flex-col items-center">
            <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mb-2"></div>
            Running analysis...
          </div>
        ) : (
          <PostsTable posts={posts} />
        )}
      </main>
    </div>
  );
}