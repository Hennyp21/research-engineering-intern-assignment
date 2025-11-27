// client/pages/index.jsx
import { useEffect, useState } from "react";
import TimeSeriesKPI from "../components/TimeSeriesKPI";
import FiltersBar from "../components/FiltersBar";
import CommunityPie from "../components/CommunityPie";
import ChatbotPanel from "../components/ChatbotPanel";
import AISummaryBox from "../components/AISummaryBox";
import PoliticalEventsTimeline from "../components/PoliticalEventsTimeline";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";


export default function HomePage() {
  const [filters, setFilters] = useState({
    subreddit: "",
    keyword: "",
    startDate: "",
    endDate: "",
    author: "",
  });

  // Data States
  const [timeSeries, setTimeSeries] = useState([]);
  const [topicSeries, setTopicSeries] = useState([]);
  const [kpi, setKpi] = useState({});
  const [communityData, setCommunityData] = useState([]);
  
  // AI Summary State
  const [aiSummary, setAiSummary] = useState(null);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loadingTS, setLoadingTS] = useState(false);

  // --------------------------------------------------
  // Helper: Build Query String
  // --------------------------------------------------
  const buildParams = () => {
    const params = new URLSearchParams();
    if (filters.startDate) params.set("start", filters.startDate);
    if (filters.endDate) params.set("end", filters.endDate);
    if (filters.subreddit) params.set("subreddit", filters.subreddit);
    if (filters.keyword) params.set("keyword", filters.keyword);
    if (filters.author) params.set("author", filters.author);
    return params.toString();
  };

  // --------------------------------------------------
  // 1. Load Charts & KPIs
  // --------------------------------------------------
  const loadOverview = async () => {
    setLoadingTS(true);
    try {
      const qs = buildParams();
      const tsRes = await fetch(`${API_BASE}/api/time-series?${qs}`);
      const tsJson = await tsRes.json();
      
      if (Array.isArray(tsJson)) {
        setTimeSeries(tsJson);
      } else {
        setTimeSeries([]);
      }

      const tlRes = await fetch(`${API_BASE}/api/top-lists?${qs}`);
      const tlJson = await tlRes.json();

      setKpi({
        total_posts: tlJson.total_posts,
        avg_per_day: tlJson.avg_per_day,
        growth_7d: tlJson.growth_7d,
        top_subreddit: tlJson.top_subreddits?.[0]?.key || tlJson.top_subreddits?.[0]?.subreddit || null,
      });

      setCommunityData(
        (tlJson.top_subreddits || []).map((d) => ({
          subreddit: d.key || d.subreddit,
          count: d.count,
        }))
      );
    } catch (err) {
      console.error("Error loading overview:", err);
    } finally {
      setLoadingTS(false);
    }
  };

  const loadTopicSeries = async () => {
    try {
      const qs = buildParams();
      const res = await fetch(`${API_BASE}/api/topic-time-series?${qs}`);
      if (res.ok) {
        const json = await res.json();
        setTopicSeries(json || []);
      }
    } catch (err) {
      console.error("Error loading topics:", err);
    }
  };

  // --------------------------------------------------
  // 2. Load AI Summary
  // --------------------------------------------------
  const loadAiSummary = async () => {
    setLoadingSummary(true);
    try {
      const body = {
        mode: "range",
        start: filters.startDate || null,
        end: filters.endDate || null,
        subreddit: filters.subreddit || null,
        keyword: filters.keyword || null,
        author: filters.author || null
      };

      const res = await fetch(`${API_BASE}/api/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      
      const data = await res.json();
      setAiSummary(data);
    } catch (err) {
      console.error("Error loading AI summary:", err);
      setAiSummary(null);
    } finally {
      setLoadingSummary(false);
    }
  };

  // --------------------------------------------------
  // Effects
  // --------------------------------------------------
  useEffect(() => {
    loadOverview();
    loadTopicSeries();
    loadAiSummary();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleApplyFilters = async () => {
    await Promise.all([
      loadOverview(),
      loadTopicSeries(),
      loadAiSummary()
    ]);
  };

  const handleExplainRange = async ({ startDate, endDate }) => {
    return "See the AI Intelligence Brief on the right for full context.";
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
      
      {/* --- 1. MODERN NAVBAR --- */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            {/* Logo Icon (Simple colored square) */}
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold text-lg shadow-sm">
              S
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-800">REDDIT ANALYZER</h1>
            </div>
          </div>
          <div className="text-xs font-medium text-slate-400 bg-slate-100 px-2 py-1 rounded">
            v1.0 Live Dashboard
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8 space-y-8">
        
        {/* --- 2. FILTERS SECTION (Now Below Title) --- */}
        <section className="bg-white rounded-xl shadow-sm border border-slate-200 p-1">
           {/* We wrap FiltersBar to allow it to sit nicely in the card */}
           <div className="p-2">
              <FiltersBar
                values={filters}
                onChange={setFilters}
                onApply={handleApplyFilters}
              />
           </div>
        </section>

        {/* --- 3. MAIN DASHBOARD GRID --- */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* LEFT COLUMN: Data Visualization */}
          <div className="lg:col-span-2 space-y-8">
            <TimeSeriesKPI
              timeSeries={timeSeries}
              topicSeries={topicSeries}
              kpi={kpi}
              loading={loadingTS}
              onExplainRange={handleExplainRange}
            />
            
            {/* NEW: Political Events Timeline */}
            <PoliticalEventsTimeline />
            
            {/* Community Pie with a nice header */}
            <div className="space-y-3">
              <h3 className="text-lg font-bold text-slate-700">Community Distribution</h3>
              <CommunityPie data={communityData} />
            </div>
          </div>

          {/* RIGHT COLUMN: Intelligence Layer */}
          <div className="lg:col-span-1 flex flex-col gap-6">
            
            {/* AI Summary Box */}
            <div className="sticky top-24 space-y-6">
              <AISummaryBox 
                data={aiSummary} 
                loading={loadingSummary} 
              />
              
              {/* Chatbot */}
              <div className="h-[500px] shadow-lg rounded-xl overflow-hidden border border-slate-200"> 
                <ChatbotPanel />
              </div>
            </div>
            
          </div>
        </div>

        {/* --- 4. FOOTER LINKS --- */}
        <section className="pt-8 border-t border-slate-200">
          <div className="flex items-center justify-between mb-4">
             <h2 className="text-lg font-bold text-slate-800">Deep Dive Tools</h2>
             <span className="text-xs text-slate-400">Advanced investigation</span>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <a href="/investigate" className="group bg-white hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 rounded-xl p-5 transition-all duration-200">
              <div className="flex items-center justify-between">
                <div className="font-semibold text-slate-800 group-hover:text-indigo-700">Investigate Posts</div>
                <span className="text-slate-300 group-hover:text-indigo-400">→</span>
              </div>
              <div className="text-xs text-slate-500 mt-1 group-hover:text-indigo-600/70">Granular table view of individual posts</div>
            </a>

            <a href="/network" className="group bg-white hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 rounded-xl p-5 transition-all duration-200">
              <div className="flex items-center justify-between">
                <div className="font-semibold text-slate-800 group-hover:text-indigo-700">Network Graph</div>
                <span className="text-slate-300 group-hover:text-indigo-400">→</span>
              </div>
              <div className="text-xs text-slate-500 mt-1 group-hover:text-indigo-600/70">Visualize connection between authors & domains</div>
            </a>
          </div>
        </section>

      </main>
    </div>
  );
}