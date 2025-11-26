// client/components/TimeSeriesKPI.jsx
import React, { useState } from "react";
import dynamic from "next/dynamic";

const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

export default function TimeSeriesKPI({
  timeSeries = [],
  topicSeries = [],
  kpi = {},
  loading = false,
  onExplainRange,
}) {
  const [selectedRange, setSelectedRange] = useState(null);
  const [explanation, setExplanation] = useState(null);

  // --------------------------------------------------------------------------
  // FIX: Safety Checks
  // Ensure we rely on real arrays. If data is an error object or null, use [].
  // --------------------------------------------------------------------------
  const safeTimeSeries = Array.isArray(timeSeries) ? timeSeries : [];
  const safeTopicSeries = Array.isArray(topicSeries) ? topicSeries : [];

  // ---- Main posts time-series (single line) ----
  const mainSeries = [
    {
      name: "Posts",
      // Use safeTimeSeries here
      data: safeTimeSeries.map((p) => ({
        x: p.date,
        y: p.count || 0,
      })),
    },
  ];

  const mainOptions = {
    chart: {
      id: "posts-timeseries",
      type: "area",
      zoom: { enabled: true },
      toolbar: { autoSelected: "zoom" },
      events: {
        selection: (chartContext, { xaxis }) => {
          if (!xaxis || xaxis.min == null || xaxis.max == null) return;
          const startISO = new Date(xaxis.min).toISOString().slice(0, 10);
          const endISO = new Date(xaxis.max).toISOString().slice(0, 10);
          const range = { startDate: startISO, endDate: endISO };
          setSelectedRange(range);
          setExplanation(null);
        },
      },
    },
    dataLabels: { enabled: false },
    stroke: { curve: "smooth" },
    xaxis: { type: "datetime" },
    tooltip: { x: { format: "yyyy-MM-dd" } },
    markers: { size: 3 },
  };

  // ---- Topic trends time-series (multi-line) ----
  // Use safeTopicSeries here
  const topicsApexSeries = safeTopicSeries.map((topic) => ({
    name: topic.name,
    data: (Array.isArray(topic.data) ? topic.data : []).map((p) => ({
      x: p.date,
      y: p.count || 0,
    })),
  }));

  const topicOptions = {
    chart: {
      id: "topic-timeseries",
      type: "line",
      zoom: { enabled: false },
      toolbar: { show: false },
    },
    dataLabels: { enabled: false },
    stroke: { curve: "smooth" },
    xaxis: { type: "datetime" },
    tooltip: {
      x: { format: "yyyy-MM-dd" },
    },
    legend: {
      position: "bottom",
    },
  };

  const fmt = (n) =>
    n === undefined || n === null ? "—" : Number(n).toLocaleString();

  const handleExplainClick = async () => {
    if (!selectedRange || !onExplainRange) return;
    try {
      const maybe = await onExplainRange(selectedRange);
      if (typeof maybe === "string") {
        setExplanation(maybe);
      } else if (maybe && maybe.summary) {
        setExplanation(maybe.summary);
      }
    } catch (err) {
      console.error("Error in onExplainRange:", err);
    }
  };

  return (
    <div className="space-y-6">
      {/* Heading */}
   {/* Sub-header for the section */}
      <div className="mb-2">
          <h2 className="text-lg font-bold text-slate-700">Activity Overview</h2>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <KpiCard label="Total posts" value={fmt(kpi.total_posts)} />
        <KpiCard label="Avg / day" value={fmt(kpi.avg_per_day)} />
        <KpiCard
          label="Growth (last 7d)"
          value={
            kpi.growth_7d === null || kpi.growth_7d === undefined
              ? "—"
              : `${kpi.growth_7d}%`
          }
        />
        <KpiCard label="Top subreddit" value={kpi.top_subreddit || "—"} />
      </div>

      {/* Main posts chart */}
      <div className="bg-white rounded-xl shadow-sm p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold">Time-series (daily posts)</h2>
          <span className="text-xs text-gray-500">
            {loading ? "Loading…" : `${safeTimeSeries.length} points`}
          </span>
        </div>

        {safeTimeSeries.length === 0 ? (
          <div className="py-10 text-center text-gray-500">
            {loading ? "Loading data..." : "No data available (Server might be offline)."}
          </div>
        ) : (
          <Chart
            type="area"
            height={360}
            series={mainSeries}
            options={mainOptions}
          />
        )}

        <div className="mt-3 flex items-center justify-between">
          {selectedRange ? (
            <div className="text-xs text-gray-600">
              Selected range:{" "}
              <span className="font-medium">
                {selectedRange.startDate} – {selectedRange.endDate}
              </span>
            </div>
          ) : (
            <div className="text-xs text-gray-400">
              Drag on the chart to select a range to explain.
            </div>
          )}

          <button
            onClick={handleExplainClick}
            disabled={!selectedRange || !onExplainRange}
            className={`px-3 py-1 rounded text-xs font-medium ${
              selectedRange && onExplainRange
                ? "bg-indigo-600 text-white"
                : "bg-gray-200 text-gray-500 cursor-not-allowed"
            }`}
          >
            Explain selection
          </button>
        </div>

        {explanation && (
          <div className="mt-4 border rounded-md p-3 bg-indigo-50 text-sm text-gray-800">
            {explanation}
          </div>
        )}
      </div>

      {/* Topic trends chart (rubric b) */}
      <div className="bg-white rounded-xl shadow-sm p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold">
            Topic trends over time (key themes)
          </h2>
          <span className="text-xs text-gray-500">
            {topicsApexSeries.length
              ? `${topicsApexSeries.length} topics`
              : "No topic data"}
          </span>
        </div>

        {topicsApexSeries.length === 0 ? (
          <div className="py-6 text-center text-gray-500 text-sm">
            Provide <code>topicSeries</code> prop to visualize trends for
            key topics (e.g., “election”, “protest”, “economy”).
          </div>
        ) : (
          <Chart
            type="line"
            height={320}
            series={topicsApexSeries}
            options={topicOptions}
          />
        )}
      </div>
    </div>
  );
}

function KpiCard({ label, value }) {
  return (
    <div className="bg-white rounded-xl shadow-sm p-4">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-xl font-semibold mt-1">{value}</div>
    </div>
  );
}