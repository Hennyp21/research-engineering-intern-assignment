// client/components/CommunityPie.jsx
import React from "react";
import dynamic from "next/dynamic";

const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

/**
 * Props:
 *  - data: [{ subreddit: string, count: number }]
 */
export default function CommunityPie({ data = [] }) {
  if (!data.length) {
    return (
      <div className="bg-white rounded-xl shadow-sm p-4 text-sm text-gray-500">
        No community data.
      </div>
    );
  }

  const labels = data.map((d) => d.subreddit || d.key || "unknown");
  const series = data.map((d) => d.count || 0);

  const options = {
    labels,
    legend: { position: "bottom" },
    tooltip: {
      y: {
        formatter: (val, opts) => {
          const total = series.reduce((a, b) => a + b, 0) || 1;
          const pct = ((val / total) * 100).toFixed(1);
          return `${val.toLocaleString()} posts (${pct}%)`;
        },
      },
    },
  };

  return (
    <div className="bg-white rounded-xl shadow-sm p-4">
      <h2 className="font-semibold mb-2">Community distribution</h2>
      <Chart type="donut" height={320} series={series} options={options} />
    </div>
  );
}
