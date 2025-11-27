// client/components/FiltersBar.jsx
import React from "react";

/**
 * Props:
 * - values: { subreddit, author, keyword, startDate, endDate, minUps }
 * - onChange: (newValues) => void
 * - onApply: () => void
 */
export default function FiltersBar({ values, onChange, onApply }) {
  const v = values || {};

  const handleChange = (field) => (e) => {
    onChange && onChange({ ...v, [field]: e.target.value });
  };

  return (
    <div className="bg-white rounded-xl shadow-sm p-4 flex flex-wrap gap-3 items-end">
      <InputField
        label="Subreddit"
        placeholder="e.g. politics"
        value={v.subreddit || ""}
        onChange={handleChange("subreddit")}
      />
      
      {/* --- NEW: Author Input --- */}
      <InputField
        label="Author"
        placeholder="e.g. AutoModerator"
        value={v.author || ""}
        onChange={handleChange("author")}
      />

      <InputField
        label="Keyword"
        placeholder="e.g. election"
        value={v.keyword || ""}
        onChange={handleChange("keyword")}
      />
      <InputField
        label="Start date"
        type="date"
        value={v.startDate || ""}
        onChange={handleChange("startDate")}
      />
      <InputField
        label="End date"
        type="date"
        value={v.endDate || ""}
        onChange={handleChange("endDate")}
      />
      <InputField
        label="Min upvotes"
        type="number"
        value={v.minUps || ""}
        onChange={handleChange("minUps")}
        min={0}
      />
      <button
        onClick={onApply}
        className="ml-auto px-4 py-2 rounded-md bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 transition"
      >
        Apply filters
      </button>
    </div>
  );
}

function InputField({ label, ...rest }) {
  return (
    <div className="flex flex-col text-xs">
      <label className="text-gray-500 mb-1">{label}</label>
      <input
        className="border border-slate-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 w-32 sm:w-40"
        {...rest}
      />
    </div>
  );
}