// client/components/FiltersBar.jsx
import React from "react";

/**
 * Props:
 *  - values: { subreddit, keyword, startDate, endDate, minUps }
 *  - onChange: (newValues) => void
 *  - onApply: () => void
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
        className="ml-auto px-4 py-2 rounded-md bg-indigo-600 text-white text-sm font-medium"
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
        className="border rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
        {...rest}
      />
    </div>
  );
}
