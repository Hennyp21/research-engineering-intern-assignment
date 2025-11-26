// client/components/PostsTable.jsx
import React from "react";

/**
 * Props:
 * - posts: [{ id, subreddit, author, title, ups, created_utc, domain, sentiment }]
 * - onPostClick?: (post) => void
 */
export default function PostsTable({ posts = [], onPostClick }) {
  if (!posts.length) {
    return (
      <div className="bg-white rounded-xl shadow-sm p-4 text-sm text-gray-500">
        No posts for this selection.
      </div>
    );
  }

  // Helper for sentiment color
  const getSentimentColor = (score) => {
    if (score === undefined || score === null) return "text-gray-400";
    if (score >= 0.05) return "text-green-600 font-medium"; // Positive
    if (score <= -0.05) return "text-red-600 font-medium";   // Negative/Rage
    return "text-gray-400"; // Neutral
  };

  return (
    <div className="bg-white rounded-xl shadow-sm overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-50">
          <tr>
            <Th>ID</Th>
            <Th>Subreddit</Th>
            <Th>Author</Th>
            <Th>Title</Th>
            <Th>Ups</Th>
            <Th>Date</Th>
            <Th>Domain</Th>
            <Th>Sentiment</Th> {/* NEW COLUMN */}
          </tr>
        </thead>
        <tbody>
          {posts.map((p) => (
            <tr
              key={p.id}
              className="border-t hover:bg-indigo-50 cursor-pointer"
              onClick={() => onPostClick && onPostClick(p)}
            >
              <Td className="font-mono text-xs max-w-[80px] truncate">
                {p.id}
              </Td>
              <Td>{p.subreddit}</Td>
              <Td>{p.author}</Td>
              <Td className="max-w-[400px] truncate">{p.title}</Td>
              <Td>{p.ups}</Td>
              <Td className="text-xs text-gray-500">
                {p.created_utc
                  ? String(p.created_utc).slice(0, 10)
                  : "—"}
              </Td>
              <Td className="text-xs text-gray-500">
                {p.domain || "—"}
              </Td>
              {/* NEW SENTIMENT CELL */}
              <Td className={getSentimentColor(p.sentiment)}>
                {p.sentiment !== undefined && p.sentiment !== null
                  ? p.sentiment.toFixed(2)
                  : "—"}
              </Td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Th({ children }) {
  return (
    <th className="px-3 py-2 text-left text-xs font-semibold text-gray-600 uppercase">
      {children}
    </th>
  );
}

function Td({ children, className = "" }) {
  return (
    <td className={`px-3 py-2 align-top ${className}`}>{children}</td>
  );
}