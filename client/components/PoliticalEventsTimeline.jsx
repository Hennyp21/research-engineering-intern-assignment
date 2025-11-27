import React from 'react';

const PoliticalEventsTimeline = () => {
  const events = [
    {
      date: "2024-08-06",
      title: "Walz VP Pick",
      description: "Tim Walz selected as Harris running mate",
      color: "bg-blue-500",
      icon: "👥"
    },
    {
      date: "2024-08-19",
      title: "DNC Convention",
      description: "Democratic National Convention in Chicago",
      color: "bg-blue-600",
      icon: "🎤"
    },
    {
      date: "2024-09-10",
      title: "Presidential Debate",
      description: "Harris vs Trump debate",
      color: "bg-purple-500",
      icon: "⚔️"
    },
    {
      date: "2024-10-01",
      title: "VP Debate",
      description: "Vance vs Walz debate",
      color: "bg-purple-400",
      icon: "🎯"
    },
    {
      date: "2024-11-05",
      title: "Election Day",
      description: "2024 Presidential Election",
      color: "bg-red-600",
      icon: "🗳️"
    },
    {
      date: "2024-11-06",
      title: "Trump Victory",
      description: "Trump wins presidency",
      color: "bg-red-500",
      icon: "🏆"
    },
    {
      date: "2024-12-17",
      title: "Electoral College",
      description: "Electoral College votes certified",
      color: "bg-amber-500",
      icon: "📜"
    },
    {
      date: "2025-01-06",
      title: "Congress Certification",
      description: "Congress certifies election results",
      color: "bg-emerald-500",
      icon: "✅"
    },
    {
      date: "2025-01-20",
      title: "Inauguration",
      description: "Trump inaugurated as 47th President",
      color: "bg-indigo-600",
      icon: "🇺🇸"
    }
  ];

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-slate-700">Major Political Events Timeline</h3>
        <span className="text-xs text-slate-400 bg-slate-100 px-2 py-1 rounded">Aug 2024 - Feb 2025</span>
      </div>
      
      {/* Timeline Container with Horizontal Scroll */}
      <div className="relative overflow-x-auto pb-2">
        <div className="flex gap-4 min-w-max">
          {events.map((event, idx) => (
            <div key={idx} className="flex flex-col items-center min-w-[140px]">
              {/* Event Card */}
              <div className="group relative">
                {/* Icon Circle */}
                <div className={`${event.color} w-14 h-14 rounded-full flex items-center justify-center text-2xl shadow-lg transform transition-transform duration-200 group-hover:scale-110 mb-3`}>
                  {event.icon}
                </div>
                
                {/* Event Info */}
                <div className="text-center">
                  <div className="font-semibold text-sm text-slate-800 mb-1">
                    {event.title}
                  </div>
                  <div className="text-xs text-slate-500 mb-2">
                    {new Date(event.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                  </div>
                  
                  {/* Hover Card */}
                  <div className="invisible group-hover:visible opacity-0 group-hover:opacity-100 transition-opacity duration-200 absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 z-10">
                    <div className="bg-slate-900 text-white text-xs rounded-lg py-2 px-3 whitespace-nowrap shadow-xl">
                      {event.description}
                      <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-slate-900"></div>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Connector Line */}
              {idx < events.length - 1 && (
                <div className="absolute top-7 left-[50%] w-[140px] h-0.5 bg-gradient-to-r from-slate-300 to-slate-200 transform translate-x-[10px]"></div>
              )}
            </div>
          ))}
        </div>
      </div>
      
      {/* Legend */}
      <div className="mt-6 pt-4 border-t border-slate-100 flex flex-wrap gap-3 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500"></div>
          <span className="text-slate-600">Campaign Events</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-purple-500"></div>
          <span className="text-slate-600">Debates</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <span className="text-slate-600">Election</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
          <span className="text-slate-600">Certification</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-indigo-600"></div>
          <span className="text-slate-600">Inauguration</span>
        </div>
      </div>
    </div>
  );
};

export default PoliticalEventsTimeline;