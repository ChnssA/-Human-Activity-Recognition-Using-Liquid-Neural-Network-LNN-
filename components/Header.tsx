import React, { useState } from 'react';
import { Users, ChevronDown, ChevronUp, BookOpen, LayoutDashboard, Code, BookText } from 'lucide-react';

interface TeamMember {
  name: string;
  email: string;
}

interface HeaderProps {
  title: string;
  teamMembers: TeamMember[];
  currentView: 'dashboard' | 'report' | 'codeExplanation' | 'code';
  setView: (view: 'dashboard' | 'report' | 'codeExplanation' | 'code') => void;
}

const Header: React.FC<HeaderProps> = ({ title, teamMembers, currentView, setView }) => {
  const [isTeamVisible, setIsTeamVisible] = useState(false);

  const baseButtonClass = "flex items-center justify-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg transition-all duration-300 shadow-md border flex-shrink-0";
  const activeButtonClass = "bg-cyan-600 text-white border-cyan-500";
  const inactiveButtonClass = "bg-cyan-600/20 hover:bg-cyan-600/50 text-cyan-300 border-cyan-700";

  return (
    <header className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 shadow-lg">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <h1 className="text-3xl md:text-4xl font-bold text-center md:text-left text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
          {title}
        </h1>
        <div className="flex flex-wrap items-center justify-center gap-2">
           <button
             onClick={() => setView('dashboard')}
             className={`${baseButtonClass} ${currentView === 'dashboard' ? activeButtonClass : inactiveButtonClass}`}
           >
             <LayoutDashboard className="h-5 w-5" />
             <span>Dashboard</span>
           </button>
           <button
             onClick={() => setView('report')}
             className={`${baseButtonClass} ${currentView === 'report' ? activeButtonClass : inactiveButtonClass}`}
           >
             <BookOpen className="h-5 w-5" />
             <span>Detailed Report</span>
           </button>
           <button
             onClick={() => setView('codeExplanation')}
             className={`${baseButtonClass} ${currentView === 'codeExplanation' ? activeButtonClass : inactiveButtonClass}`}
           >
             <BookText className="h-5 w-5" />
             <span>Code Explanation</span>
           </button>
           <button
             onClick={() => setView('code')}
             className={`${baseButtonClass} ${currentView === 'code' ? activeButtonClass : inactiveButtonClass}`}
           >
             <Code className="h-5 w-5" />
             <span>Code Implementation</span>
           </button>
        </div>
      </div>

      <div className="mt-6">
        <button
          onClick={() => setIsTeamVisible(!isTeamVisible)}
          className="w-full flex justify-between items-center p-3 bg-gray-700/60 hover:bg-gray-700 rounded-lg transition-colors duration-300"
        >
          <div className="flex items-center">
            <Users className="h-5 w-5 mr-3 text-cyan-400" />
            <span className="font-semibold">Team Members</span>
          </div>
          {isTeamVisible ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
        </button>
        {isTeamVisible && (
          <div className="mt-3 bg-gray-800 p-4 rounded-lg border border-gray-700 animate-fade-in-down">
            <ul className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {teamMembers.map((member) => (
                <li key={member.name} className="text-sm">
                  <p className="font-bold text-gray-200">{member.name}</p>
                  <a href={`mailto:${member.email}`} className="text-cyan-400 hover:text-cyan-300 break-all">{member.email}</a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;