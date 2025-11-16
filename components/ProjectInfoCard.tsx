
import React from 'react';

interface ProjectInfoCardProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}

const ProjectInfoCard: React.FC<ProjectInfoCardProps> = ({ title, icon, children }) => {
  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl shadow-lg overflow-hidden">
      <div className="flex items-center p-4 bg-gray-900/40 border-b border-gray-700">
        {icon}
        <h2 className="text-xl font-bold ml-3 text-white">{title}</h2>
      </div>
      <div className="p-6">
        {children}
      </div>
    </div>
  );
};

export default ProjectInfoCard;
