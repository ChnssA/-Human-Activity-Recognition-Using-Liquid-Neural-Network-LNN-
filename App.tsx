import React, { useState } from 'react';
import Header from './components/Header';
import ProjectInfoCard from './components/ProjectInfoCard';
import MethodologyStepper from './components/MethodologyStepper';
import ResultsDashboard from './components/ResultsDashboard';
import LnnExplainer from './components/LnnExplainer';
import ProjectReport from './components/ProjectReport';
import CodeExplainer from './components/CodeExplainer';
import CodeExplanation from './components/CodeExplanation';
import { BrainCircuit, Target, BookOpen, BarChart4 } from 'lucide-react';

const App: React.FC = () => {
  const [view, setView] = useState<'dashboard' | 'report' | 'codeExplanation' | 'code'>('dashboard');

  const teamMembers = [
    { name: 'Damarla Neeraj Sai', email: 'am.sc.p2aml25030@am.students.amrita.edu' },
    { name: 'Shanmukeswara Reddy M', email: 'am.sc.p2aml25018@am.students.amrita.edu' },
    { name: 'Nikhil Reddy D', email: 'am.sc.p2aml25023@am.students.amrita.edu' },
    { name: 'CH Narasimha Shruti Sagar', email: 'am.sc.p2aml25010@am.students.amrita.edu' },
    { name: 'Boppana Ramnivas', email: 'am.sc.p2ari25007@am.students.amrita.edu' },
  ];

  const projectObjective = `The main goal of this project is to create a highly effective and resilient model for categorizing human activities (including walking, standing, and running) through the use of time-series sensor data. Our intention is to utilize the advanced temporal modeling features of Liquid Time-Constant (LTC) Networks, a subset of Liquid Neural Networks (LNNs), to attain elevated classification accuracy while considerably minimizing model complexity and the quantity of trainable parameters in comparison to conventional Recurrent Neural Networks (RNNs) such as LSTMs or GRUs.`;

  const datasetInfo = {
    'Dataset Name': 'UCI Human Activity Recognition (HAR) Using Smartphones Dataset',
    'Physical Input': 'Sensor signals (accelerometer and gyroscope) from 30 subjects.',
    'Signal Type': 'Multivariate time-series data.',
    'Feature Set': '561 features from time and frequency domain of 3-axial acceleration and angular velocity.',
    'Activity Labels': 'Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Lying.',
    'Data Window': '2.56-second segments with 50% overlap.',
  };

  const DashboardView = () => (
    <main className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-fade-in">
      <div className="lg:col-span-2 space-y-8">
        <ProjectInfoCard title="Project Objective" icon={<Target className="h-6 w-6 text-cyan-400" />}>
          <p className="text-gray-300 leading-relaxed">{projectObjective}</p>
        </ProjectInfoCard>

        <ProjectInfoCard title="Methodology Overview" icon={<BrainCircuit className="h-6 w-6 text-cyan-400" />}>
           <div className="mb-6">
                <h3 className="text-lg font-semibold text-white mb-4">Blueprint Architecture Diagram</h3>
                
           </div>
           <div className="border-t border-gray-700 pt-6">
               <h3 className="text-lg font-semibold text-white mb-4">Project Stages</h3>
               <MethodologyStepper />
           </div>
        </ProjectInfoCard>

        <ProjectInfoCard title="Results & Performance" icon={<BarChart4 className="h-6 w-6 text-cyan-400" />}>
            <div className="space-y-6 mb-8">
                <h3 className="text-lg font-semibold text-white">Visual Results from Report</h3>
                
            </div>
            <div className="border-t border-gray-700 pt-6">
                <h3 className="text-lg font-semibold text-white mb-4">Interactive Performance Dashboard</h3>
                <ResultsDashboard />
            </div>
        </ProjectInfoCard>
      </div>

      <div className="lg:col-span-1 space-y-8">
         <ProjectInfoCard title="Dataset Overview" icon={<BookOpen className="h-6 w-6 text-cyan-400" />}>
          <ul className="space-y-3">
            {Object.entries(datasetInfo).map(([key, value]) => (
              <li key={key}>
                <strong className="font-semibold text-cyan-400 block">{key}:</strong>
                <span className="text-gray-300">{value}</span>
              </li>
            ))}
          </ul>
        </ProjectInfoCard>

        <LnnExplainer />
      </div>
    </main>
  );

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 font-sans p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <Header
          title="Human Activity Recognition Using Liquid Neural Network (LNN)"
          teamMembers={teamMembers}
          currentView={view}
          setView={setView}
        />

        {view === 'dashboard' && <DashboardView />}
        {view === 'report' && <ProjectReport />}
        {view === 'codeExplanation' && <CodeExplanation />}
        {view === 'code' && <CodeExplainer />}
        
      </div>
    </div>
  );
};

export default App;