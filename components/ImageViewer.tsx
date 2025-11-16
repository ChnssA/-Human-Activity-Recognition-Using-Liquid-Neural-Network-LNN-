import React from 'react';
import ProjectInfoCard from './ProjectInfoCard';
import { Image } from 'lucide-react';

const images = [
  {
    src: '/Blueprint%20Architecture%20Diagram%20Human%20Activity%20Recogrininion%20(HAR)%20Methrolology.png',
    title: 'Blueprint Architecture Diagram: Human Activity Recognition (HAR) Methodology',
    alt: 'A diagram showing the end-to-end methodology for the HAR project, from data acquisition to model validation.'
  },
  {
    src: '/Figure%204.1%20Model%20Test%20Accuracy%20Comparison.png',
    title: 'Figure 4.1: Model Test Accuracy Comparison',
    alt: 'Bar chart comparing the test accuracy of the Baseline (LSTM) model at 89.51% and the Challenger (LTC) model at 92.16%.'
  },
  {
    src: '/Figure%204.2%20Model%20Macro%20F1-Score%20Comparison.png',
    title: 'Figure 4.2: Model Macro F1-Score Comparison',
    alt: 'Bar chart comparing the Macro F1-Score of the Baseline (LSTM) at 0.8932 and the Challenger (LTC) at 0.9205.'
  },
  {
    src: '/Figure%204.3%20Model%20Efficiency%20Comparison%20(Trainable%20Parameters).png',
    title: 'Figure 4.3: Model Efficiency Comparison (Trainable Parameters)',
    alt: 'Bar chart comparing the trainable parameters of the Baseline (LSTM) with 31,830 and the Challenger (LTC) with 5,622, showing an 82.34% reduction.'
  },
];

const ImageViewer: React.FC = () => {
  return (
    <div className="animate-fade-in">
      <ProjectInfoCard title="Project Images & Figures" icon={<Image className="h-6 w-6 text-cyan-400" />}>
        <p className="text-gray-300 mb-6">A collection of key figures and diagrams from the project report, visualizing the architecture, methodology, and comparative results.</p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {images.map((image, index) => (
            <div key={index} className="bg-gray-900/50 border border-gray-700 rounded-lg overflow-hidden shadow-md">
              <img src={image.src} alt={image.alt} className="w-full h-auto object-contain bg-white p-2" />
              <div className="p-4">
                <h3 className="font-semibold text-gray-200">{image.title}</h3>
              </div>
            </div>
          ))}
        </div>
      </ProjectInfoCard>
    </div>
  );
};

export default ImageViewer;
