
import React from 'react';
import { Database, Zap, Scaling, Orbit, CheckCircle } from 'lucide-react';

const methodologySteps = [
  {
    title: 'A. Data Preprocessing',
    icon: <Database className="h-8 w-8 text-cyan-400" />,
    points: [
      'Load and Segmentation: Load pre-segmented feature vectors and activity labels.',
      'Normalization: Apply Min-Max or Z-score normalization to all 561 features.',
      'Reshaping: Reshape data into sequence format (Samples, Timesteps, Features) for RNNs/LNNs.',
    ],
  },
  {
    title: 'B. Model Implementation',
    icon: <Zap className="h-8 w-8 text-cyan-400" />,
    points: [
      'Baseline Model (LSTM/GRU): Design a 2-layer LSTM as a performance benchmark.',
      'LNN Model (LTC Network): Implement the Liquid Time-Constant network using ODEs to model state evolution.',
      'Optimization: Train both models using Adam optimizer and categorical cross-entropy loss.',
    ],
  },
  {
    title: 'C. Evaluation',
    icon: <Scaling className="h-8 w-8 text-cyan-400" />,
    points: [
      'Data Split: Use the same data splits (e.g., 70% train, 30% test) for both models.',
      'Comparison: Evaluate LNN against the LSTM/GRU baseline on accuracy, parameter efficiency, and robustness.',
    ],
  },
    {
    title: 'D. Experiment Objective',
    icon: <CheckCircle className="h-8 w-8 text-cyan-400" />,
    points: [
        'Accuracy Target: Achieve ≥90.0% accuracy on the test set.',
        'Parameter Efficiency: Decrease trainable parameters by at least 50% compared to baseline.',
        'Robustness: Validate more consistent training and faster convergence for the LNN model.',
    ]
  }
];

const MethodologyStepper: React.FC = () => {
  return (
    <div className="space-y-6">
      {methodologySteps.map((step, index) => (
        <div key={index} className="flex items-start space-x-4">
          <div className="flex flex-col items-center">
            <div className="bg-gray-700 rounded-full p-3 border-2 border-cyan-500">
              {step.icon}
            </div>
            {index < methodologySteps.length - 1 && (
              <div className="w-0.5 h-16 bg-gray-600 mt-2"></div>
            )}
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-white">{step.title}</h3>
            <ul className="mt-2 space-y-2 text-gray-300 list-disc list-inside">
              {step.points.map((point, pIndex) => (
                <li key={pIndex}>{point}</li>
              ))}
            </ul>
          </div>
        </div>
      ))}
    </div>
  );
};

export default MethodologyStepper;
