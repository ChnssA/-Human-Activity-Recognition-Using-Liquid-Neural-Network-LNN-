import React, { useState } from 'react';
import { ChevronRight, RefreshCw, X } from 'lucide-react';

interface FormulaExplainerProps {
  formulaId: 'z-score' | 'lstm' | 'ltc' | 'cross-entropy';
}

const FormulaExplainer: React.FC<FormulaExplainerProps> = ({ formulaId }) => {
  const [step, setStep] = useState(0);

  const explanations = {
    'z-score': {
      title: 'Formula 1: Z-score Normalization',
      formula: <>z = (x - &mu;) / &sigma;</>,
      steps: [
        { term: 'z', explanation: 'The new, normalized value. This is the output.' },
        { term: 'x', explanation: 'The original data point from the dataset.' },
        { term: '&mu; (mu)', explanation: 'The mean (average) of all data points in the feature\'s column.' },
        { term: '&sigma; (sigma)', explanation: 'The standard deviation of all data points in the feature\'s column.' },
        { term: 'Purpose', explanation: 'This process rescales features to have a mean of 0 and a standard deviation of 1, ensuring no single feature dominates the learning process due to its scale.' }
      ]
    },
    'lstm': {
        title: 'LSTM Gate Equations',
        formula: <>f<sub>t</sub> = &sigma;(W<sub>f</sub>[h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>f</sub>)<br/>C<sub>t</sub> = f<sub>t</sub> &odot; C<sub>t-1</sub> + i<sub>t</sub> &odot; &Ctilde;<sub>t</sub></>,
        steps: [
            { term: 'f_t', explanation: 'Forget Gate: Decides what old information to discard from the cell state. A sigmoid function σ outputs 0 ("forget") or 1 ("keep").' },
            { term: 'i_t', explanation: 'Input Gate: Decides what new information from the input xt to store in the cell state.' },
            { term: 'C_t', explanation: 'Cell State: The internal "memory" of the LSTM. It\'s updated by forgetting old info (ft ⊙ Ct-1) and adding new info (it ⊙ C̃t).' },
            { term: 'h_t', explanation: 'Hidden State: The output for the current timestep, which is a filtered version of the cell state.' },
            { term: 'W, b', explanation: 'Weights (W) and biases (b) are the learnable parameters of the model.' }
        ]
    },
    'ltc': {
        title: 'Formula 2: The LTC Differential Equation',
        formula: <>&part;h(t)/&part;t = -1/&tau;(x<sub>t</sub>, h<sub>t</sub>) &sdot; (h(t) - A(x<sub>t</sub>, h<sub>t</sub>))</>,
        steps: [
            { term: '∂h(t)/∂t', explanation: 'The instantaneous rate of change of the hidden state h at time t. This is the core of the ODE.' },
            { term: 'h(t)', explanation: 'The hidden state vector of the model at continuous time t.' },
            { term: 'A(...)', explanation: 'An "attraction state" that the hidden state is pulled towards. This is defined by a small neural network.' },
            { term: 'τ(...)', explanation: 'The Time-Constant. This is the "liquid" part. It\'s another small neural network that dynamically controls how fast h(t) moves towards A, based on the current input xt and state ht.' },
            { term: 'Concept', explanation: 'Instead of discrete updates, the LTC model learns the continuous "physics" of how its state should evolve over time, making it highly flexible and efficient.' }
        ]
    },
    'cross-entropy': {
        title: 'Formula 3: Cross-Entropy Loss (L)',
        formula: <>L = -&sum;<sub>c=1</sub><sup>M</sup> y<sub>c</sub> log(p<sub>c</sub>)</>,
        steps: [
            { term: 'L', explanation: 'The Loss value. The goal of training is to make this value as small as possible.' },
            { term: 'Σ', explanation: 'A summation over all possible classes (c=1 to M).' },
            { term: 'M', explanation: 'The total number of classes (in this project, M=6).' },
            { term: 'y_c', explanation: 'A binary indicator (0 or 1). It\'s 1 if c is the correct class, and 0 otherwise.' },
            { term: 'p_c', explanation: 'The model\'s predicted probability that the input belongs to class c.' },
            { term: 'log(p_c)', explanation: 'The logarithm of the predicted probability. The loss increases dramatically as the model predicts a low probability for the correct class.' },
        ]
    }
  };

  const current = explanations[formulaId];
  const totalSteps = current.steps.length;

  const handleNext = () => setStep(s => Math.min(s + 1, totalSteps));
  const handleReset = () => setStep(0);

  const isFinished = step === totalSteps;

  return (
    <div className="bg-gray-700/50 border border-gray-600 rounded-lg p-4 mt-4 space-y-4">
      <h4 className="font-semibold text-gray-200">{current.title}</h4>
      <div className="bg-gray-900 p-4 rounded-md text-center">
        <p className="font-mono text-cyan-300 text-lg" style={{ lineHeight: '1.75' }}>{current.formula}</p>
      </div>

      <div className="min-h-[6rem] bg-gray-800 p-3 rounded-md transition-all duration-300">
        {step > 0 ? (
          <div className="animate-fade-in">
            <p className="font-bold text-cyan-400 text-lg">{current.steps[step - 1].term}</p>
            <p className="text-gray-300">{current.steps[step - 1].explanation}</p>
          </div>
        ) : (
             <p className="text-gray-400 italic">Click "Next Step" to begin the explanation.</p>
        )}
      </div>

      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-400">Step: {step} / {totalSteps}</span>
        <div className="flex gap-2">
            <button onClick={handleReset} className="p-2 bg-gray-600 hover:bg-gray-500 rounded-full transition-colors disabled:opacity-50" title="Reset Explanation" disabled={step === 0}>
                <RefreshCw className="h-4 w-4" />
            </button>
            <button
                onClick={handleNext}
                disabled={isFinished}
                className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg font-semibold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
                {isFinished ? 'Completed' : 'Next Step'} <ChevronRight className="h-4 w-4" />
            </button>
        </div>
      </div>
    </div>
  );
};

export default FormulaExplainer;
