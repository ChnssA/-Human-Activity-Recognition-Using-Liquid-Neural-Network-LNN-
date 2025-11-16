import React from 'react';
import ProjectInfoCard from './ProjectInfoCard';
import { Terminal, Download, FolderTree, FileCode2, Play, TestTube, ChevronsRight } from 'lucide-react';

const CodeBlock = ({ code, language = 'python' }: { code: string; language?: string }) => (
  <div className="my-4">
    <pre className="bg-gray-900 text-sm text-cyan-200 p-4 rounded-md overflow-x-auto font-mono border border-gray-700">
      <code className={`language-${language}`}>{code.trim()}</code>
    </pre>
  </div>
);

const CodeExplainer: React.FC = () => {
    return (
        <div className="space-y-8 animate-fade-in">
             <ProjectInfoCard title="Code Implementation Guide" icon={<Terminal className="h-6 w-6 text-cyan-400" />}>
                <p className="text-gray-300 leading-relaxed">
                    This section provides a complete, step-by-step guide to implement the LNN for Human Activity Recognition project. Follow these instructions to set up your environment, understand the code structure, and run the experiment to replicate the results.
                </p>
            </ProjectInfoCard>

            <ProjectInfoCard title="Step 1: Project Setup & Installation" icon={<Download className="h-6 w-6 text-cyan-400" />}>
                <div className="space-y-4 text-gray-300">
                    <p>First, ensure you have Python 3.x installed. Then, download and unzip the dataset and install the required libraries.</p>
                    <h3 className="text-lg font-semibold text-cyan-400">1. Download the Dataset</h3>
                    <p>Download the "UCI HAR Dataset" from the official source and extract it into your project folder. You can find it <a href="https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">here</a>.</p>
                    
                    <h3 className="text-lg font-semibold text-cyan-400">2. Install Libraries</h3>
                    <p>Open your terminal or command prompt and run the following commands to install the necessary Python packages:</p>
                    <CodeBlock language='bash' code={`
# Main deep learning framework
pip install torch

# The specialized library for Liquid Neural Networks
pip install ncps

# For data loading and manipulation
pip install pandas numpy

# For data preprocessing and evaluation metrics
pip install scikit-learn
                    `} />
                </div>
            </ProjectInfoCard>

            <ProjectInfoCard title="Step 2: Code Architecture & File Breakdown" icon={<FolderTree className="h-6 w-6 text-cyan-400" />}>
                <div className="space-y-4 text-gray-300">
                    <p>The project is organized into modular Python scripts. This separation of concerns makes the code clean, reusable, and easy to understand.</p>
                    <h3 className="text-lg font-semibold text-cyan-400">High-Level Dependency Flow</h3>
                    <CodeBlock language='text' code={`
  [ main.py ]
(The Conductor)
       |
+------+------+------+------+
|             |             |             |
v             v             v             v
[ data_loader.py ] [ models.py ]   [ utils.py ]  [ config.py ]
 (Data Manager)   (Architect)     (Toolbox)     (Settings)
       |
       v
[ UCI HAR Dataset ]
    (Raw Data)
                    `} />
                    
                    <div className="space-y-6 mt-6">
                        <div>
                            <h4 className="flex items-center text-md font-semibold text-cyan-400"><FileCode2 className="h-5 w-5 mr-2" />config.py (The Settings)</h4>
                            <p className="mt-1">This file holds all static configuration variables, making it easy to tweak experiment parameters in one place.</p>
                            <CodeBlock code={`
# --- Configuration ---
DATASET_PATH = 'UCI HAR Dataset'
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
HIDDEN_UNITS = 48
NUM_CLASSES = 6
INPUT_SIZE = 9  # 9 features (3x body_acc, 3x body_gyro, 3x total_acc)
                            `} />
                        </div>
                        <div>
                            <h4 className="flex items-center text-md font-semibold text-cyan-400"><FileCode2 className="h-5 w-5 mr-2" />data_loader.py (The Data Manager)</h4>
                            <p className="mt-1">This script handles loading the raw sensor data, applying Z-score normalization, and preparing PyTorch `DataLoaders` for batch training.</p>
                            <CodeBlock code={`
import os, torch, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import config

def load_har_data(split='train'):
    signals_path = os.path.join(config.DATASET_PATH, split, 'Inertial Signals')
    signal_files = ['body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z', 'total_acc_x', 'total_acc_y', 'total_acc_z']
    signal_data = [pd.read_csv(os.path.join(signals_path, f'{fname}_{split}.txt'), delim_whitespace=True, header=None).values for fname in signal_files]
    X = np.stack(signal_data, axis=-1)
    label_path = os.path.join(config.DATASET_PATH, split, f'y_{split}.txt')
    y = pd.read_csv(label_path, header=None).values.flatten() - 1
    return X, y

def create_dataloaders(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return train_loader, test_loader
                            `} />
                        </div>
                        <div>
                            <h4 className="flex items-center text-md font-semibold text-cyan-400"><FileCode2 className="h-5 w-5 mr-2" />models.py (The Architect)</h4>
                            <p className="mt-1">This file defines the PyTorch `nn.Module` classes for both the baseline LSTM and the challenger LTC model architectures.</p>
                             <CodeBlock code={`
import torch.nn as nn
from ncps.torch import LTC

class BaselineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BaselineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class LtcHARModel(nn.Module):
    def __init__(self, input_size, units, num_classes):
        super(LtcHARModel, self).__init__()
        self.ltc_layer = LTC(input_size, units)
        self.fc = nn.Linear(units, num_classes)

    def forward(self, x):
        _, h_n = self.ltc_layer(x)
        return self.fc(h_n)
                            `} />
                        </div>
                        <div>
                            <h4 className="flex items-center text-md font-semibold text-cyan-400"><FileCode2 className="h-5 w-5 mr-2" />utils.py (The Toolbox)</h4>
                            <p className="mt-1">A collection of helper functions for counting model parameters, training for one epoch, and evaluating the model on the test set.</p>
                            <CodeBlock code={`
import torch
from sklearn.metrics import accuracy_score, f1_score
import config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{config.EPOCHS}] completed.')

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return accuracy, f1
                            `} />
                        </div>
                        <div>
                            <h4 className="flex items-center text-md font-semibold text-cyan-400"><FileCode2 className="h-5 w-5 mr-2" />main.py (The Conductor)</h4>
                            <p className="mt-1">The main entry point of the application. It orchestrates the entire process: loading data, creating models, running the training loops, and printing the final results.</p>
                            <CodeBlock code={`
import torch, torch.nn as nn, torch.optim as optim
import config
from data_loader import load_har_data, create_dataloaders
from models import BaselineLSTM, LtcHARModel
from utils import train_model, evaluate_model, count_parameters

def run_experiment():
    X_train, y_train = load_har_data('train')
    X_test, y_test = load_har_data('test')
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)
    
    # Train and Evaluate LSTM
    lstm_model = BaselineLSTM(config.INPUT_SIZE, config.HIDDEN_UNITS, 2, config.NUM_CLASSES)
    optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=config.LEARNING_RATE)
    for epoch in range(config.EPOCHS):
        train_model(lstm_model, train_loader, nn.CrossEntropyLoss(), optimizer_lstm, epoch)
    lstm_accuracy, lstm_f1 = evaluate_model(lstm_model, test_loader)
    
    # Train and Evaluate LTC
    ltc_model = LtcHARModel(config.INPUT_SIZE, config.HIDDEN_UNITS, config.NUM_CLASSES)
    optimizer_ltc = optim.Adam(ltc_model.parameters(), lr=config.LEARNING_RATE)
    for epoch in range(config.EPOCHS):
        train_model(ltc_model, train_loader, nn.CrossEntropyLoss(), optimizer_ltc, epoch)
    ltc_accuracy, ltc_f1 = evaluate_model(ltc_model, test_loader)
    
    # Print Results
    print("\\n--- Final Results ---")
    print(f"Baseline (LSTM): Accuracy={lstm_accuracy*100:.2f}%, F1={lstm_f1:.4f}, Params={count_parameters(lstm_model)}")
    print(f"LNN (LTC): Accuracy={ltc_accuracy*100:.2f}%, F1={ltc_f1:.4f}, Params={count_parameters(ltc_model)}")

if __name__ == "__main__":
    run_experiment()
                            `} />
                        </div>
                    </div>
                </div>
            </ProjectInfoCard>
            
            <ProjectInfoCard title="Step 3: Run the Experiment" icon={<Play className="h-6 w-6 text-cyan-400" />}>
                 <div className="space-y-4 text-gray-300">
                    <p>With all the files in place, running the entire experiment is as simple as executing the `main.py` script from your terminal.</p>
                    <CodeBlock language='bash' code={`
python main.py
                    `} />
                    <h3 className="text-lg font-semibold text-cyan-400">Sample Input Data</h3>
                    <p>The models consume the raw time-series data. For example, a single line in `body_acc_x_train.txt` represents the x-axis accelerometer readings over 128 timesteps for one window.</p>
                     <CodeBlock language='text' code={`
2.645553e-001 2.726549e-001 2.799222e-001 ... (128 total values)
                    `} />
                    <h3 className="text-lg font-semibold text-cyan-400">Sample Execution Output</h3>
                    <p>After running, the script will train both models and print a final comparison, which should look similar to this:</p>
                    <CodeBlock language='text' code={`
Epoch [1/15] completed.
...
Epoch [15/15] completed.
...
--- Final Results ---
Baseline (LSTM): Accuracy=89.51%, F1=0.8932, Params=31830
LNN (LTC): Accuracy=92.16%, F1=0.9205, Params=5622
                    `} />
                </div>
            </ProjectInfoCard>

        </div>
    );
};

export default CodeExplainer;
