import React from 'react';
import ProjectInfoCard from './ProjectInfoCard';
import { BookText } from 'lucide-react';

const CodeBlock = ({ code, language = 'bash' }: { code: string; language?: string }) => (
  <div className="my-4">
    <pre className="bg-gray-900 text-sm text-cyan-200 p-4 rounded-md overflow-x-auto font-mono border border-gray-700">
      <code className={`language-${language}`}>{code.trim()}</code>
    </pre>
  </div>
);


const CodeExplanation: React.FC = () => {
    return (
        <div className="space-y-8 animate-fade-in">
            <ProjectInfoCard title="Project Implementation Plan" icon={<BookText className="h-6 w-6 text-cyan-400" />}>
                <div className="space-y-4 text-gray-300 leading-relaxed">
                    <p>This is a comprehensive step-by-step project implementation plan for <strong>Human Activity Recognition (HAR) Using Liquid Neural Networks (LNN)</strong>, structured for real-time application using the UCI HAR Dataset.</p>

                    <h2 className="text-xl font-semibold text-cyan-400 pt-4">1. 📂 Project Setup and File Structure</h2>
                    <p>Your requested file structure forms the basis, which we'll expand to support model training and real-time inference.</p>
                    <CodeBlock language="text" code={`
LNN_HAR_Project/
├── venv/                       # Python Virtual Environment
├── UCI HAR Dataset/            # The downloaded and unzipped dataset
│   ├── activity_labels.txt
│   ├── features.txt
│   ├── features_info.txt
│   ├── README.txt
│   ├── test/
│   └── train/
├── src/                        # Core source code
│   ├── data_processor.py       # For reading, cleaning, and windowing the data
│   ├── lnn_model.py            # LNN architecture definition
│   ├── train.py                # Script for model training
│   ├── predict.py              # Script for real-time prediction
│   └── utils.py                # Helper functions (e.g., plot metrics)
├── models/                     # Trained models are saved here
│   └── lnn_har_model.h5
├── requirements.txt            # List of required Python packages
├── config.yaml                 # Configuration file (hyperparameters, paths)
└── README.md                   # Project documentation
                    `} />
                    
                    <h3 className="text-lg font-semibold text-cyan-400">Initial Steps</h3>
                    <ol className="list-decimal list-inside space-y-2">
                        <li>
                            <strong>Create Project Directory and Virtual Environment:</strong>
                            <CodeBlock language="bash" code={`
mkdir LNN_HAR_Project
cd LNN_HAR_Project
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
                            `} />
                        </li>
                        <li>
                            <strong>Install Dependencies:</strong> Create <code>requirements.txt</code> with necessary packages (e.g., <code>numpy</code>, <code>pandas</code>, <code>scikit-learn</code>, <code>tensorflow</code>/<code>keras</code>, <code>tflite</code>, a specific LNN library if available, or custom LNN implementation).
                             <CodeBlock language="bash" code={`
pip install -r requirements.txt
                             `} />
                        </li>
                        <li><strong>Place Dataset:</strong> Download and unzip the <strong>UCI HAR Dataset</strong> into the <code>UCI HAR Dataset/</code> folder.</li>
                    </ol>

                    <h2 className="text-xl font-semibold text-cyan-400 pt-4">2. ⚙️ Data Preprocessing (Inside <code>src/data_processor.py</code>)</h2>
                    <p>The UCI HAR Dataset is pre-processed, but you'll need to re-format it for the LNN's time-series nature.</p>
                     <ol className="list-decimal list-inside space-y-2">
                        <li><strong>Load Data and Metadata:</strong> Read feature names (<code>features.txt</code>), activity labels (<code>activity_labels.txt</code>), and the time-series data from <code>train/</code> and <code>test/</code> folders.</li>
                        <li>
                            <strong>Segmentation/Windowing (If necessary):</strong> The UCI HAR data is already windowed (at 2.56 seconds with 50% overlap). For LNNs, which excel at continuous time-series, you might process the raw sequential data files directly if available, or treat the existing windows as a sequence of feature vectors.
                            <ul className="list-disc list-inside ml-4 mt-2">
                                <li><strong>LNN Input:</strong> An LNN expects a time series input of shape <code>(batch_size, sequence_length, n_features)</code>. You'll load the data such that a sequence of sensor readings is the input.</li>
                            </ul>
                        </li>
                        <li><strong>Normalization/Scaling:</strong> The sensor readings (features) should be normalized (e.g., using <strong>MinMaxScaler</strong> or <strong>StandardScaler</strong>) across the training set and then applied to the test set.</li>
                        <li><strong>Label Encoding:</strong> Convert activity labels (e.g., 'WALKING', 'SITTING') into numerical/one-hot encoded vectors.</li>
                     </ol>

                    <h2 className="text-xl font-semibold text-cyan-400 pt-4">3. 🧠 Liquid Neural Network Model (Inside <code>src/lnn_model.py</code>)</h2>
                    <p>Since there isn't one universal LNN library, you'll typically use a custom layer or an implementation based on <strong>Neural ODEs</strong> (Ordinary Differential Equations) or <strong>Closed-form Continuous-Time RNNs</strong>.</p>
                     <ol className="list-decimal list-inside space-y-2">
                        <li><strong>Define LNN Layer:</strong> Implement the core Liquid Neural Network layer, often a variation of the <strong>Closed-form Continuous-Time (CfC) RNN</strong>.</li>
                        <li>
                            <strong>Build Model Architecture:</strong>
                             <ul className="list-disc list-inside ml-4 mt-2">
                                <li><strong>Input Layer:</strong> <code>(sequence_length, n_features)</code></li>
                                <li><strong>LNN Layer(s):</strong> The core recurrent layer (e.g., CfC-RNN) to learn temporal dynamics. This is the key difference from standard RNNs (LSTM/GRU).</li>
                                <li><strong>Dense Output Layer:</strong> A fully connected layer with a <strong>Softmax</strong> activation function to classify the activity into one of the N classes.</li>
                                <li><strong>Compilation:</strong> Compile the model using a suitable loss function (e.g., <code>categorical_crossentropy</code>) and an optimizer (e.g., <code>Adam</code>).</li>
                            </ul>
                        </li>
                    </ol>

                    <h2 className="text-xl font-semibold text-cyan-400 pt-4">4. 🚀 Model Training (Inside <code>src/train.py</code>)</h2>
                     <ol className="list-decimal list-inside space-y-2">
                        <li><strong>Instantiate:</strong> Load the preprocessed data and the LNN model.</li>
                        <li>
                            <strong>Training Loop:</strong> Train the model on the training data. Given LNNs are recurrent, they are suited for batch training.
                            <CodeBlock language="python" code={`
# Example training logic
model.fit(
    X_train, y_train,
    epochs=hyperparams['epochs'],
    batch_size=hyperparams['batch_size'],
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)
                            `} />
                        </li>
                        <li><strong>Save Model:</strong> Save the trained model weights and architecture to the <code>models/</code> directory.</li>
                    </ol>
                    
                    <h2 className="text-xl font-semibold text-cyan-400 pt-4">5. 🔬 Real-Time Inference Preparation</h2>
                    <p>To prepare for real-time, the model needs to be optimized for low-latency predictions, often on a resource-constrained device.</p>
                     <ol className="list-decimal list-inside space-y-2">
                        <li>
                            <strong>Optimization:</strong> Convert the trained Keras/TensorFlow model to a format optimized for deployment, such as <strong>TensorFlow Lite (TFLite)</strong>. This significantly reduces model size and latency.
                            <CodeBlock language="python" code={`
# Example TFLite conversion
converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
tflite_model = converter.convert()
with open('models/lnn_har_model.tflite', 'wb') as f:
    f.write(tflite_model)
                            `} />
                        </li>
                         <li>
                            <strong>Windowing Function:</strong> The prediction script (<code>src/predict.py</code>) must implement the *exact same* data sampling and preprocessing logic used during training.
                            <ul className="list-disc list-inside ml-4 mt-2">
                                <li><strong>Sliding Window:</strong> For real-time input (e.g., from an accelerometer stream), you'll constantly sample the last <code>sequence_length</code> points and apply the original <strong>MinMaxScaler</strong> or <strong>StandardScaler</strong> to this window.</li>
                            </ul>
                        </li>
                    </ol>

                    <h2 className="text-xl font-semibold text-cyan-400 pt-4">6. 🌐 Real-Time Deployment and Output</h2>
                    <p>The final step is the <strong>real-time implementation</strong>, where a continuous stream of sensor data is fed into the model.</p>
                     <ol className="list-decimal list-inside space-y-2">
                        <li>
                            <strong>Simulate Real-Time Input:</strong> In a real application, this would be a <strong>live stream from a sensor</strong> (e.g., a smartphone's accelerometer/gyroscope) via an API or Bluetooth. For this project, you can:
                             <ul className="list-disc list-inside ml-4 mt-2">
                                <li>Use a continuous segment of the *original raw* UCI HAR data that wasn't used for training/testing.</li>
                                <li>Simulate a sensor reading loop that reads data points sequentially.</li>
                            </ul>
                        </li>
                        <li>
                            <strong>Prediction Loop (Inside <code>src/predict.py</code>):</strong>
                            <ul className="list-disc list-inside ml-4 mt-2">
                                <li><strong>Data Acquisition:</strong> Continuously collect a new data point.</li>
                                <li><strong>Sliding Window:</strong> Maintain a buffer of the last <code>sequence_length</code> data points.</li>
                                <li><strong>Preprocess:</strong> Normalize the buffer using the saved scaler parameters.</li>
                                <li><strong>Inference:</strong> Feed the preprocessed window into the loaded <strong>TFLite/LNN model</strong> to get the activity probability distribution.</li>
                                <li><strong>Output:</strong> The script returns the predicted activity label (the class with the highest probability).</li>
                            </ul>
                            <CodeBlock language="python" code={`
# Real-time pseudocode in predict.py
# 1. Load scaler and TFLite model
# 2. Initialize a buffer/queue for the sliding window
# 3. while True:
#        new_sensor_data = read_live_sensor()
#        window_buffer.append(new_sensor_data)
#        if len(window_buffer) == sequence_length:
#            processed_window = scaler.transform(window_buffer)
#            prediction = tflite_model.predict(processed_window)
#            activity_label = decode_label(np.argmax(prediction))
#            print(f"Predicted Activity: {activity_label}")
#            # Slide the window forward (e.g., remove the oldest 50% for overlap)
                            `} />
                        </li>
                    </ol>
                    
                    <div className="border-t border-gray-700 pt-4 mt-4">
                        <p>The video below discusses Human Activity Recognition using a combined Convolutional and Long-Short Term Memory network, which provides a good conceptual framework for handling time-series data like in this project, which you can adapt for the LNN.</p>
                        <p className="mt-2">This video is relevant as it provides a guide to implementing an activity recognition system using a recurrent network (CNN+LSTM), which shares the core data processing and time-series modeling challenges with a project using Liquid Neural Networks.</p>
                        <div className="mt-4">
                            <a href="https://www.youtube.com/watch?v=QmtSkq3DYko" target="_blank" rel="noopener noreferrer" className="block bg-gray-900/50 hover:bg-gray-800 transition-colors p-4 rounded-lg border border-gray-600">
                                <p className="font-semibold text-blue-400">Human Activity Recognition using TensorFlow (CNN + LSTM) | 2 Methods</p>
                                <img src="http://googleusercontent.com/youtube_content/0" alt="YouTube video thumbnail" className="mt-2 rounded-md w-full max-w-sm"/>
                            </a>
                        </div>
                    </div>
                </div>
            </ProjectInfoCard>
        </div>
    );
};
export default CodeExplanation;