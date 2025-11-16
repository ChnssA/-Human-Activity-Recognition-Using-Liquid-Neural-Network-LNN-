import React from 'react';
import ProjectInfoCard from './ProjectInfoCard';
import ResultsDashboard from './ResultsDashboard';
import FormulaExplainer from './FormulaExplainer';
import { FileText, Microscope, FlaskConical, BarChart4, CheckSquare, Library, TestTube2 } from 'lucide-react';

const ProjectReport: React.FC = () => {
    return (
        <div className="space-y-8 animate-fade-in">
            <ProjectInfoCard title="Abstract" icon={<FileText className="h-6 w-6 text-cyan-400" />}>
                <p className="text-gray-300 leading-relaxed">
                    Human Activity Recognition (HAR) using data from wearable sensors is a foundational task in ubiquitous computing, with applications ranging from digital health monitoring to sports analytics. The primary challenge in this domain is the development of models that are not only accurate but also computationally efficient enough for real-time, on-device (edge) deployment. This project presents a comparative analysis between a traditional recurrent model, the Long Short-Term Memory (LSTM) network, and a bio-inspired, continuous-time model, the Liquid Time-Constant (LTC) network. We evaluate both architectures on the raw time-series data from the UCI HAR dataset. Our findings demonstrate that the LTC model achieves superior classification accuracy (92.16%) compared to the LSTM baseline (89.51%). More significantly, the LTC model does so with an 82.3% reduction in trainable parameters, proving its viability as a high-performance, low-resource architecture for next-generation wearable and IoT applications.
                </p>
            </ProjectInfoCard>

            <ProjectInfoCard title="Introduction" icon={<Microscope className="h-6 w-6 text-cyan-400" />}>
                <div className="space-y-4 text-gray-300 leading-relaxed">
                    <p>The ability to automatically recognize human physical activities has become a cornerstone of modern human-computer interaction. From tracking fitness goals to monitoring elderly patients for fall detection, the societal and medical benefits are immense. This capability is largely powered by deep learning models designed to interpret time-series data from embedded sensors like accelerometers and gyroscopes.</p>
                    <p>For years, Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks, have been the state-of-the-art. Their gating mechanisms allow them to capture long-term dependencies in temporal data, making them highly effective. However, this effectiveness comes at a high computational cost. These models are "heavy," possessing millions of parameters, which translates to high memory (RAM) usage and significant energy consumption. This computational burden makes them poorly suited for the primary frontier of HAR: real-time, continuous monitoring on resource-constrained edge devices like smartwatches or IoT sensors.</p>
                    <p>This report investigates a promising alternative: the Liquid Neural Network (LNN), specifically its implementation as a Liquid Time-Constant (LTC) network. Derived from the study of bio-inspired neural circuits (like that of the C. elegans worm) and formalized as a system of Neural Ordinary Differential Equations (Neural ODEs), LTCs operate in continuous time. This "liquid" architecture is inherently smaller, more flexible, and more robust to irregular or sparse data.</p>
                    <p>Our project's objective is to quantify this theorized advantage. We conduct a direct, empirical comparison between a baseline LSTM and an LTC model, training both on the raw, unprocessed 9-axis sensor data from the well-benchmarked UCI HAR dataset. We hypothesize that the LTC network will not only match or exceed the accuracy of the LSTM but will also do so with a fraction of the trainable parameters, thereby validating its suitability for the future of edge-based AI.</p>
                </div>
            </ProjectInfoCard>
            
            <ProjectInfoCard title="Methodology" icon={<FlaskConical className="h-6 w-6 text-cyan-400" />}>
                <div className="space-y-6 text-gray-300">
                    <h3 className="text-lg font-semibold text-cyan-400">3.1 Dataset Acquisition</h3>
                    <p>The foundation of this study is the UCI Human Activity Recognition (HAR) Using Smartphones Dataset. This dataset is a standard benchmark in the field. It consists of data from 30 volunteers performing six distinct activities: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying. Sensor data was collected from a smartphone's embedded accelerometer and gyroscope at a rate of 50Hz. The data was then segmented into 2.56-second sliding windows (128 timesteps) with 50% overlap.</p>

                    <h3 className="text-lg font-semibold text-cyan-400">3.2 Data Cleaning and Preprocessing</h3>
                    <p>The dataset is pre-cleaned and well-labeled. Our primary preprocessing step is Z-score Normalization (Standardization). This is a critical step, as the accelerometer (acc) and gyroscope (gyro) data are measured in different units and have different natural scales. Standardization rescales each of the 9 input features to have a mean (μ) of 0 and a standard deviation (σ) of 1.</p>
                    <FormulaExplainer formulaId="z-score" />

                    <h3 className="text-lg font-semibold text-cyan-400">3.3 End-to-End Feature Learning</h3>
                    <p>A key methodological decision is to avoid traditional feature engineering. We feed the raw, normalized 9-channel time-series data (shape: [samples, 128_timesteps, 9_features]) directly into our models. This forces the models to learn their own internal representations and relevant features from the raw temporal signals, a hallmark of modern deep learning.</p>

                    <h3 className="text-lg font-semibold text-cyan-400">3.4 Model Development and Implementation</h3>
                    <p className="font-semibold text-gray-200">3.4.1 Baseline: Long Short-Term Memory (LSTM)</p>
                    <p>The LSTM is our baseline, representing the industry standard. It solves the "vanishing gradient" problem of simple RNNs by introducing an internal cell state ($C_t$) and a series of "gates" that control the flow of information.</p>
                    <FormulaExplainer formulaId="lstm" />

                    <p className="font-semibold text-gray-200 mt-4">3.4.2 Challenger: Liquid Time-Constant (LTC) Network</p>
                    <p>The LTC network operates in continuous time by defining and solving a system of Neural Ordinary Differential Equations (Neural ODEs). Its hidden state $h(t)$ is defined by a differential equation.</p>
                    <FormulaExplainer formulaId="ltc" />

                    <h3 className="text-lg font-semibold text-cyan-400">3.5 Experimental Design</h3>
                    <p>Both models are trained to minimize a Cross-Entropy Loss, a standard for multi-class classification.</p>
                    <FormulaExplainer formulaId="cross-entropy" />
                    <p>We use the Adam (Adaptive Moment Estimation) optimizer, which adapts the learning rate for each weight individually, allowing for faster and more stable convergence.</p>

                    <h3 className="text-lg font-semibold text-cyan-400">3.6 Libraries and Tools Used</h3>
                    <ul className="list-disc list-inside grid grid-cols-2 md:grid-cols-3 gap-2">
                        <li>Python 3.x</li>
                        <li>PyTorch</li>
                        <li>NCPS (for LTC)</li>
                        <li>Scikit-learn</li>
                        <li>Pandas & NumPy</li>
                    </ul>
                </div>
            </ProjectInfoCard>
            
             <ProjectInfoCard title="Results and Analysis" icon={<BarChart4 className="h-6 w-6 text-cyan-400" />}>
                <div className="space-y-4 text-gray-300">
                    <p>The two models were trained for 15 epochs and evaluated on the held-out test set. The results are a clear confirmation of our hypothesis.</p>
                    <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse">
                            <caption className="text-lg font-semibold p-2 text-white">Table 1: Model Performance and Efficiency Comparison</caption>
                            <thead>
                                <tr className="bg-gray-700">
                                    <th className="p-3 border border-gray-600">Metric</th>
                                    <th className="p-3 border border-gray-600">Baseline (LSTM)</th>
                                    <th className="p-3 border border-gray-600">Challenger (LTC)</th>
                                    <th className="p-3 border border-gray-600">Change</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr className="bg-gray-800">
                                    <td className="p-3 border border-gray-600 font-semibold">Test Accuracy</td>
                                    <td className="p-3 border border-gray-600">89.51%</td>
                                    <td className="p-3 border border-gray-600">92.16%</td>
                                    <td className="p-3 border border-gray-600 text-green-400 font-bold">+2.65%</td>
                                </tr>
                                <tr className="bg-gray-800">
                                    <td className="p-3 border border-gray-600 font-semibold">Macro F1-Score</td>
                                    <td className="p-3 border border-gray-600">0.8932</td>
                                    <td className="p-3 border border-gray-600">0.9205</td>
                                    <td className="p-3 border border-gray-600 text-green-400 font-bold">+0.0273</td>
                                </tr>
                                <tr className="bg-gray-800">
                                    <td className="p-3 border border-gray-600 font-semibold">Trainable Parameters</td>
                                    <td className="p-3 border border-gray-600">31,830</td>
                                    <td className="p-3 border border-gray-600">5,622</td>
                                    <td className="p-3 border border-gray-600 text-green-400 font-bold">-82.34%</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <ResultsDashboard />
                </div>
             </ProjectInfoCard>

            <ProjectInfoCard title="Conclusion" icon={<CheckSquare className="h-6 w-6 text-cyan-400" />}>
                <div className="space-y-4 text-gray-300 leading-relaxed">
                    <p>This project successfully demonstrated the superior performance and efficiency of Liquid Time-Constant (LTC) networks over traditional LSTMs for the task of Human Activity Recognition. Our experiment proved our hypothesis: the LTC model achieved a higher accuracy (92.16%) than the LSTM (89.51%) while requiring 82.34% fewer parameters.</p>
                    <p>The results are unambiguous. The mathematical framework of Neural ODEs that underpins the LTC is not just a theoretical curiosity but a practical and powerful tool. By learning a continuous-time representation of the data, the LTC model is able to build a more compact, robust, and effective model of the underlying human movements.</p>
                    <p>The implications of this finding are significant. As the demand for real-time, "always-on" AI in wearables and IoT devices grows, the computational bottleneck of traditional deep learning becomes a primary constraint. Our results validate that LNNs are a premier architecture for overcoming this barrier.</p>
                    <p><strong className="text-cyan-400">Future work:</strong> Focus on deploying this LTC model on a physical device (e.g., a Raspberry Pi or an Android smartphone) to empirically measure its latency and power consumption in situ, further solidifying the real-world advantages of this efficient and powerful architecture.</p>
                </div>
            </ProjectInfoCard>

        </div>
    );
};

export default ProjectReport;
