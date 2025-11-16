import React from 'react';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, Check, Zap, Clock } from 'lucide-react';

const resultsData = [
  {
    name: 'Test Accuracy',
    Baseline: 89.51,
    'LNN (LTC)': 92.16,
    unit: '%',
  },
  {
    name: 'F1-Score (Macro)',
    Baseline: 0.893,
    'LNN (LTC)': 0.921,
    unit: '',
  },
];

const parameterData = [
    { name: 'LSTM Parameters', value: 31830 },
    { name: 'LTC Parameters', value: 5622 },
];
const PARAM_COLORS = ['#3182CE', '#38B2AC'];


const outcomes = [
    {
        icon: <TrendingUp className="h-6 w-6 text-green-400"/>,
        metric: 'Superior Accuracy',
        reason: 'LTC outperformed LSTM by 2.65% on the test set.'
    },
    {
        icon: <Zap className="h-6 w-6 text-yellow-400"/>,
        metric: 'Radical Efficiency',
        reason: 'Achieved higher accuracy with an 82.3% reduction in parameters.'
    },
    {
        icon: <Check className="h-6 w-6 text-blue-400"/>,
        metric: 'Robust Performance',
        reason: 'Higher Macro F1-Score indicates better balance across all activity classes.'
    },
    {
        icon: <Clock className="h-6 w-6 text-purple-400"/>,
        metric: 'Edge-Ready',
        reason: 'Low parameter count makes it ideal for real-time, on-device deployment.'
    }
]

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const unit = payload[0]?.payload?.unit || '';
    return (
      <div className="bg-gray-800 border border-gray-600 p-3 rounded-lg shadow-xl">
        <p className="font-bold text-cyan-400">{label}</p>
        {payload.map((entry: any) => (
            <p key={entry.name} style={{ color: entry.color }}>
                {`${entry.name}: ${entry.value.toLocaleString()}${unit}`}
            </p>
        ))}
      </div>
    );
  }
  return null;
};

const ResultsDashboard: React.FC = () => {
  return (
    <div className="space-y-8">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
            <div className="lg:col-span-3">
                <h3 className="text-lg font-semibold mb-4 text-white">Performance Metrics Comparison</h3>
                <div style={{ width: '100%', height: 250 }}>
                    <ResponsiveContainer>
                    <BarChart data={resultsData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#4A5568" />
                        <XAxis dataKey="name" stroke="#A0AEC0" fontSize={12} />
                        <YAxis stroke="#A0AEC0" fontSize={12} tickFormatter={(value) => typeof value === 'number' ? value.toLocaleString() : value} />
                        <Tooltip content={<CustomTooltip />} cursor={{fill: 'rgba(100, 116, 139, 0.1)'}}/>
                        <Legend wrapperStyle={{fontSize: "14px"}}/>
                        <Bar dataKey="Baseline" fill="#3182CE" />
                        <Bar dataKey="LNN (LTC)" fill="#38B2AC" />
                    </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
             <div className="lg:col-span-2">
                <h3 className="text-lg font-semibold mb-4 text-white">Parameter Efficiency</h3>
                 <div style={{ width: '100%', height: 250 }} className="relative">
                    <ResponsiveContainer>
                        <PieChart>
                            <Pie
                                data={parameterData}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                fill="#8884d8"
                                paddingAngle={5}
                                dataKey="value"
                                labelLine={false}
                            >
                                {parameterData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={PARAM_COLORS[index % PARAM_COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip formatter={(value: number) => value.toLocaleString()} />
                            <Legend wrapperStyle={{fontSize: "14px"}}/>
                        </PieChart>
                    </ResponsiveContainer>
                    <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                        <span className="text-3xl font-bold text-green-400">-82.3%</span>
                        <span className="text-sm text-gray-300">Reduction</span>
                    </div>
                 </div>
            </div>
        </div>
        <div>
            <h3 className="text-lg font-semibold mb-4 text-white">Key Outcomes & Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {outcomes.map((outcome, index) => (
                    <div key={index} className="bg-gray-700/50 p-4 rounded-lg flex items-start space-x-4">
                        <div className="flex-shrink-0">{outcome.icon}</div>
                        <div>
                            <p className="font-semibold text-gray-200">{outcome.metric}</p>
                            <p className="text-sm text-gray-400">{outcome.reason}</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    </div>
  );
};

export default ResultsDashboard;
