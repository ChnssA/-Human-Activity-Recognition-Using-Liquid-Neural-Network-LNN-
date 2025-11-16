
import React, { useState, useRef, useEffect } from 'react';
import { Bot, Send, Loader, Sparkles } from 'lucide-react';
import { askGemini } from '../services/geminiService';

interface Message {
  sender: 'user' | 'bot';
  text: string;
}

const LnnExplainer: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    { sender: 'bot', text: "Hi there! I'm an AI assistant. Ask me about LNNs, LTC Networks, or any other concept from this project." }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSend = async () => {
    if (input.trim() === '' || isLoading) return;

    const userMessage: Message = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const botResponse = await askGemini(input);
      const botMessage: Message = { sender: 'bot', text: botResponse };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error("Error fetching from Gemini:", error);
      const errorMessage: Message = { sender: 'bot', text: "Sorry, I'm having trouble connecting. Please try again later." };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };
  
  const suggestedQuestions = [
    "What is a Liquid Neural Network?",
    "How are LNNs different from LSTMs?",
    "Explain Ordinary Differential Equations in this context.",
    "Why are LNNs parameter-efficient?"
  ];

  const handleSuggestionClick = (question: string) => {
    setInput(question);
  }

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl shadow-lg flex flex-col h-[40rem]">
      <div className="flex items-center p-4 bg-gray-900/40 border-b border-gray-700">
        <Sparkles className="h-6 w-6 text-cyan-400" />
        <h2 className="text-xl font-bold ml-3 text-white">Concept Explainer AI</h2>
      </div>
      <div className="flex-1 p-4 overflow-y-auto space-y-4">
        {messages.map((msg, index) => (
          <div key={index} className={`flex items-end gap-2 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.sender === 'bot' && <Bot className="h-8 w-8 text-cyan-400 flex-shrink-0" />}
            <div
              className={`max-w-xs md:max-w-sm lg:max-w-md rounded-2xl px-4 py-2 ${
                msg.sender === 'user'
                  ? 'bg-blue-600 text-white rounded-br-none'
                  : 'bg-gray-700 text-gray-200 rounded-bl-none'
              }`}
            >
              <p className="text-sm break-words whitespace-pre-wrap">{msg.text}</p>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex items-end gap-2 justify-start">
            <Bot className="h-8 w-8 text-cyan-400 flex-shrink-0" />
            <div className="bg-gray-700 rounded-2xl rounded-bl-none px-4 py-3">
              <Loader className="h-5 w-5 animate-spin text-gray-300" />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
       <div className="p-4 border-t border-gray-700">
        <div className="flex flex-wrap gap-2 mb-2">
            {suggestedQuestions.map(q => (
                <button 
                    key={q}
                    onClick={() => handleSuggestionClick(q)}
                    className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-xs text-cyan-300 rounded-full transition-colors"
                >
                    {q}
                </button>
            ))}
        </div>
        <div className="flex items-center bg-gray-700 rounded-lg">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question..."
            className="w-full bg-transparent p-3 focus:outline-none text-white"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading}
            className="p-3 text-white disabled:text-gray-500 hover:text-cyan-400 transition-colors"
          >
            <Send className="h-5 w-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default LnnExplainer;
