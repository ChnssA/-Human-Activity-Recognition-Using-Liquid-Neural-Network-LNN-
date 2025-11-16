
import { GoogleGenAI } from "@google/genai";

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  // In a real app, you might want to handle this more gracefully.
  // For this context, we will proceed, and the user will see a console error
  // if the key is not set in their environment.
  console.warn("Gemini API key not found in environment variables.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

export const askGemini = async (userPrompt: string): Promise<string> => {
  if (!API_KEY) {
    return "The API key for Gemini is not configured. Please set the API_KEY environment variable.";
  }
  
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: [{
        parts: [{ text: userPrompt }]
      }],
      config: {
        systemInstruction: `You are an expert in Machine Learning and Neural Networks, specializing in Recurrent Neural Networks (RNNs), LSTMs, and Liquid Neural Networks (LNNs). Your role is to explain complex concepts from the 'Human Activity Recognition Using Liquid Neural Network' project in a clear, concise, and easy-to-understand manner for a university student. Break down jargon and use analogies where helpful. Keep responses focused and relatively short.`,
      }
    });

    return response.text;
  } catch (error) {
    console.error("Gemini API call failed:", error);
    throw new Error("Failed to get a response from the AI model.");
  }
};
