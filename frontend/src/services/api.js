/**
 * API Service for AI Tutor
 * Handles communication with the Flask backend
 */

import axios from "axios";

// Base URL (Vite uses import.meta.env instead of process.env)
const API_BASE_URL = import.meta.env.VITE_API_URL || "https://celinda-undeft-badgeringly.ngrok-free.dev";

// Axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30000,
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Future: attach token here
    // const token = localStorage.getItem("token");
    // if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      console.error("API Error:", error.response.data);
    } else if (error.request) {
      console.error("Network Error:", error.request);
    } else {
      console.error("Error:", error.message);
    }
    return Promise.reject(error);
  }
);

const apiService = {
  // Health check
  healthCheck: async () => {
    const response = await api.get("/");
    return response.data;
  },

  // Send chat message
  sendMessage: async (query, sessionId) => {
    const response = await api.post("/api/chat", {
      query,
      session_id: sessionId,
    });
    return response.data;
  },

  // Get history
  getHistory: async (sessionId, limit = 50) => {
    const response = await api.get(`/api/history/${sessionId}`, {
      params: { limit },
    });
    return response.data;
  },

  // Submit feedback
  submitFeedback: async (conversationId, rating, feedbackText = "") => {
    const response = await api.post("/api/feedback", {
      conversation_id: conversationId,
      rating,
      feedback_text: feedbackText,
    });
    return response.data;
  },

  // Get progress
  getProgress: async (sessionId) => {
    const response = await api.get(`/api/progress/${sessionId}`);
    return response.data;
  },

  // Update progress
  updateProgress: async (sessionId, topic, completed = false, score = null) => {
    const response = await api.post("/api/progress", {
      session_id: sessionId,
      topic,
      completed,
      score,
    });
    return response.data;
  },

  // Get topics
  getTopics: async () => {
    const response = await api.get("/api/topics");
    return response.data;
  },

  // Get practice problems
  getPracticeProblems: async (topic) => {
    const response = await api.post("/api/practice", { topic });
    return response.data;
  },

  // Get analytics
  getAnalytics: async () => {
    const response = await api.get("/api/analytics");
    return response.data;
  },

  // Get statistics
  getStats: async () => {
    const response = await api.get("/api/stats");
    return response.data;
  },
};

export default apiService;

