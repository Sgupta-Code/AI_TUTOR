/**
 * Chat Page Component
 * Main chat interface for the AI Tutor
 */

import React, { useState, useEffect, useRef } from 'react';
import { Bot, AlertCircle, RefreshCw, BookOpen, TrendingUp } from 'lucide-react';
import ChatMessage from '../components/ChatMessage';
import ChatInput from '../components/ChatInput';
import apiService from '../services/api.js';

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sessionId] = useState(`session_${Date.now()}`);
  const [apiStatus, setApiStatus] = useState('checking');
  const messagesEndRef = useRef(null);

  // Scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check API health on mount
  useEffect(() => {
    checkApiHealth();
    loadHistory();
  }, []);

  // Listen for suggestion clicks
  useEffect(() => {
    const handleSuggestionClick = (event) => {
      const { suggestion } = event.detail;
      handleSendMessage(suggestion);
    };

    window.addEventListener('suggestionClick', handleSuggestionClick);
    return () => window.removeEventListener('suggestionClick', handleSuggestionClick);
  }, []);

  // Check if backend API is running
  const checkApiHealth = async () => {
    try {
      await apiService.healthCheck();
      setApiStatus('online');
    } catch (e) {
      setApiStatus('offline');
      setError('Unable to connect to AI Tutor backend. Please make sure the server is running.');
    }
  };

  // Load conversation history
  const loadHistory = async () => {
    try {
      const response = await apiService.getHistory(sessionId);
      if (response.success && response.history.length > 0) {
        const formattedHistory = response.history.reverse().map((item) => [
          {
            id: `user_${item.id}`,
            type: 'user',
            content: item.query,
            timestamp: item.timestamp,
          },
          {
            id: `ai_${item.id}`,
            type: 'ai',
            content: item.response,
            intent: item.intent,
            confidence: item.confidence,
            timestamp: item.timestamp,
            response_time: item.response_time,
          },
        ]).flat();
        
        setMessages(formattedHistory);
      }
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  // Handle sending a message
  const handleSendMessage = async (content) => {
    if (!content.trim() || isLoading) return;

    // Add user message
    const userMessage = {
      id: `user_${Date.now()}`,
      type: 'user',
      content,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      // Send to backend
      const response = await apiService.sendMessage(content, sessionId);

      if (response.success) {
        // Add AI response
        const aiMessage = {
          id: `ai_${response.conversation_id}`,
          type: 'ai',
          content: response.response,
          intent: response.intent,
          confidence: response.confidence,
          suggestions: response.suggestions || [],
          timestamp: response.timestamp,
          response_time: response.response_time,
        };

        setMessages((prev) => [...prev, aiMessage]);
      } else {
        throw new Error(response.error || 'Failed to get response');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setError('Failed to send message. Please try again.');
      
      // Add error message
      const errorMessage = {
        id: `error_${Date.now()}`,
        type: 'ai',
        content: '❌ Sorry, I encountered an error processing your request. Please try again or rephrase your question.',
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle feedback
  const handleFeedback = async (messageId, rating) => {
    try {
      // Extract conversation ID from message ID
      const conversationId = parseInt(messageId.split('_')[1]);
      await apiService.submitFeedback(conversationId, rating);
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  // Clear conversation
  const handleClearChat = () => {
    if (window.confirm('Are you sure you want to clear the chat?')) {
      setMessages([]);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-linear-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            {/* Logo and Title */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-linear-to-br from-primary-500 to-purple-600 rounded-xl flex items-center justify-center">
                <Bot className="text-white" size={24} />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">AI Tutor</h1>
                <p className="text-sm text-gray-500">Your Personal Learning Assistant</p>
              </div>
            </div>

            {/* Status and Actions */}
            <div className="flex items-center gap-4">
              {/* API Status */}
              <div className="flex items-center gap-2">
                <div
                  className={`w-2 h-2 rounded-full ${
                    apiStatus === 'online'
                      ? 'bg-green-500'
                      : apiStatus === 'offline'
                      ? 'bg-red-500'
                      : 'bg-yellow-500 animate-pulse'
                  }`}
                />
                <span className="text-sm text-gray-600">
                  {apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Checking...'}
                </span>
              </div>

              {/* Clear Chat Button */}
              {messages.length > 0 && (
                <button
                  onClick={handleClearChat}
                  className="btn-secondary text-sm"
                  title="Clear conversation"
                >
                  <RefreshCw size={16} className="inline mr-1" />
                  Clear
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-500 p-4">
          <div className="flex items-center max-w-7xl mx-auto">
            <AlertCircle className="text-red-500 mr-3" size={20} />
            <div>
              <p className="text-red-800 font-medium">Error</p>
              <p className="text-red-600 text-sm">{error}</p>
            </div>
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-500 hover:text-red-700"
            >
              ×
            </button>
          </div>
        </div>
      )}

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.length === 0 ? (
            /* Welcome Screen */
            <div className="text-center py-12 animate-fade-in">
              <div className="w-20 h-20 bg-linear-to-br from-primary-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <Bot className="text-white" size={40} />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 mb-3">
                Welcome to AI Tutor!
              </h2>
              <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
                I'm here to help you learn about AI, Machine Learning, Programming, and more.
                Ask me anything!
              </p>

              {/* Quick Start Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto">
                <div className="card text-left">
                  <BookOpen className="text-primary-600 mb-3" size={24} />
                  <h3 className="font-semibold text-gray-900 mb-2">Learn Concepts</h3>
                  <p className="text-sm text-gray-600">
                    Ask for explanations, definitions, and detailed tutorials
                  </p>
                </div>
                <div className="card text-left">
                  <TrendingUp className="text-purple-600 mb-3" size={24} />
                  <h3 className="font-semibold text-gray-900 mb-2">Practice Skills</h3>
                  <p className="text-sm text-gray-600">
                    Get practice problems and hands-on coding examples
                  </p>
                </div>
                <div className="card text-left">
                  <Bot className="text-pink-600 mb-3" size={24} />
                  <h3 className="font-semibold text-gray-900 mb-2">Get Instant Help</h3>
                  <p className="text-sm text-gray-600">
                    Real-time responses powered by advanced AI models
                  </p>
                </div>
              </div>
            </div>
          ) : (
            /* Messages */
            messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                onFeedback={handleFeedback}
              />
            ))
          )}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex items-start gap-3 p-4 bg-white rounded-lg shadow-sm mr-8 animate-slide-up">
              <div className="shrink-0 w-10 h-10 rounded-full bg-linear-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white">
                <Bot size={20} />
              </div>
              <div className="flex-1">
                <span className="font-semibold text-gray-900">AI Tutor</span>
                <div className="typing-indicator mt-2">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <ChatInput
        onSend={handleSendMessage}
        isLoading={isLoading}
        placeholder="Ask me anything about AI, ML, programming..."
      />
    </div>
  );
};

export default Chat;