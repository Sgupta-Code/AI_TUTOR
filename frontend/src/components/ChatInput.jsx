/**
 * ChatInput Component
 * Input field for sending messages with enhanced UX
 */

import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Sparkles } from 'lucide-react';

const ChatInput = ({ onSend, isLoading, placeholder }) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  // Handle send message
  const handleSend = () => {
    if (input.trim() && !isLoading) {
      onSend(input.trim());
      setInput('');
    }
  };

  // Handle key press (Enter to send, Shift+Enter for new line)
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Sample questions for quick access
  const sampleQuestions = [
    'What is machine learning?',
    'How to implement a neural network?',
    'Show me Python examples',
    'Explain deep learning',
  ];

  const [showSamples, setShowSamples] = useState(false);

  return (
    <div className="bg-white border-t border-gray-200 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Sample Questions */}
        {showSamples && (
          <div className="mb-3 flex flex-wrap gap-2 animate-slide-down">
            {sampleQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => {
                  setInput(question);
                  setShowSamples(false);
                }}
                className="text-sm px-3 py-1.5 bg-primary-50 hover:bg-primary-100 text-primary-700 rounded-full transition-colors"
              >
                {question}
              </button>
            ))}
          </div>
        )}

        {/* Input Container */}
        <div className="relative flex items-end gap-2">
          {/* Sample Questions Toggle */}
          <button
            onClick={() => setShowSamples(!showSamples)}
            className="shrink-0 p-2 text-gray-400 hover:text-primary-600 transition-colors rounded-lg hover:bg-gray-100"
            title="Sample questions"
          >
            <Sparkles size={20} />
          </button>

          {/* Textarea */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={placeholder || 'Ask me anything about AI, ML, programming...'}
              disabled={isLoading}
              rows={1}
              className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500 transition-all max-h-32 overflow-y-auto"
            />

            {/* Character Count */}
            {input.length > 0 && (
              <div className="absolute bottom-1 right-12 text-xs text-gray-400">
                {input.length}
              </div>
            )}
          </div>

          {/* Send Button */}
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className={`shrink-0 p-3 rounded-xl transition-all ${
              input.trim() && !isLoading
                ? 'bg-primary-600 hover:bg-primary-700 text-white shadow-lg hover:shadow-xl'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'
            }`}
            title="Send message"
          >
            {isLoading ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Send size={20} />
            )}
          </button>
        </div>

        {/* Helper Text */}
        <div className="mt-2 text-xs text-gray-500 text-center">
          Press <kbd className="px-2 py-0.5 bg-gray-100 rounded">Enter</kbd> to
          send, <kbd className="px-2 py-0.5 bg-gray-100 rounded">Shift + Enter</kbd> for new line
        </div>
      </div>
    </div>
  );
};

export default ChatInput;