/**
 * ChatMessage Component
 * Displays individual chat messages with proper formatting
 */

import React from 'react';
import { User, Bot, ThumbsUp, ThumbsDown, Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const ChatMessage = ({ message, onFeedback }) => {
  const [copied, setCopied] = React.useState(false);
  const [feedbackGiven, setFeedbackGiven] = React.useState(false);

  const isUser = message.type === 'user';

  // Copy response to clipboard
  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Handle feedback
  const handleFeedback = (isPositive) => {
    if (onFeedback && !feedbackGiven) {
      onFeedback(message.id, isPositive ? 5 : 2);
      setFeedbackGiven(true);
    }
  };

  return (
    <div
      className={`flex items-start gap-3 p-4 rounded-lg animate-slide-up ${
        isUser
          ? 'bg-primary-50 ml-8'
          : 'bg-white shadow-sm mr-8'
      }`}
    >
      {/* Avatar */}
      <div
        className={`shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
          isUser
            ? 'bg-primary-600 text-white'
            : 'bg-linear-to-br from-purple-500 to-pink-500 text-white'
        }`}
      >
        {isUser ? <User size={20} /> : <Bot size={20} />}
      </div>

      {/* Message Content */}
      <div className="flex-1 min-w-0">
        {/* Header */}
        <div className="flex items-center justify-between mb-2">
          <span className="font-semibold text-gray-900">
            {isUser ? 'You' : 'AI Tutor'}
          </span>
          {message.timestamp && (
            <span className="text-xs text-gray-500">
              {new Date(message.timestamp).toLocaleTimeString()}
            </span>
          )}
        </div>

        {/* Message Text */}
        <div className="prose prose-sm max-w-none">
          {isUser ? (
            <p className="text-gray-800">{message.content}</p>
          ) : (
            <div className="markdown-content">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Intent Badge (for AI responses) */}
        {!isUser && message.intent && (
          <div className="mt-3 flex items-center gap-2">
            <span className="badge badge-primary text-xs">
              {message.intent}
            </span>
            {message.confidence && (
              <span className="text-xs text-gray-500">
                Confidence: {(message.confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
        )}

        {/* Suggestions */}
        {!isUser && message.suggestions && message.suggestions.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {message.suggestions.map((suggestion, index) => (
              <button
                key={index}
                className="text-xs px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full transition-colors"
                onClick={() => {
                  // Trigger suggestion click (handled by parent)
                  const event = new CustomEvent('suggestionClick', {
                    detail: { suggestion },
                  });
                  window.dispatchEvent(event);
                }}
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}

        {/* Actions (for AI responses) */}
        {!isUser && (
          <div className="mt-4 flex items-center gap-3 text-gray-500">
            {/* Copy Button */}
            <button
              onClick={handleCopy}
              className="flex items-center gap-1 text-xs hover:text-primary-600 transition-colors"
              title="Copy response"
            >
              {copied ? (
                <>
                  <Check size={14} />
                  <span>Copied!</span>
                </>
              ) : (
                <>
                  <Copy size={14} />
                  <span>Copy</span>
                </>
              )}
            </button>

            {/* Feedback Buttons */}
            {!feedbackGiven ? (
              <>
                <button
                  onClick={() => handleFeedback(true)}
                  className="flex items-center gap-1 text-xs hover:text-green-600 transition-colors"
                  title="Helpful"
                >
                  <ThumbsUp size={14} />
                  <span>Helpful</span>
                </button>
                <button
                  onClick={() => handleFeedback(false)}
                  className="flex items-center gap-1 text-xs hover:text-red-600 transition-colors"
                  title="Not helpful"
                >
                  <ThumbsDown size={14} />
                  <span>Not helpful</span>
                </button>
              </>
            ) : (
              <span className="text-xs text-green-600">
                ✓ Feedback submitted
              </span>
            )}
          </div>
        )}

        {/* Response Time (for AI responses) */}
        {!isUser && message.response_time && (
          <div className="mt-2 text-xs text-gray-400">
            Response time: {message.response_time.toFixed(2)}s
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;