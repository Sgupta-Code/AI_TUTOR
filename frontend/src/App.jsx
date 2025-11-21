import React from 'react'
import './App.css'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Chat from './pages/Chat.jsx';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path = "/" element = {<Chat/>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App
