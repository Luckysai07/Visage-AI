import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Search, Sparkles } from 'lucide-react';
import AnalyzeView from './components/AnalyzeView';
import SearchView from './components/SearchView';
import './App.css';

const tabs = [
  { id: 'analyze', label: 'Face Analysis', icon: Brain, desc: 'Upload & analyze any face' },
  { id: 'search', label: 'Face Search', icon: Search, desc: 'Find similar faces by attributes' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('analyze');

  return (
    <div className="app-root">
      {/* Top Navigation */}
      <header className="topnav glass-panel">
        <div className="brand">
          <div className="brand-icon">
            <Sparkles size={16} />
          </div>
          <span className="brand-name">Visage<span className="brand-ai"> AI</span></span>
        </div>

        <nav className="tab-row">
          {tabs.map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                className={`tab-btn ${active ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {active && (
                  <motion.span
                    layoutId="tab-pill"
                    className="tab-pill"
                    transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                  />
                )}
                <Icon size={16} style={{ position: 'relative' }} />
                <span style={{ position: 'relative' }}>{tab.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="status-chip">
          <span className="dot pulse" />
          Live
        </div>
      </header>

      {/* Page Content */}
      <main className="page-main">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -16 }}
            transition={{ duration: 0.25 }}
            className="page-view"
          >
            {activeTab === 'analyze' ? <AnalyzeView /> : <SearchView />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
