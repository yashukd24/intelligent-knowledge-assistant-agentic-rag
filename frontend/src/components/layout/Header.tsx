import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { HealthBadge } from '../common/HealthBadge';
import { Badge } from '../common/Badge';
import { Button } from '../common/Button';
import { Bot, Database, Menu, MessageSquarePlus, PlusCircle } from 'lucide-react';

const PAGE_TITLES: Record<string, { title: string; subtitle: string }> = {
  '/': { title: 'Platform Overview', subtitle: 'Intelligent Knowledge Assistant & System Status' },
  '/collections': { title: 'Knowledge Collections', subtitle: 'Manage persistent documents, chunks, and vector index' },
  '/chat': { title: 'AI Assistant & Retrieval', subtitle: 'Interactive conventional RAG and agentic Q&A' },
  '/agent': { title: 'Agent Activity & State Trace', subtitle: 'Iterative multi-step retrieval state machine (Phase 10)' },
  '/evaluation': { title: 'Evaluation & Benchmarking', subtitle: 'Empirical comparison: Conventional vs. Agentic RAG (Phase 20)' },
  '/settings': { title: 'System Configuration', subtitle: 'Environment parameters, models, and safety thresholds' },
};

export interface HeaderProps {
  onOpenMobileMenu?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onOpenMobileMenu }) => {
  const location = useLocation();
  const navigate = useNavigate();

  // Find matching title or default
  const pathKey = Object.keys(PAGE_TITLES).find(
    (key) => key !== '/' && location.pathname.startsWith(key)
  ) || (location.pathname === '/' ? '/' : '/collections');

  const { title, subtitle } = PAGE_TITLES[pathKey] || {
    title: 'Knowledge Assistant',
    subtitle: 'Agentic RAG Platform',
  };

  return (
    <header className="h-16 bg-white border-b border-slate-200/90 px-3 sm:px-6 flex items-center justify-between sticky top-0 z-10 shrink-0">
      {/* Mobile Hamburger Button & Page Title */}
      <div className="flex items-center gap-2 min-w-0 pr-2">
        <button
          type="button"
          onClick={onOpenMobileMenu}
          className="p-1.5 -ml-1 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg md:hidden focus:outline-none focus:ring-2 focus:ring-brand-500 transition-colors shrink-0"
          aria-label="Open navigation menu"
          aria-haspopup="dialog"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex flex-col min-w-0">
          <h1 className="text-sm sm:text-base font-semibold text-slate-900 tracking-tight leading-tight truncate">
            {title}
          </h1>
          <p className="text-xs text-slate-500 font-normal truncate max-w-md hidden sm:block">
            {subtitle}
          </p>
        </div>
      </div>

      {/* System Status Indicators & Header Actions */}
      <div className="flex items-center gap-1.5 sm:gap-2.5 shrink-0">
        {/* Model Badge */}
        <Badge variant="outline" className="hidden lg:flex items-center gap-1.5 py-1 text-slate-600 bg-slate-50/60 font-mono text-[11px]">
          <Bot className="w-3.5 h-3.5 text-brand-600" />
          <span>Qwen 2.5 3B (Local)</span>
        </Badge>

        {/* Vector Store Badge */}
        <Badge variant="outline" className="hidden xl:flex items-center gap-1.5 py-1 text-slate-600 bg-slate-50/60 font-mono text-[11px]">
          <Database className="w-3.5 h-3.5 text-emerald-600" />
          <span>ChromaDB + BGE-Small</span>
        </Badge>

        {/* Live Backend Connection Badge */}
        <HealthBadge />

        <div className="h-5 w-[1px] bg-slate-200 mx-0.5 sm:mx-1 hidden sm:block" />

        {/* Dynamic Contextual Action */}
        {location.pathname === '/chat' ? (
          <Button
            size="sm"
            variant="outline"
            onClick={() => window.location.reload()}
            className="text-xs h-8 px-2.5 sm:px-3 shrink-0"
          >
            <MessageSquarePlus className="w-3.5 h-3.5 sm:mr-1" />
            <span className="hidden sm:inline">New Session</span>
          </Button>
        ) : (
          <Button
            size="sm"
            variant="primary"
            onClick={() => navigate('/collections')}
            className="text-xs h-8 px-2.5 sm:px-3 shrink-0"
          >
            <PlusCircle className="w-3.5 h-3.5 sm:mr-1" />
            <span className="hidden sm:inline">Add Knowledge</span>
          </Button>
        )}
      </div>
    </header>
  );
};
