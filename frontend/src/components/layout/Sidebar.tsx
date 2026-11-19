import React, { useEffect, useRef } from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  FolderArchive,
  MessageSquare,
  Activity,
  BarChart3,
  Settings,
  ShieldCheck,
  Cpu,
  Layers,
  X,
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { Badge } from '../common/Badge';

interface NavItem {
  label: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
}

const NAV_ITEMS: NavItem[] = [
  { label: 'Dashboard', path: '/', icon: LayoutDashboard },
  { label: 'College Knowledge Base', path: '/collections', icon: FolderArchive },
  { label: 'AI Assistant', path: '/chat', icon: MessageSquare },
  { label: 'Agent Activity', path: '/agent', icon: Activity, badge: 'Agentic' },
  { label: 'Evaluation', path: '/evaluation', icon: BarChart3, badge: 'Benchmark' },
  { label: 'System Settings', path: '/settings', icon: Settings },
];

interface SidebarContentProps {
  onItemClick?: () => void;
  onClose?: () => void;
  isMobile?: boolean;
}

const SidebarContent: React.FC<SidebarContentProps> = ({
  onItemClick,
  onClose,
  isMobile = false,
}) => {
  return (
    <div className="flex flex-col h-full bg-white select-none">
      {/* Brand & Workspace Header */}
      <div className="p-5 border-b border-slate-100 flex items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-brand-700 to-brand-500 flex items-center justify-center text-white shadow-sm ring-1 ring-brand-600/20 shrink-0">
            <Layers className="w-5 h-5" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5">
              <span className="font-semibold text-slate-900 text-sm tracking-tight truncate">
                College Intelligence
              </span>
            </div>
            <div className="text-[11px] text-slate-500 font-medium truncate flex items-center gap-1">
              <span>Agentic RAG</span>
              <span className="text-slate-300">•</span>
              <span className="text-brand-600 font-semibold">Institutional AI</span>
            </div>
          </div>
        </div>

        {/* Mobile Close Button */}
        {isMobile && onClose && (
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors focus:outline-none focus:ring-2 focus:ring-brand-500"
            aria-label="Close navigation menu"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <div className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
          Core Platform
        </div>
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            onClick={onItemClick}
            className={({ isActive }) =>
              cn(
                'flex items-center justify-between px-3 py-2 rounded-lg text-sm font-medium transition-all group',
                isActive
                  ? 'bg-brand-50 text-brand-700 font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70'
              )
            }
          >
            {({ isActive }) => (
              <>
                <div className="flex items-center gap-3">
                  <item.icon
                    className={cn(
                      'w-4 h-4 transition-colors',
                      isActive ? 'text-brand-600' : 'text-slate-400 group-hover:text-slate-600'
                    )}
                  />
                  <span>{item.label}</span>
                </div>
                {item.badge && (
                  <span
                    className={cn(
                      'text-[10px] px-1.5 py-0.5 rounded font-mono font-medium',
                      isActive
                        ? 'bg-brand-200/60 text-brand-800'
                        : 'bg-slate-100 text-slate-500 group-hover:bg-slate-200'
                    )}
                  >
                    {item.badge}
                  </span>
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* System Engine & Safety Status Footer */}
      <div className="p-3.5 m-3 rounded-xl bg-slate-50/80 border border-slate-200/70 space-y-2 text-xs shrink-0">
        <div className="flex items-center justify-between text-slate-700">
          <span className="flex items-center gap-1.5 font-medium text-[11px]">
            <Cpu className="w-3.5 h-3.5 text-brand-600" />
            <span>Plain Python Engine</span>
          </span>
          <Badge variant="primary" size="sm" className="text-[10px] py-0 px-1.5">
            Active
          </Badge>
        </div>

        <div className="flex items-center justify-between text-slate-600 text-[11px] pt-1 border-t border-slate-200/50">
          <span className="flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
            <span>RAM Safe Gate</span>
          </span>
          <span className="font-mono text-[10px] text-slate-500 font-semibold">
            &ge; 5.0 GB
          </span>
        </div>
      </div>
    </div>
  );
};

export interface SidebarProps {
  isOpen?: boolean;
  onClose?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ isOpen = false, onClose }) => {
  const drawerRef = useRef<HTMLDivElement>(null);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && onClose) {
        onClose();
      }
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden'; // Prevent background scroll
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [isOpen, onClose]);

  return (
    <>
      {/* Desktop Persistent Sidebar (md and above: 768px+) */}
      <aside className="hidden md:flex flex-col w-64 bg-white border-r border-slate-200/90 shrink-0 select-none h-screen sticky top-0">
        <SidebarContent />
      </aside>

      {/* Mobile Slide-Over Drawer (< 768px) */}
      {isOpen && (
        <div
          className="fixed inset-0 z-50 md:hidden flex"
          role="dialog"
          aria-modal="true"
          aria-label="Navigation drawer"
        >
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-slate-900/50 backdrop-blur-xs transition-opacity animate-in fade-in duration-200"
            onClick={onClose}
            aria-hidden="true"
          />

          {/* Drawer Container */}
          <aside
            ref={drawerRef}
            className="relative w-72 max-w-[85vw] bg-white h-full shadow-2xl flex flex-col z-10 animate-in slide-in-from-left duration-200"
          >
            <SidebarContent onItemClick={onClose} onClose={onClose} isMobile />
          </aside>
        </div>
      )}
    </>
  );
};
