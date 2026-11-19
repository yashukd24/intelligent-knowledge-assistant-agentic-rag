import React, { useState, useEffect } from 'react';
import {
  X,
  Activity,
  Layers,
  Database,
  ShieldCheck,
  Code2,
  Copy,
  Check,
  Clock,
  Cpu,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react';
import type { RAGAnswerResponse } from '../../types/api';
import { ExecutionTimeline } from './ExecutionTimeline';
import { RetrievalInspector } from './RetrievalInspector';
import { CitationValidationCard } from './CitationValidationCard';

interface TraceDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  responseMeta: RAGAnswerResponse | null;
  onOpenSourceInDrawer?: (sourceId: string) => void;
}

type TabKey = 'timeline' | 'retrieval' | 'citations' | 'raw';

export const TraceDrawer: React.FC<TraceDrawerProps> = ({
  isOpen,
  onClose,
  responseMeta,
  onOpenSourceInDrawer,
}) => {
  const [activeTab, setActiveTab] = useState<TabKey>('timeline');
  const [copiedRaw, setCopiedRaw] = useState(false);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !responseMeta) return null;

  const trace = responseMeta.execution_trace;
  const stages = trace?.stages || [];
  const completedCount = stages.filter((s) => s.status === 'completed').length;
  const failedCount = stages.filter((s) => s.status === 'failed').length;
  const skippedCount = stages.filter((s) => s.status === 'skipped').length;

  const handleCopyRaw = () => {
    navigator.clipboard.writeText(JSON.stringify(responseMeta, null, 2));
    setCopiedRaw(true);
    setTimeout(() => setCopiedRaw(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-slate-900/40 backdrop-blur-xs transition-opacity"
        onClick={onClose}
        aria-hidden="true"
      />

      <div className="fixed inset-y-0 right-0 max-w-full flex pl-4 sm:pl-10">
        <div className="w-screen max-w-2xl bg-white shadow-2xl flex flex-col border-l border-slate-200">
          {/* Header */}
          <div className="p-4 sm:p-5 border-b border-slate-200/90 bg-slate-50/70 flex items-center justify-between shrink-0">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="p-2 rounded-xl bg-purple-100 text-purple-700 shrink-0">
                <Activity className="w-5 h-5" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <h3 className="text-sm sm:text-base font-bold text-slate-900 truncate">
                    Agent Activity & Reasoning Visualizer
                  </h3>
                  <span
                    className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border ${
                      failedCount > 0
                        ? 'bg-rose-50 text-rose-700 border-rose-200'
                        : responseMeta.status === 'insufficient_evidence'
                        ? 'bg-amber-50 text-amber-700 border-amber-200'
                        : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                    }`}
                  >
                    {failedCount > 0 ? 'Execution Failed' : responseMeta.status}
                  </span>
                </div>
                <div className="text-[11px] text-slate-500 font-mono mt-0.5 truncate">
                  Trace ID: {trace?.trace_id || 'trace-session'}
                </div>
              </div>
            </div>

            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-200/60 transition-colors cursor-pointer shrink-0"
              aria-label="Close trace visualizer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Compact Execution Summary Bar */}
          <div className="px-4 sm:px-5 py-3 bg-white border-b border-slate-100 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs shrink-0">
            <div>
              <span className="text-slate-500 block text-[10px] uppercase font-bold tracking-wider">
                Total Latency
              </span>
              <span className="font-mono font-semibold text-slate-900 flex items-center gap-1 mt-0.5">
                <Clock className="w-3.5 h-3.5 text-slate-400" />
                {responseMeta.total_latency_ms?.toFixed(1) || '0.0'} ms
              </span>
            </div>

            <div>
              <span className="text-slate-500 block text-[10px] uppercase font-bold tracking-wider">
                Stages Summary
              </span>
              <span className="font-medium text-slate-900 flex items-center gap-1 mt-0.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                {completedCount}/{stages.length || 9} Done
                {skippedCount > 0 && (
                  <span className="text-amber-600 text-[11px]">({skippedCount} skip)</span>
                )}
              </span>
            </div>

            <div>
              <span className="text-slate-500 block text-[10px] uppercase font-bold tracking-wider">
                Provider / Model
              </span>
              <span className="font-medium text-slate-900 truncate block mt-0.5" title={responseMeta.model}>
                {responseMeta.model || 'rule_based'}
              </span>
            </div>

            <div>
              <span className="text-slate-500 block text-[10px] uppercase font-bold tracking-wider">
                Token Usage
              </span>
              <span className="font-mono font-semibold text-slate-900 mt-0.5 block">
                {responseMeta.usage?.total_tokens ? `${responseMeta.usage.total_tokens} tok` : '0 tok'}
              </span>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex border-b border-slate-200 px-4 sm:px-5 bg-slate-50/50 shrink-0 gap-1 overflow-x-auto">
            <button
              onClick={() => setActiveTab('timeline')}
              className={`py-2.5 px-3 text-xs font-semibold border-b-2 flex items-center gap-1.5 transition-colors cursor-pointer shrink-0 ${
                activeTab === 'timeline'
                  ? 'border-brand-600 text-brand-600'
                  : 'border-transparent text-slate-600 hover:text-slate-900'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              9-Stage Timeline
            </button>

            <button
              onClick={() => setActiveTab('retrieval')}
              className={`py-2.5 px-3 text-xs font-semibold border-b-2 flex items-center gap-1.5 transition-colors cursor-pointer shrink-0 ${
                activeTab === 'retrieval'
                  ? 'border-brand-600 text-brand-600'
                  : 'border-transparent text-slate-600 hover:text-slate-900'
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              Retrieval ({responseMeta.sources.length})
            </button>

            <button
              onClick={() => setActiveTab('citations')}
              className={`py-2.5 px-3 text-xs font-semibold border-b-2 flex items-center gap-1.5 transition-colors cursor-pointer shrink-0 ${
                activeTab === 'citations'
                  ? 'border-brand-600 text-brand-600'
                  : 'border-transparent text-slate-600 hover:text-slate-900'
              }`}
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              Citations ({responseMeta.valid_citations.length} valid)
            </button>

            <button
              onClick={() => setActiveTab('raw')}
              className={`py-2.5 px-3 text-xs font-semibold border-b-2 flex items-center gap-1.5 transition-colors cursor-pointer shrink-0 ${
                activeTab === 'raw'
                  ? 'border-brand-600 text-brand-600'
                  : 'border-transparent text-slate-600 hover:text-slate-900'
              }`}
            >
              <Code2 className="w-3.5 h-3.5" />
              Raw Trace
            </button>
          </div>

          {/* Tab Content Body */}
          <div className="flex-1 overflow-y-auto p-4 sm:p-5">
            {activeTab === 'timeline' && (
              <ExecutionTimeline
                stages={stages}
                onInspectSource={(sourceId) => {
                  if (onOpenSourceInDrawer) onOpenSourceInDrawer(sourceId);
                }}
              />
            )}

            {activeTab === 'retrieval' && (
              <RetrievalInspector
                sources={responseMeta.sources}
                validCitations={responseMeta.valid_citations}
                onOpenSourceInDrawer={(sourceId) => {
                  if (onOpenSourceInDrawer) onOpenSourceInDrawer(sourceId);
                }}
              />
            )}

            {activeTab === 'citations' && (
              <CitationValidationCard
                validCitations={responseMeta.valid_citations}
                invalidCitations={responseMeta.invalid_citations}
                uncitedSources={responseMeta.uncited_sources}
                limitationsNote={responseMeta.limitations_note}
                onInspectCitation={(citationId) => {
                  if (onOpenSourceInDrawer) onOpenSourceInDrawer(citationId);
                }}
              />
            )}

            {activeTab === 'raw' && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-700">
                    Complete Diagnostic Response JSON
                  </span>
                  <button
                    onClick={handleCopyRaw}
                    className="text-xs font-medium text-brand-600 hover:text-brand-700 inline-flex items-center gap-1 cursor-pointer"
                  >
                    {copiedRaw ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-emerald-600" /> Copied JSON
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" /> Copy JSON
                      </>
                    )}
                  </button>
                </div>
                <pre className="p-3 bg-slate-900 text-slate-100 rounded-xl text-[11px] font-mono overflow-x-auto leading-relaxed max-h-[60vh]">
                  {JSON.stringify(responseMeta, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
