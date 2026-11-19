import React, { useState } from 'react';
import {
  CheckCircle2,
  XCircle,
  Clock,
  MinusCircle,
  Loader2,
  ChevronDown,
  ChevronRight,
  Database,
  Search,
  Filter,
  Layers,
  Cpu,
  ShieldCheck,
  Check,
  AlertTriangle,
} from 'lucide-react';
import type { TraceStage, TraceStageStatus } from '../../types/api';
import { Badge } from '../common/Badge';

interface ExecutionTimelineProps {
  stages: TraceStage[];
  onInspectSource?: (sourceId: string) => void;
}

const STAGE_ICONS: Record<string, React.ReactNode> = {
  'stage-1-query': <Search className="w-4 h-4 text-brand-600" />,
  'stage-2-validation': <Check className="w-4 h-4 text-emerald-600" />,
  'stage-3-collection': <Database className="w-4 h-4 text-indigo-600" />,
  'stage-4-retrieval': <Database className="w-4 h-4 text-blue-600" />,
  'stage-5-filtering': <Filter className="w-4 h-4 text-purple-600" />,
  'stage-6-context': <Layers className="w-4 h-4 text-amber-600" />,
  'stage-7-generation': <Cpu className="w-4 h-4 text-teal-600" />,
  'stage-8-citation': <ShieldCheck className="w-4 h-4 text-emerald-600" />,
  'stage-9-final': <CheckCircle2 className="w-4 h-4 text-brand-600" />,
};

export const ExecutionTimeline: React.FC<ExecutionTimelineProps> = ({
  stages,
}) => {
  const [expandedStageIds, setExpandedStageIds] = useState<Record<string, boolean>>({
    'stage-4-retrieval': true,
    'stage-7-generation': true,
  });

  const toggleStage = (id: string) => {
    setExpandedStageIds((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const expandAll = () => {
    const all: Record<string, boolean> = {};
    stages.forEach((s) => (all[s.id] = true));
    setExpandedStageIds(all);
  };

  const collapseAll = () => {
    setExpandedStageIds({});
  };

  const formatTimestamp = (iso?: string | null) => {
    if (!iso) return null;
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) +
        '.' +
        d.getMilliseconds().toString().padStart(3, '0');
    } catch {
      return iso;
    }
  };

  const renderStatusBadge = (status: TraceStageStatus) => {
    switch (status) {
      case 'completed':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
            <CheckCircle2 className="w-3 h-3 text-emerald-600" />
            Completed
          </span>
        );
      case 'failed':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-rose-50 text-rose-700 border border-rose-200">
            <XCircle className="w-3 h-3 text-rose-600" />
            Failed
          </span>
        );
      case 'skipped':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-amber-50 text-amber-700 border border-amber-200">
            <MinusCircle className="w-3 h-3 text-amber-600" />
            Skipped
          </span>
        );
      case 'running':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-blue-50 text-blue-700 border border-blue-200">
            <Loader2 className="w-3 h-3 text-blue-600 animate-spin" />
            Running
          </span>
        );
      case 'pending':
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-slate-100 text-slate-600 border border-slate-200">
            <Clock className="w-3 h-3 text-slate-400" />
            Pending
          </span>
        );
    }
  };

  return (
    <div className="space-y-3">
      {/* Timeline Controls */}
      <div className="flex items-center justify-between text-xs text-slate-600 px-1">
        <span className="font-semibold text-slate-700">
          9-Stage Pipeline Lifecycle ({stages.length} Stages Recorded)
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={expandAll}
            className="text-brand-600 hover:text-brand-700 hover:underline font-medium cursor-pointer"
          >
            Expand All
          </button>
          <span>•</span>
          <button
            onClick={collapseAll}
            className="text-slate-500 hover:text-slate-700 hover:underline font-medium cursor-pointer"
          >
            Collapse All
          </button>
        </div>
      </div>

      {/* Timeline Stage List */}
      <div className="relative pl-6 space-y-4 before:absolute before:left-2.5 before:top-3 before:bottom-3 before:w-0.5 before:bg-slate-200">
        {stages.map((stage, idx) => {
          const isExpanded = !!expandedStageIds[stage.id];
          const hasDetails = stage.details && Object.keys(stage.details).length > 0;
          const isFailed = stage.status === 'failed';
          const isSkipped = stage.status === 'skipped';

          return (
            <div key={stage.id} className="relative group">
              {/* Timeline Node Dot */}
              <div
                className={`absolute -left-6 top-2 w-5 h-5 rounded-full border-2 flex items-center justify-center bg-white z-10 transition-colors ${
                  isFailed
                    ? 'border-rose-500 text-rose-500 bg-rose-50'
                    : isSkipped
                    ? 'border-amber-400 text-amber-500 bg-amber-50'
                    : stage.status === 'completed'
                    ? 'border-emerald-500 text-emerald-500 bg-emerald-50'
                    : 'border-slate-300 text-slate-400'
                }`}
              >
                <span className="text-[10px] font-bold">{idx + 1}</span>
              </div>

              {/* Stage Card */}
              <div
                className={`rounded-xl border transition-all ${
                  isFailed
                    ? 'bg-rose-50/40 border-rose-200/90'
                    : isSkipped
                    ? 'bg-slate-50/80 border-slate-200/80 text-slate-500'
                    : 'bg-white border-slate-200/90 shadow-2xs hover:border-slate-300'
                }`}
              >
                {/* Header (Clickable Accordion) */}
                <button
                  type="button"
                  onClick={() => toggleStage(stage.id)}
                  className="w-full text-left p-3 sm:p-3.5 flex items-center justify-between gap-3 focus:outline-none cursor-pointer"
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <div className="p-1.5 rounded-lg bg-slate-100 text-slate-700 shrink-0">
                      {STAGE_ICONS[stage.id] || <Layers className="w-4 h-4 text-slate-600" />}
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-xs sm:text-sm font-semibold text-slate-900 truncate">
                          {stage.name}
                        </span>
                        {renderStatusBadge(stage.status)}
                      </div>
                      <div className="text-[11px] text-slate-500 flex items-center gap-2 mt-0.5">
                        {stage.duration_ms !== undefined && stage.duration_ms !== null ? (
                          <span className="font-mono font-medium text-slate-600">
                            {stage.duration_ms.toFixed(1)} ms
                          </span>
                        ) : (
                          <span className="font-mono text-slate-400 text-[10px]">
                            Timing unmeasured
                          </span>
                        )}
                        {stage.started_at && (
                          <span className="text-slate-400">
                            @ {formatTimestamp(stage.started_at)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {isExpanded ? (
                      <ChevronDown className="w-4 h-4 text-slate-400" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-slate-400" />
                    )}
                  </div>
                </button>

                {/* Collapsible Details */}
                {isExpanded && (
                  <div className="px-3.5 pb-3.5 pt-1 border-t border-slate-100 space-y-2 text-xs">
                    {/* Error Callout if Failed */}
                    {stage.error_message && (
                      <div className="p-2.5 rounded-lg bg-rose-50 border border-rose-200 flex items-start gap-2 text-rose-800">
                        <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                        <div>
                          <div className="font-semibold text-rose-900">Stage Error</div>
                          <p className="text-[11px] mt-0.5">{stage.error_message}</p>
                        </div>
                      </div>
                    )}

                    {/* Observable Metadata & Parameters */}
                    {hasDetails && (
                      <div className="bg-slate-50/90 rounded-lg p-2.5 border border-slate-100 space-y-1.5">
                        <div className="text-[11px] font-semibold text-slate-700 uppercase tracking-wider">
                          Observable Stage Metadata
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1 font-mono text-[11px]">
                          {Object.entries(stage.details || {}).map(([key, val]) => (
                            <div key={key} className="flex items-start justify-between gap-2 py-0.5 border-b border-slate-100 last:border-none">
                              <span className="text-slate-500 font-sans">{key}:</span>
                              <span className="text-slate-800 text-right truncate max-w-[200px]" title={String(val)}>
                                {typeof val === 'boolean'
                                  ? val ? 'true' : 'false'
                                  : Array.isArray(val)
                                  ? `[${val.join(', ')}]`
                                  : String(val)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Timestamps Provenance */}
                    <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono px-1">
                      <span>Start: {formatTimestamp(stage.started_at) || 'N/A'}</span>
                      <span>End: {formatTimestamp(stage.completed_at) || 'N/A'}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
