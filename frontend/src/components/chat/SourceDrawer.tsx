import React, { useEffect } from 'react';
import {
  X,
  BookOpen,
  FileText,
  AlertTriangle,
  CheckCircle2,
  Layers,
  Sparkles,
  Info,
  Clock,
} from 'lucide-react';
import { Badge } from '../common/Badge';
import { Button } from '../common/Button';
import type { RAGAnswerResponse, SourceReference } from '../../types/api';

export interface SourceDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  responseMeta: RAGAnswerResponse | null;
  selectedSourceId: string | null;
}

export const SourceDrawer: React.FC<SourceDrawerProps> = ({
  isOpen,
  onClose,
  responseMeta,
  selectedSourceId,
}) => {
  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !responseMeta) return null;

  const sources = responseMeta.sources || [];
  const validCitations = responseMeta.valid_citations || [];
  const invalidCitations = responseMeta.invalid_citations || [];
  const uncitedSources = responseMeta.uncited_sources || [];

  // Sort sources so focused source appears first, or ordered by source_id
  const sortedSources = [...sources].sort((a, b) => {
    if (a.source_id === selectedSourceId) return -1;
    if (b.source_id === selectedSourceId) return 1;
    return a.source_id.localeCompare(b.source_id, undefined, { numeric: true });
  });

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end"
      role="dialog"
      aria-modal="true"
      aria-label="Citation Explorer"
    >
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs transition-opacity animate-in fade-in duration-200"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Drawer Panel */}
      <aside className="relative w-full max-w-lg bg-white h-full shadow-2xl flex flex-col z-10 animate-in slide-in-from-right duration-200 border-l border-slate-200">
        {/* Header */}
        <div className="p-4 sm:p-5 border-b border-slate-100 flex items-center justify-between gap-3 bg-slate-50/50 shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-brand-50 text-brand-600 flex items-center justify-center shrink-0">
              <BookOpen className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-900">Citation Explorer</h3>
              <p className="text-[11px] text-slate-500">
                Retrieved chunks, source verification, and similarity metrics
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200/60 transition-colors"
            aria-label="Close citation explorer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-4">
          {/* Validation Summary Bar */}
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80 space-y-2 text-xs">
            <div className="flex items-center justify-between font-medium text-slate-700">
              <span>Citation Reference Integrity</span>
              <div className="flex items-center gap-1.5">
                {invalidCitations.length === 0 ? (
                  <Badge variant="success" size="sm">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>All Citations Valid</span>
                  </Badge>
                ) : (
                  <Badge variant="danger" size="sm">
                    <AlertTriangle className="w-3 h-3" />
                    <span>{invalidCitations.length} Invalid</span>
                  </Badge>
                )}
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2 pt-1 border-t border-slate-200/60 text-center font-mono text-[11px]">
              <div className="p-1.5 rounded bg-white border border-slate-100">
                <span className="text-slate-400 block text-[10px]">CITED</span>
                <span className="font-bold text-slate-800">{validCitations.length}</span>
              </div>
              <div className="p-1.5 rounded bg-white border border-slate-100">
                <span className="text-slate-400 block text-[10px]">UNCITED</span>
                <span className="font-bold text-amber-700">{uncitedSources.length}</span>
              </div>
              <div className="p-1.5 rounded bg-white border border-slate-100">
                <span className="text-slate-400 block text-[10px]">INVALID</span>
                <span className="font-bold text-rose-700">{invalidCitations.length}</span>
              </div>
            </div>
          </div>

          {/* Invalid Citations Warning Alert */}
          {invalidCitations.length > 0 && (
            <div className="p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs space-y-1.5 animate-in fade-in">
              <div className="flex items-center gap-2 font-semibold">
                <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0" />
                <span>Detected Fabricated / Invalid Citation References</span>
              </div>
              <p className="text-[11px] text-rose-700 leading-relaxed">
                The language model cited{' '}
                <strong>{invalidCitations.join(', ')}</strong> in its answer, but no
                corresponding source exists in the retrieved context. This citation was flagged by
                the reference validation engine.
              </p>
            </div>
          )}

          {/* Source Chunks List */}
          <div className="space-y-3">
            <h4 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
              Retrieved Context Sources ({sortedSources.length})
            </h4>

            {sortedSources.length === 0 ? (
              <div className="p-6 text-center text-xs text-slate-400 border border-dashed rounded-lg">
                No sources available for this response.
              </div>
            ) : (
              sortedSources.map((source) => {
                const isFocused = source.source_id === selectedSourceId;
                const isCited = validCitations.includes(source.source_id);
                const isUncited = uncitedSources.includes(source.source_id);

                return (
                  <div
                    key={source.source_id}
                    className={`rounded-xl border p-4 space-y-3 transition-all ${
                      isFocused
                        ? 'border-brand-500 bg-brand-50/30 ring-2 ring-brand-500/20 shadow-sm'
                        : 'border-slate-200/90 bg-white shadow-2xs hover:border-slate-300'
                    }`}
                  >
                    {/* Source Card Header */}
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-slate-900 text-white">
                          {source.source_id}
                        </span>
                        {isCited && (
                          <Badge variant="success" size="sm">
                            <CheckCircle2 className="w-3 h-3" />
                            <span>Cited in Answer</span>
                          </Badge>
                        )}
                        {isUncited && (
                          <Badge variant="warning" size="sm">
                            <Clock className="w-3 h-3" />
                            <span>Uncited in Context</span>
                          </Badge>
                        )}
                      </div>

                      <span className="text-[11px] font-mono font-semibold text-slate-600 bg-slate-100 px-2 py-0.5 rounded">
                        Score: {source.similarity_score.toFixed(4)}
                      </span>
                    </div>

                    {/* Metadata Details */}
                    <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-600">
                      <div className="flex items-center gap-1.5 truncate">
                        <FileText className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                        <span className="truncate" title={source.filename || 'Unknown file'}>
                          {source.filename || 'Unknown file'}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 justify-end">
                        <Layers className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                        <span>
                          {source.page_number ? `Page ${source.page_number}` : 'Flow format'} • Chunk #{source.chunk_index}
                        </span>
                      </div>
                    </div>

                    {/* Exact Chunk Text Content */}
                    <div className="space-y-1">
                      <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
                        Exact Chunk Text ({source.char_count} chars)
                      </span>
                      <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/70 text-xs text-slate-800 font-serif leading-relaxed max-h-48 overflow-y-auto whitespace-pre-wrap select-text">
                        {source.chunk_text || source.snippet || source.text || '(Full chunk text not captured)'}
                      </div>
                    </div>

                    {/* Provenance Footer */}
                    <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[10px] text-slate-400 font-mono">
                      <span className="truncate max-w-[200px]" title={source.chunk_id}>
                        id: {source.chunk_id}
                      </span>
                      <span>cosine distance: {(1.0 - source.similarity_score).toFixed(4)}</span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Footer Disclaimer */}
        <div className="p-3.5 px-5 border-t border-slate-100 bg-slate-50 text-[11px] text-slate-500 flex items-start gap-2 shrink-0">
          <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
          <p className="leading-tight">
            <strong>Citation Note:</strong> Verification checks reference existence in retrieved context. Semantic entailment is evaluated in Phase 15.
          </p>
        </div>
      </aside>
    </div>
  );
};
