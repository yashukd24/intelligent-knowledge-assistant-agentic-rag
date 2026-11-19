import React, { useState } from 'react';
import {
  FileText,
  Bookmark,
  CheckCircle2,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Sparkles,
} from 'lucide-react';
import type { SourceReference } from '../../types/api';
import { Badge } from '../common/Badge';

interface RetrievalInspectorProps {
  sources: SourceReference[];
  validCitations: string[];
  onOpenSourceInDrawer?: (sourceId: string) => void;
}

export const RetrievalInspector: React.FC<RetrievalInspectorProps> = ({
  sources,
  validCitations,
  onOpenSourceInDrawer,
}) => {
  const [expandedChunkIds, setExpandedChunkIds] = useState<Record<string, boolean>>({});

  const toggleChunk = (id: string) => {
    setExpandedChunkIds((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  if (!sources || sources.length === 0) {
    return (
      <div className="p-6 text-center bg-slate-50 border border-dashed border-slate-200 rounded-xl">
        <Bookmark className="w-8 h-8 text-slate-300 mx-auto mb-2" />
        <h4 className="text-xs font-semibold text-slate-700">No Retrieved Chunks in Context</h4>
        <p className="text-[11px] text-slate-500 mt-1 max-w-sm mx-auto">
          No chunks were retrieved or all candidates were omitted by the similarity threshold.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-xs px-1">
        <span className="font-semibold text-slate-700">
          Retrieved Chunks in Evidence Budget ({sources.length} sources)
        </span>
        <span className="text-[11px] text-slate-500">
          Ranked by Cosine Similarity
        </span>
      </div>

      <div className="space-y-2.5">
        {sources.map((source, index) => {
          const isCited = validCitations.some(
            (c) => c === source.source_id || c === `[${source.source_id.replace(/[[\]]/g, '')}]`
          );
          const isExpanded = !!expandedChunkIds[source.chunk_id];
          const text = source.chunk_text || source.text || source.snippet || '';

          return (
            <div
              key={source.chunk_id || `source-${index}`}
              className={`rounded-xl border transition-all ${
                isCited
                  ? 'bg-emerald-50/20 border-emerald-200/80 shadow-2xs'
                  : 'bg-white border-slate-200/90'
              }`}
            >
              {/* Source Card Header */}
              <div className="p-3 sm:p-3.5 flex items-start justify-between gap-3">
                <div className="flex items-start gap-2.5 min-w-0">
                  <span className="px-2 py-0.5 rounded-md font-mono text-xs font-bold bg-slate-100 text-slate-700 border border-slate-200 shrink-0">
                    {source.source_id}
                  </span>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs font-semibold text-slate-800 truncate" title={source.filename || 'Unknown Document'}>
                        {source.filename || 'Document Chunk'}
                      </span>
                      {isCited ? (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-100/80 text-emerald-800 border border-emerald-300">
                          <CheckCircle2 className="w-2.5 h-2.5 text-emerald-600" />
                          Cited in Final Answer
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-medium bg-slate-100 text-slate-600 border border-slate-200">
                          Uncited (Context Only)
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-2 text-[11px] text-slate-500 mt-1 flex-wrap">
                      <span>
                        Page: <strong className="text-slate-700">{source.page_number ?? (source.page_range || 'N/A')}</strong>
                      </span>
                      <span>•</span>
                      <span>
                        Chunk: <strong className="text-slate-700">#{source.chunk_index}</strong>
                      </span>
                      <span>•</span>
                      <span>
                        Chars: <strong className="text-slate-700">{source.char_count}</strong>
                      </span>
                    </div>
                  </div>
                </div>

                {/* Similarity Score Badge */}
                <div className="text-right shrink-0">
                  <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Similarity</div>
                  <span
                    className={`inline-block font-mono text-xs font-bold px-2 py-0.5 rounded-md border mt-0.5 ${
                      source.similarity_score >= 0.8
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : source.similarity_score >= 0.7
                        ? 'bg-blue-50 text-blue-700 border-blue-200'
                        : 'bg-amber-50 text-amber-700 border-amber-200'
                    }`}
                  >
                    {(source.similarity_score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Text Snippet / Expanded Content */}
              <div className="px-3 sm:px-3.5 pb-3">
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100 text-xs text-slate-700 font-serif leading-relaxed">
                  {isExpanded ? text : text.slice(0, 180) + (text.length > 180 ? '...' : '')}
                </div>

                <div className="flex items-center justify-between pt-2">
                  {text.length > 180 ? (
                    <button
                      type="button"
                      onClick={() => toggleChunk(source.chunk_id)}
                      className="text-[11px] font-medium text-brand-600 hover:text-brand-700 hover:underline inline-flex items-center gap-1 cursor-pointer"
                    >
                      {isExpanded ? (
                        <>
                          <ChevronDown className="w-3 h-3" /> Show less
                        </>
                      ) : (
                        <>
                          <ChevronRight className="w-3 h-3" /> Show complete chunk text ({text.length} chars)
                        </>
                      )}
                    </button>
                  ) : (
                    <div />
                  )}

                  {onOpenSourceInDrawer && (
                    <button
                      type="button"
                      onClick={() => onOpenSourceInDrawer(source.source_id)}
                      className="text-[11px] font-medium text-slate-600 hover:text-brand-600 inline-flex items-center gap-1 cursor-pointer ml-auto"
                    >
                      <ExternalLink className="w-3 h-3" /> View in Citation Explorer
                    </button>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
