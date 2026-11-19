import React from 'react';
import {
  ShieldCheck,
  AlertTriangle,
  FileQuestion,
  CheckCircle2,
  Info,
  Layers,
} from 'lucide-react';
import { Badge } from '../common/Badge';

interface CitationValidationCardProps {
  validCitations: string[];
  invalidCitations: string[];
  uncitedSources: string[];
  limitationsNote?: string;
  onInspectCitation?: (citationId: string) => void;
}

export const CitationValidationCard: React.FC<CitationValidationCardProps> = ({
  validCitations,
  invalidCitations,
  uncitedSources,
  limitationsNote = 'Citation validation verifies reference existence, not semantic entailment.',
  onInspectCitation,
}) => {
  const hasInvalid = invalidCitations.length > 0;
  const hasValid = validCitations.length > 0;
  const totalCitations = validCitations.length + invalidCitations.length;

  return (
    <div className="space-y-3">
      {/* Integrity Overview Header */}
      <div
        className={`p-3.5 rounded-xl border flex items-start gap-3 ${
          hasInvalid
            ? 'bg-rose-50/60 border-rose-200 text-rose-900'
            : hasValid
            ? 'bg-emerald-50/60 border-emerald-200 text-emerald-900'
            : 'bg-slate-50 border-slate-200 text-slate-800'
        }`}
      >
        {hasInvalid ? (
          <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
        ) : hasValid ? (
          <ShieldCheck className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
        ) : (
          <Info className="w-5 h-5 text-slate-500 shrink-0 mt-0.5" />
        )}

        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <span className="text-xs sm:text-sm font-semibold">
              {hasInvalid
                ? 'Citation Integrity Alert: Unverified References Detected'
                : hasValid
                ? 'Deterministic Citation Verification: Reference Integrity Verified'
                : 'No In-Text Citations Detected'}
            </span>
            <span
              className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full border ${
                hasInvalid
                  ? 'bg-rose-100 text-rose-800 border-rose-300'
                  : hasValid
                  ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
                  : 'bg-slate-200 text-slate-700 border-slate-300'
              }`}
            >
              {hasInvalid ? 'Integrity Warning' : hasValid ? 'References Validated' : 'N/A'}
            </span>
          </div>

          <p className="text-xs mt-1 text-slate-600">
            {hasInvalid
              ? `The answer cited ${invalidCitations.length} reference(s) that do NOT exist in the retrieved knowledge context. These have been flagged in red.`
              : hasValid
              ? `All ${validCitations.length} cited source reference(s) strictly map to verified chunks loaded in the context budget.`
              : 'The generated response or refusal did not cite any source identifiers.'}
          </p>
        </div>
      </div>

      {/* Grid of Citations & Provenance */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Valid Citations */}
        <div className="p-3 rounded-xl border border-emerald-200/80 bg-white shadow-2xs space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-emerald-900 flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              Valid Citations ({validCitations.length})
            </span>
          </div>
          {validCitations.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {validCitations.map((c) => (
                <button
                  key={c}
                  type="button"
                  onClick={() => onInspectCitation && onInspectCitation(c)}
                  className="px-2 py-0.5 rounded font-mono text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 transition-colors cursor-pointer"
                  title="Inspect source in Citation Explorer"
                >
                  {c}
                </button>
              ))}
            </div>
          ) : (
            <div className="text-[11px] text-slate-400 italic">None cited</div>
          )}
        </div>

        {/* Invalid Citations */}
        <div className="p-3 rounded-xl border border-rose-200/80 bg-white shadow-2xs space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-rose-900 flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5 text-rose-600" />
              Invalid References ({invalidCitations.length})
            </span>
          </div>
          {invalidCitations.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {invalidCitations.map((c) => (
                <span
                  key={c}
                  className="px-2 py-0.5 rounded font-mono text-xs font-bold bg-rose-50 text-rose-700 border border-rose-300"
                  title="Fabricated or missing source reference"
                >
                  {c} (Missing)
                </span>
              ))}
            </div>
          ) : (
            <div className="text-[11px] text-emerald-600 font-medium flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3 text-emerald-500" /> 0 invalid references
            </div>
          )}
        </div>

        {/* Uncited Sources */}
        <div className="p-3 rounded-xl border border-slate-200/80 bg-white shadow-2xs space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-slate-500" />
              Uncited Sources ({uncitedSources.length})
            </span>
          </div>
          {uncitedSources.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {uncitedSources.map((c) => (
                <button
                  key={c}
                  type="button"
                  onClick={() => onInspectCitation && onInspectCitation(c)}
                  className="px-2 py-0.5 rounded font-mono text-xs font-medium bg-slate-100 text-slate-600 border border-slate-200 hover:bg-slate-200 transition-colors cursor-pointer"
                  title="Inspect uncited context chunk"
                >
                  {c}
                </button>
              ))}
            </div>
          ) : (
            <div className="text-[11px] text-slate-400 italic">All context cited</div>
          )}
        </div>
      </div>

      {/* Limitations Disclaimer Note */}
      {limitationsNote && (
        <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200/80 flex items-start gap-2 text-[11px] text-slate-500">
          <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
          <span>
            <strong>Validation scope:</strong> {limitationsNote}
          </span>
        </div>
      )}
    </div>
  );
};
