import React from 'react';
import { BookOpen, AlertTriangle } from 'lucide-react';
import { cn } from '../../lib/utils';

export interface CitationPillProps {
  sourceId: string;
  isValid: boolean;
  isInvalid: boolean;
  onClick: (sourceId: string) => void;
}

export const CitationPill: React.FC<CitationPillProps> = ({
  sourceId,
  isValid,
  isInvalid,
  onClick,
}) => {
  return (
    <button
      type="button"
      onClick={() => onClick(sourceId)}
      className={cn(
        'inline-flex items-center gap-1 px-1.5 py-0.5 mx-0.5 rounded text-xs font-mono font-semibold transition-all select-none border',
        isInvalid
          ? 'bg-rose-50 text-rose-700 border-rose-300 hover:bg-rose-100 shadow-2xs'
          : isValid
          ? 'bg-brand-50 text-brand-700 border-brand-300 hover:bg-brand-100 shadow-2xs'
          : 'bg-slate-100 text-slate-700 border-slate-300 hover:bg-slate-200'
      )}
      title={
        isInvalid
          ? `${sourceId}: Invalid citation reference (not in retrieved context)`
          : `${sourceId}: Verified source reference (click to view chunk)`
      }
    >
      {isInvalid ? (
        <AlertTriangle className="w-3 h-3 text-rose-600 shrink-0" />
      ) : (
        <BookOpen className="w-3 h-3 text-brand-600 shrink-0" />
      )}
      <span>{sourceId}</span>
    </button>
  );
};
