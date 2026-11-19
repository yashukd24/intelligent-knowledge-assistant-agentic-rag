import React from 'react';
import { CitationPill } from './CitationPill';

export interface CitationTextRendererProps {
  text: string;
  validCitations: string[];
  invalidCitations: string[];
  onCitationClick: (sourceId: string) => void;
}

export const CitationTextRenderer: React.FC<CitationTextRendererProps> = ({
  text,
  validCitations,
  invalidCitations,
  onCitationClick,
}) => {
  // Regex to match [S1], [S2], [S12], etc.
  const regex = /(\[S\d+\])/g;
  const parts = text.split(regex);

  return (
    <span className="whitespace-pre-wrap leading-relaxed">
      {parts.map((part, index) => {
        if (regex.test(part)) {
          const isValid = validCitations.includes(part);
          const isInvalid = invalidCitations.includes(part);
          return (
            <CitationPill
              key={index}
              sourceId={part}
              isValid={isValid}
              isInvalid={isInvalid}
              onClick={onCitationClick}
            />
          );
        }
        return <React.Fragment key={index}>{part}</React.Fragment>;
      })}
    </span>
  );
};
