import React, { useState, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { fetchCollections } from '../api/collections';
import { submitAgentQuery } from '../api/rag';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { Button } from '../components/common/Button';
import { CitationTextRenderer } from '../components/chat/CitationTextRenderer';
import {
  Workflow,
  Terminal,
  Send,
  Sparkles,
  Layers,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Play,
  RotateCcw,
  Search,
  Calculator,
  ShieldCheck,
  Cpu,
} from 'lucide-react';
import type { AgentRunResponse, AgentEvent, SourceReference } from '../types/api';

interface TraceStep {
  id: string;
  state: 'PLANNING' | 'RETRIEVING' | 'EVALUATING' | 'REFINING' | 'SYNTHESIZING' | 'DONE';
  title: string;
  detail: string;
  durationMs?: number;
  score?: number;
  isRefinement?: boolean;
  payload?: Record<string, unknown>;
}

const SAMPLE_AGENT_TRACE: TraceStep[] = [
  {
    id: 's1',
    state: 'PLANNING',
    title: 'Query Understanding & Intent Decomposition',
    detail: 'Decomposed original multi-part query into 2 atomic sub-queries for precision retrieval.',
    durationMs: 140,
    payload: { intent: 'comparative', subqueries: ['Operating revenue Q3 2025', 'CapEx Q3 2025'] },
  },
  {
    id: 's2',
    state: 'RETRIEVING',
    title: 'Tool Selection: semantic_search (ChromaDB)',
    detail: 'Retrieved top-5 candidate chunks using BAAI/bge-small-en-v1.5 embeddings.',
    durationMs: 85,
    payload: { tool: 'semantic_search', top_k: 5 },
  },
  {
    id: 's3',
    state: 'EVALUATING',
    title: 'Evidence Sufficiency Evaluation',
    detail: 'Sufficiency score evaluated at 0.54 (below 0.70 threshold). Missing context on fiscal liability.',
    durationMs: 210,
    score: 0.54,
    payload: { is_sufficient: false, reason: 'INSUFFICIENT_SUBQUERY_COVERAGE', score: 0.54 },
  },
  {
    id: 's4',
    state: 'REFINING',
    title: 'Adaptive Query Refinement (Round 1/2)',
    detail: 'Refined query to: "financial liability and indemnity clauses in quarterly statement".',
    durationMs: 95,
    isRefinement: true,
    payload: { strategy: 'TARGET_UNCOVERED_SUBQUERIES', adjusted_top_k: 7 },
  },
  {
    id: 's5',
    state: 'RETRIEVING',
    title: 'Targeted Re-Retrieval',
    detail: 'Retrieved 3 supplementary chunks with semantic similarity >= 0.81.',
    durationMs: 78,
    payload: { tool: 'semantic_search', top_k: 7 },
  },
  {
    id: 's6',
    state: 'EVALUATING',
    title: 'Secondary Sufficiency Evaluation',
    detail: 'Sufficiency score reached 0.88 (exceeds 0.70 threshold). Evidence complete.',
    durationMs: 190,
    score: 0.88,
    payload: { is_sufficient: true, reason: 'SUFFICIENT_EVIDENCE', score: 0.88 },
  },
  {
    id: 's7',
    state: 'SYNTHESIZING',
    title: 'Answer Synthesis & Grounding Verification',
    detail: 'Generated grounded answer with 3 valid citations. All assertions cross-verified.',
    durationMs: 820,
    payload: { valid_citations: ['[S1]', '[S2]'], grounding_score: 1.0 },
  },
  {
    id: 's8',
    state: 'DONE',
    title: 'Agent State Finalized',
    detail: 'Answer and structured metadata packaged for frontend streaming.',
    durationMs: 15,
    payload: { status: 'COMPLETED' },
  },
];

export const AgentPage: React.FC = () => {
  const [selectedCollectionId, setSelectedCollectionId] = useState<string>('');
  const [queryInput, setQueryInput] = useState(
    'What was the operating revenue in Q3 2025, and what were the major capital expenditures?'
  );
  const [agentResponse, setAgentResponse] = useState<AgentRunResponse | null>(null);
  const [activeStep, setActiveStep] = useState<TraceStep>(SAMPLE_AGENT_TRACE[2]);

  // Fetch real collections
  const { data: collectionsData, isLoading: colsLoading } = useQuery({
    queryKey: ['collections'],
    queryFn: fetchCollections,
  });

  const collections = collectionsData?.collections || [];

  useEffect(() => {
    if (!selectedCollectionId && collections.length > 0) {
      setSelectedCollectionId(collections[0].id);
    }
  }, [collections, selectedCollectionId]);

  // Agent Run Mutation
  const agentMutation = useMutation({
    mutationFn: async () => {
      if (!selectedCollectionId || !queryInput.trim()) return null;
      return submitAgentQuery(selectedCollectionId, {
        query: queryInput.trim(),
        top_k: 5,
      });
    },
    onSuccess: (data) => {
      if (data) {
        setAgentResponse(data);
        // Build dynamic trace steps from events
        if (data.events && data.events.length > 0) {
          const mapped: TraceStep[] = data.events.map((ev, idx) => {
            let mappedState: TraceStep['state'] = 'PLANNING';
            if (ev.event_type.includes('RETRIEV') || ev.event_type.includes('TOOL')) {
              mappedState = 'RETRIEVING';
            } else if (ev.event_type.includes('EVIDENCE')) {
              mappedState = 'EVALUATING';
            } else if (ev.event_type.includes('REFIN')) {
              mappedState = 'REFINING';
            } else if (ev.event_type.includes('SYNTHESIZ') || ev.event_type.includes('VERIF') || ev.event_type.includes('CITAT')) {
              mappedState = 'SYNTHESIZING';
            } else if (ev.event_type.includes('DONE')) {
              mappedState = 'DONE';
            }

            const score = typeof ev.payload?.evidence_score === 'number'
              ? ev.payload.evidence_score
              : typeof ev.payload?.score === 'number'
              ? ev.payload.score
              : undefined;

            return {
              id: ev.event_id || `ev-${idx}`,
              state: mappedState,
              title: ev.event_type.replace(/_/g, ' '),
              detail: ev.message,
              score,
              isRefinement: ev.event_type.includes('REFIN'),
              payload: ev.payload,
            };
          });
          setActiveStep(mapped[mapped.length - 1] || mapped[0]);
        }
      }
    },
  });

  const currentTraceSteps: TraceStep[] =
    agentResponse?.events && agentResponse.events.length > 0
      ? agentResponse.events.map((ev, idx) => {
          let mappedState: TraceStep['state'] = 'PLANNING';
          if (ev.event_type.includes('RETRIEV') || ev.event_type.includes('TOOL')) {
            mappedState = 'RETRIEVING';
          } else if (ev.event_type.includes('EVIDENCE')) {
            mappedState = 'EVALUATING';
          } else if (ev.event_type.includes('REFIN')) {
            mappedState = 'REFINING';
          } else if (ev.event_type.includes('SYNTHESIZ') || ev.event_type.includes('VERIF') || ev.event_type.includes('CITAT')) {
            mappedState = 'SYNTHESIZING';
          } else if (ev.event_type.includes('DONE')) {
            mappedState = 'DONE';
          }

          const score = typeof ev.payload?.evidence_score === 'number'
            ? ev.payload.evidence_score
            : typeof ev.payload?.score === 'number'
            ? ev.payload.score
            : undefined;

          return {
            id: ev.event_id || `ev-${idx}`,
            state: mappedState,
            title: ev.event_type.replace(/_/g, ' '),
            detail: ev.message,
            score,
            isRefinement: ev.event_type.includes('REFIN'),
            payload: ev.payload,
          };
        })
      : SAMPLE_AGENT_TRACE;

  return (
    <div className="space-y-6">
      {/* Architecture Disclaimer */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200/80 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2.5">
          <Workflow className="w-5 h-5 text-purple-700 shrink-0" />
          <div>
            <span className="font-semibold text-purple-900 block">
              Phase 10–15 Architecture: Plain Python Agentic State Machine
            </span>
            <span className="text-purple-700">
              Zero LangChain, zero LangGraph. Complete iterative loop with decomposition, tool selection, evidence evaluation, adaptive refinement, and grounding verification.
            </span>
          </div>
        </div>
        <Badge variant="primary" className="bg-purple-100 text-purple-800 border-purple-200 shrink-0">
          {agentResponse ? 'Live Controller Run' : 'Preview Mode'}
        </Badge>
      </div>

      {/* Execution Controls Card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Agentic RAG Query Execution</CardTitle>
          <CardDescription>
            Select a knowledge collection and dispatch an iterative agent run.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-700">Knowledge Collection</label>
              <select
                value={selectedCollectionId}
                onChange={(e) => setSelectedCollectionId(e.target.value)}
                disabled={colsLoading || agentMutation.isPending}
                className="w-full text-xs rounded-lg border border-slate-300 p-2.5 bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-brand-500"
              >
                {collections.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name} ({c.document_count} doc{c.document_count !== 1 ? 's' : ''})
                  </option>
                ))}
                {collections.length === 0 && (
                  <option value="">No collections found</option>
                )}
              </select>
            </div>

            <div className="md:col-span-2 space-y-1">
              <label className="text-xs font-semibold text-slate-700">Question / Research Directive</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={queryInput}
                  onChange={(e) => setQueryInput(e.target.value)}
                  disabled={agentMutation.isPending}
                  placeholder="Enter a complex, multi-part, or factual query..."
                  className="flex-1 text-xs rounded-lg border border-slate-300 px-3 py-2 bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-brand-500"
                />
                <Button
                  onClick={() => agentMutation.mutate()}
                  disabled={agentMutation.isPending || !selectedCollectionId || !queryInput.trim()}
                  className="bg-brand-600 hover:bg-brand-700 text-white shrink-0 text-xs px-4 py-2"
                >
                  {agentMutation.isPending ? (
                    <>
                      <RotateCcw className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                      Running...
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5 mr-1.5" />
                      Run Agent
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>

          {/* Decomposed Subqueries & Active Agent Meta */}
          {agentResponse && (
            <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-2 text-xs">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-slate-700">Status:</span>
                  <Badge variant={agentResponse.status === 'COMPLETED' ? 'success' : 'default'} size="sm">
                    {agentResponse.status}
                  </Badge>
                  <span className="text-slate-400">|</span>
                  <span className="text-slate-600">
                    Latency: <strong>{agentResponse.total_latency_ms.toFixed(1)}ms</strong>
                  </span>
                  <span className="text-slate-400">|</span>
                  <span className="text-slate-600">
                    Refinements: <strong>{agentResponse.iteration_count}</strong>
                  </span>
                </div>

                {agentResponse.evidence_evaluation && (
                  <Badge
                    variant={agentResponse.evidence_evaluation.is_sufficient ? 'success' : 'warning'}
                    size="sm"
                  >
                    Evidence: {agentResponse.evidence_evaluation.reason} (Score: {(agentResponse.evidence_evaluation.overall_score * 100).toFixed(0)}%)
                  </Badge>
                )}
              </div>

              {agentResponse.decomposed_queries && agentResponse.decomposed_queries.length > 0 && (
                <div>
                  <span className="text-[11px] font-semibold text-slate-500 block mb-1">
                    Decomposed Subqueries ({agentResponse.decomposed_queries.length}):
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {agentResponse.decomposed_queries.map((sq, i) => (
                      <span key={i} className="px-2 py-0.5 bg-white border border-slate-200 rounded text-[11px] text-slate-700">
                        {i + 1}. {sq}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Synthesized Answer Display */}
              <div className="mt-2 pt-2 border-t border-slate-200/80">
                <span className="text-[11px] font-semibold text-brand-900 block mb-1">
                  Synthesized Grounded Answer:
                </span>
                <div className="bg-white p-3 rounded border border-slate-200 text-slate-800 leading-relaxed text-xs">
                  <CitationTextRenderer
                    text={agentResponse.answer}
                    validCitations={agentResponse.valid_citations}
                    invalidCitations={agentResponse.invalid_citations}
                    onCitationClick={() => {}}
                  />
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* State Pipeline Stepper */}
      <Card>
        <CardHeader>
          <CardTitle>Agent State Lifecycle</CardTitle>
          <CardDescription>
            The iterative state loop ensures retrieved evidence meets explicit sufficiency thresholds before answer synthesis.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-center text-xs">
            {[
              { name: 'PLANNING', desc: 'Query intent' },
              { name: 'RETRIEVING', desc: 'Tool search' },
              { name: 'EVALUATING', desc: 'Sufficiency check' },
              { name: 'REFINING', desc: 'Adaptive loop' },
              { name: 'SYNTHESIZING', desc: 'Grounded draft' },
              { name: 'DONE', desc: 'Verified answer' },
            ].map((st, idx) => {
              const isCurrent = activeStep.state === st.name;
              return (
                <div
                  key={st.name}
                  className={`p-3 rounded-lg border transition-all ${
                    isCurrent
                      ? 'border-brand-500 bg-brand-50 text-brand-900 shadow-sm'
                      : 'border-slate-200 bg-slate-50/70 text-slate-600'
                  }`}
                >
                  <div className="text-[10px] font-mono text-slate-400">0{idx + 1}</div>
                  <div className="font-bold">{st.name}</div>
                  <div className="text-[10px] text-slate-500">{st.desc}</div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Main Grid: Trace History + State Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trace Event Log */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Interactive Agent Execution Trace</CardTitle>
                <CardDescription>Click any event to inspect state variables</CardDescription>
              </div>
              <Badge variant="outline" className="font-mono text-[10px]">
                {currentTraceSteps.length} Events Logged
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div className="divide-y divide-slate-100 max-h-[500px] overflow-y-auto">
              {currentTraceSteps.map((step) => {
                const isSelected = activeStep.id === step.id;
                return (
                  <div
                    key={step.id}
                    onClick={() => setActiveStep(step)}
                    className={`p-4 flex items-start justify-between gap-3 cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-brand-50/70 border-l-4 border-l-brand-600'
                        : 'hover:bg-slate-50 border-l-4 border-l-transparent'
                    }`}
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={
                            step.state === 'EVALUATING'
                              ? step.score && step.score >= 0.7
                                ? 'success'
                                : 'warning'
                              : step.state === 'REFINING'
                              ? 'primary'
                              : 'default'
                          }
                          size="sm"
                          className="font-mono text-[10px]"
                        >
                          {step.state}
                        </Badge>
                        <span className="text-xs font-semibold text-slate-900">
                          {step.title}
                        </span>
                      </div>
                      <p className="text-xs text-slate-600 pl-1">{step.detail}</p>
                    </div>

                    <div className="text-right shrink-0">
                      {step.durationMs !== undefined && (
                        <span className="text-[10px] font-mono text-slate-400">
                          {step.durationMs}ms
                        </span>
                      )}
                      {step.score !== undefined && (
                        <div className="mt-1">
                          <span
                            className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                              step.score >= 0.7
                                ? 'bg-emerald-100 text-emerald-800'
                                : 'bg-amber-100 text-amber-800'
                            }`}
                          >
                            Score: {step.score.toFixed(2)}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* State Snapshot Inspector */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Terminal className="w-4 h-4 text-brand-600" />
                <span>Pydantic AgentState Snapshot</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 space-y-3 text-xs">
              <div>
                <span className="text-slate-400 block font-mono text-[10px]">CURRENT STATE</span>
                <span className="font-bold text-slate-900 font-mono text-sm">
                  {activeStep.state}
                </span>
              </div>

              <div>
                <span className="text-slate-400 block font-mono text-[10px]">EVENT TITLE</span>
                <span className="font-medium text-slate-800">{activeStep.title}</span>
              </div>

              <div>
                <span className="text-slate-400 block font-mono text-[10px]">LOGGED DETAIL</span>
                <p className="text-slate-600 mt-0.5 bg-slate-50 p-2 rounded border border-slate-200/60 font-mono text-[11px]">
                  {activeStep.detail}
                </p>
              </div>

              {activeStep.payload && (
                <div>
                  <span className="text-slate-400 block font-mono text-[10px]">STRUCTURED EVENT PAYLOAD</span>
                  <pre className="text-slate-700 mt-0.5 bg-slate-100 p-2 rounded border border-slate-200 font-mono text-[10px] overflow-x-auto max-h-40">
                    {JSON.stringify(activeStep.payload, null, 2)}
                  </pre>
                </div>
              )}

              {activeStep.score !== undefined && (
                <div>
                  <span className="text-slate-400 block font-mono text-[10px]">EVIDENCE SCORE</span>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="flex-1 bg-slate-200 h-2 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          activeStep.score >= 0.7 ? 'bg-emerald-500' : 'bg-amber-500'
                        }`}
                        style={{ width: `${Math.min(100, activeStep.score * 100)}%` }}
                      />
                    </div>
                    <span className="font-mono text-xs font-bold text-slate-700">
                      {(activeStep.score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <span className="text-[10px] text-slate-400 mt-1 block">
                    Threshold: &ge; 50% required for synthesis
                  </span>
                </div>
              )}

              <div className="pt-2 border-t border-slate-100 flex items-center justify-between">
                <span className="text-slate-400 block font-mono text-[10px]">CONTROLLER ENGINE</span>
                <span className="text-brand-700 font-mono text-[11px] font-semibold">Plain Python (No LangChain)</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
