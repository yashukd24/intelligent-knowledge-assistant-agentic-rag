import React, { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { fetchCollections } from '../api/collections';
import { submitRAGQuery, submitAgentQuery, mockRAGQuery, type MockScenarioType } from '../api/rag';
import { Card, CardContent } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { Badge } from '../components/common/Badge';
import { CitationTextRenderer } from '../components/chat/CitationTextRenderer';
import { SourceDrawer } from '../components/chat/SourceDrawer';
import { TraceDrawer } from '../components/trace/TraceDrawer';
import {
  Send,
  Bot,
  User,
  ShieldAlert,
  Sparkles,
  Layers,
  Clock,
  CheckCircle2,
  AlertTriangle,
  FolderArchive,
  BookOpen,
  Copy,
  Check,
  RefreshCw,
  AlertCircle,
  HelpCircle,
  TestTube2,
  Cpu,
  Activity,
  SlidersHorizontal,
} from 'lucide-react';
import type { ApiError } from '../api/client';
import type { RAGAnswerResponse, SourceReference } from '../types/api';

interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  content: string;
  timestamp: string;
  responseMeta?: RAGAnswerResponse;
  isError?: boolean;
}

type PipelineMode = 'conventional_live' | 'mock_safe' | 'agentic_preview';
type UserRole = 'student' | 'faculty' | 'new_admission' | 'administrator';

const ROLE_PRESETS: Record<UserRole, { label: string; icon: string; questions: string[] }> = {
  student: {
    label: 'Student',
    icon: '🎓',
    questions: [
      'Where is the CSE HOD cabin?',
      'What are the placement eligibility conditions?',
      'What are the internship rules?',
      'What is the minimum attendance requirement?',
      'How do I apply for a scholarship?',
    ],
  },
  faculty: {
    label: 'Faculty',
    icon: '👨‍🏫',
    questions: [
      'What is the faculty leave procedure?',
      'Where can I find the academic calendar?',
      'What are the examination duties?',
      'What are the department procedures?',
    ],
  },
  new_admission: {
    label: 'New Admission',
    icon: '🏛️',
    questions: [
      'What courses are available?',
      'What are the admission eligibility requirements?',
      'What documents are needed for admission?',
      'What facilities are available on campus?',
    ],
  },
  administrator: {
    label: 'Administrator',
    icon: '🛡️',
    questions: [
      'What institutional policies are uploaded?',
      'Which documents need updating?',
      'Which departments have missing information?',
    ],
  },
};

export const ChatPage: React.FC = () => {
  const [selectedCollectionId, setSelectedCollectionId] = useState<string>('');
  const [pipelineMode, setPipelineMode] = useState<PipelineMode>('agentic_preview');
  const [mockScenario, setMockScenario] = useState<MockScenarioType>('grounded');
  const [userRole, setUserRole] = useState<UserRole>('student');
  const [inputQuery, setInputQuery] = useState(
    'Where is the CSE HOD cabin?'
  );
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Citation Explorer Drawer State
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [drawerResponseMeta, setDrawerResponseMeta] = useState<RAGAnswerResponse | null>(null);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);

  // Activity Trace Visualizer Drawer State
  const [isTraceDrawerOpen, setIsTraceDrawerOpen] = useState(false);
  const [traceResponseMeta, setTraceResponseMeta] = useState<RAGAnswerResponse | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      content:
        'Welcome to the Intelligent Knowledge Assistant. Ask questions grounded strictly in your uploaded knowledge collections. Every factual statement is backed by verifiable source citations [S1], [S2].',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);

  // Fetch collections
  const { data: collectionsData, isLoading: colsLoading } = useQuery({
    queryKey: ['collections'],
    queryFn: fetchCollections,
  });

  const collections = collectionsData?.collections || [];
  const selectedCollection = collections.find((c) => c.id === selectedCollectionId);
  const hasIndexedDocs = (selectedCollection?.document_count ?? 0) > 0;

  // Auto-select first collection (prefer 'college')
  useEffect(() => {
    if ((!selectedCollectionId || selectedCollectionId === 'demo-knowledge-collection') && collections.length > 0) {
      const collegeCol = collections.find((c) => c.name.toLowerCase() === 'college');
      setSelectedCollectionId(collegeCol ? collegeCol.id : collections[0].id);
    }
  }, [collections, selectedCollectionId]);

  // Auto-scroll on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // RAG Query Mutation
  const ragMutation = useMutation({
    mutationFn: async ({ query, mode, scenario }: { query: string; mode: PipelineMode; scenario: MockScenarioType }) => {
      if (mode === 'mock_safe') {
        return mockRAGQuery(scenario, selectedCollectionId, query);
      } else if (mode === 'agentic_preview') {
        const agentResp = await submitAgentQuery(selectedCollectionId, { query, top_k: 5, user_role: userRole });
        return {
          query: agentResp.query,
          answer: agentResp.answer,
          status: agentResp.status === 'COMPLETED' ? ('answered' as const) : ('insufficient_evidence' as const),
          collection_id: agentResp.collection_id,
          sources: agentResp.sources,
          valid_citations: agentResp.valid_citations,
          invalid_citations: agentResp.invalid_citations,
          uncited_sources: [],
          provider: 'Agent Controller (Plain Python)',
          model: 'claude-3-5-sonnet',
          generation_latency_ms: agentResp.total_latency_ms,
          total_latency_ms: agentResp.total_latency_ms,
          limitations_note: agentResp.limitations_note,
          execution_trace: agentResp.execution_trace,
        };
      } else {
        return submitRAGQuery(selectedCollectionId, { query, top_k: 5 });
      }
    },
    onSuccess: (data) => {
      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        sender: 'assistant',
        content: data.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        responseMeta: data,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    },
    onError: (error: ApiError) => {
      const errorMessage: ChatMessage = {
        id: `err-${Date.now()}`,
        sender: 'assistant',
        content: `Query failed: ${error.message}. If querying a live collection, ensure it contains indexed documents.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    },
  });

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    const query = inputQuery.trim();
    if (!query || !selectedCollectionId || ragMutation.isPending) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputQuery('');

    if (pipelineMode === 'agentic_preview') {
      setTimeout(() => {
        setMessages((prev) => [
          ...prev,
          {
            id: `agentic-notice-${Date.now()}`,
            sender: 'assistant',
            content:
              '[Agentic State Machine Preview]: Evaluating query intent and initiating multi-step retrieval. Phase 10 agent state machine is currently scheduled for implementation. Executing Conventional RAG pipeline for verified answer synthesis.',
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          },
        ]);
        ragMutation.mutate({ query, mode: 'mock_safe', scenario: 'grounded' });
      }, 350);
    } else {
      ragMutation.mutate({ query, mode: pipelineMode, scenario: mockScenario });
    }
  };

  const handleCopyAnswer = (text: string, msgId: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(msgId);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleOpenCitation = (sourceId: string, meta: RAGAnswerResponse) => {
    setSelectedSourceId(sourceId);
    setDrawerResponseMeta(meta);
    setIsDrawerOpen(true);
  };

  const handleOpenSourceDrawer = (meta: RAGAnswerResponse) => {
    setSelectedSourceId(null);
    setDrawerResponseMeta(meta);
    setIsDrawerOpen(true);
  };

  const handleOpenTrace = (meta: RAGAnswerResponse) => {
    setTraceResponseMeta(meta);
    setIsTraceDrawerOpen(true);
  };

  const handleClearSession = () => {
    setMessages([
      {
        id: `welcome-${Date.now()}`,
        sender: 'assistant',
        content: 'New chat session started. Select a knowledge collection and ask a question.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8.5rem)]">
      {/* Control Bar: Collection Picker + Mode Toggle */}
      <div className="bg-white border border-slate-200/90 rounded-xl p-3 sm:p-4 mb-3 flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3 shadow-2xs shrink-0">
        {/* Collection Selector */}
        <div className="flex items-center gap-2.5 min-w-0">
          <FolderArchive className="w-4 h-4 text-slate-500 shrink-0" />
          <label className="text-xs font-semibold text-slate-700 shrink-0">Collection:</label>
          <select
            value={selectedCollectionId}
            onChange={(e) => setSelectedCollectionId(e.target.value)}
            disabled={colsLoading}
            className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-brand-500 font-medium text-slate-800 min-w-[180px] max-w-[280px] truncate"
          >
            {collections.length === 0 ? (
              <option value="demo-knowledge-collection">Demo Knowledge Collection (RAM-Safe)</option>
            ) : (
              collections.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.document_count} {c.document_count === 1 ? 'doc' : 'docs'})
                </option>
              ))
            )}
          </select>
        </div>

        {/* Role Selector (Phase 23) */}
        <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg border border-slate-200 text-xs">
          {(['student', 'faculty', 'new_admission', 'administrator'] as UserRole[]).map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => {
                setUserRole(r);
                setInputQuery(ROLE_PRESETS[r].questions[0]);
              }}
              className={`px-2 py-1 rounded-md font-medium transition-all flex items-center gap-1 ${
                userRole === r
                  ? 'bg-white text-brand-700 shadow-2xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <span>{ROLE_PRESETS[r].icon}</span>
              <span className="capitalize">{ROLE_PRESETS[r].label}</span>
            </button>
          ))}
        </div>

        {/* Pipeline Mode Toggle */}
        <div className="flex flex-wrap items-center gap-2 self-start md:self-auto">
          <div className="bg-slate-100 p-0.5 rounded-lg flex items-center border border-slate-200 text-xs">
            <button
              type="button"
              onClick={() => setPipelineMode('mock_safe')}
              className={`px-2.5 py-1 rounded-md font-medium transition-all flex items-center gap-1 ${
                pipelineMode === 'mock_safe'
                  ? 'bg-white text-emerald-700 shadow-2xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <TestTube2 className="w-3 h-3 text-emerald-600" />
              <span>Safe Test Mode</span>
            </button>

            <button
              type="button"
              onClick={() => setPipelineMode('conventional_live')}
              className={`px-2.5 py-1 rounded-md font-medium transition-all flex items-center gap-1 ${
                pipelineMode === 'conventional_live'
                  ? 'bg-white text-brand-700 shadow-2xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Cpu className="w-3 h-3 text-brand-600" />
              <span>Live API</span>
            </button>

            <button
              type="button"
              onClick={() => setPipelineMode('agentic_preview')}
              className={`px-2.5 py-1 rounded-md font-medium transition-all flex items-center gap-1 ${
                pipelineMode === 'agentic_preview'
                  ? 'bg-white text-purple-700 shadow-2xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Sparkles className="w-3 h-3 text-purple-600" />
              <span>Agentic RAG</span>
              <span className="text-[9px] bg-purple-100 text-purple-700 px-1 py-0.2 rounded font-mono">
                Preview
              </span>
            </button>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={handleClearSession}
            className="text-xs h-7 px-2 shrink-0"
            title="Clear conversation history"
          >
            <RefreshCw className="w-3 h-3 mr-1" />
            Clear
          </Button>
        </div>
      </div>

      {/* Test Scenario Selector Bar (Shown in Safe Test Mode) */}
      {pipelineMode === 'mock_safe' && (
        <div className="p-2.5 px-3.5 mb-2.5 rounded-lg bg-emerald-50/70 border border-emerald-200/80 text-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-2.5 shrink-0 animate-in fade-in">
          <div className="flex items-center gap-2 text-emerald-800">
            <TestTube2 className="w-4 h-4 text-emerald-600 shrink-0" />
            <span className="font-semibold">RAM-Safe Test Scenarios:</span>
            <span className="text-[11px] text-emerald-700 hidden lg:inline">
              Deterministic RAG execution traces with zero model memory footprint
            </span>
          </div>

          <div className="flex items-center gap-1.5 flex-wrap text-[11px]">
            <select
              value={mockScenario}
              onChange={(e) => {
                const scen = e.target.value as MockScenarioType;
                setMockScenario(scen);
                switch (scen) {
                  case 'grounded':
                    setInputQuery('What was the operating revenue and capital expenditure for Q3 2025?');
                    break;
                  case 'invalid_citation':
                    setInputQuery('What are the secondary liability obligations described in the report?');
                    break;
                  case 'insufficient_evidence':
                    setInputQuery('What were the quantum computing research expenditures in 2024?');
                    break;
                  case 'empty_collection':
                    setInputQuery('Summarize the policies in the empty collection.');
                    break;
                  case 'no_docs':
                    setInputQuery('Retrieve documents on non-existent astronomical entities.');
                    break;
                  case 'provider_error':
                    setInputQuery('Explain distributed database failover guarantees.');
                    break;
                  case 'timeout':
                    setInputQuery('Generate a full multi-chapter technical whitepaper.');
                    break;
                  case 'partial_failure':
                    setInputQuery('Process corrupted high-dimensional embedding vectors.');
                    break;
                }
              }}
              className="text-xs bg-white border border-emerald-300 rounded-md px-2 py-1 font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-emerald-500 shadow-2xs"
            >
              <option value="grounded">1. Grounded Answer [S1, S2]</option>
              <option value="invalid_citation">2. Invalid Citation Reference [S9]</option>
              <option value="insufficient_evidence">3. Insufficient Evidence Refusal</option>
              <option value="empty_collection">4. Empty Collection (0 Chunks)</option>
              <option value="no_docs">5. No Documents Found</option>
              <option value="provider_error">6. Provider Error (HTTP 502)</option>
              <option value="timeout">7. LLM Timeout Handling (120s)</option>
              <option value="partial_failure">8. Partial Pipeline Failure (Stage 5)</option>
            </select>

            <span className="text-[10px] text-emerald-800 font-mono bg-emerald-100/80 px-2 py-0.5 rounded">
              Selected: {mockScenario}
            </span>
          </div>
        </div>
      )}

      {/* Empty Collection Warning Notice */}
      {!hasIndexedDocs && collections.length > 0 && pipelineMode === 'conventional_live' && (
        <div className="p-3 mb-2.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
            <span>
              <strong>Empty Collection:</strong> "{selectedCollection?.name}" has 0 indexed documents. Live queries will return an insufficient evidence refusal.
            </span>
          </div>
          <Badge variant="warning" size="sm">0 Docs</Badge>
        </div>
      )}

      {/* RAM Safety Threshold Advisory Banner */}
      <div className="mb-3 p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-slate-700 text-xs flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-brand-600 shrink-0" />
          <span>
            <strong>Safety Constraint:</strong> Live Ollama Qwen 2.5 3B inference requires &ge; 5.0 GB available RAM. Citation references strictly verify retrieved context chunks.
          </span>
        </div>
        <span className="font-mono text-[10px] text-slate-500 font-semibold hidden md:inline">
          Threshold: 5.0 GB
        </span>
      </div>

      {/* Chat Messages Scrollable Container */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1 mb-3">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex items-start gap-3 ${
              msg.sender === 'user' ? 'flex-row-reverse' : 'flex-row'
            }`}
          >
            {/* Avatar */}
            <div
              className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                msg.sender === 'user'
                  ? 'bg-slate-800 text-white'
                  : msg.isError
                  ? 'bg-rose-600 text-white'
                  : 'bg-brand-600 text-white shadow-2xs'
              }`}
            >
              {msg.sender === 'user' ? (
                <User className="w-4 h-4" />
              ) : msg.isError ? (
                <AlertCircle className="w-4 h-4" />
              ) : (
                <Bot className="w-4 h-4" />
              )}
            </div>

            {/* Bubble */}
            <div
              className={`max-w-[88%] sm:max-w-[80%] rounded-2xl p-4 text-sm leading-relaxed ${
                msg.sender === 'user'
                  ? 'bg-brand-600 text-white rounded-tr-xs'
                  : msg.isError
                  ? 'bg-rose-50 border border-rose-200 text-rose-900 rounded-tl-xs shadow-2xs'
                  : msg.responseMeta?.status === 'insufficient_evidence'
                  ? 'bg-amber-50/70 border border-amber-200 text-amber-900 rounded-tl-xs shadow-2xs'
                  : 'bg-white border border-slate-200/90 text-slate-800 rounded-tl-xs shadow-2xs'
              }`}
            >
              {/* Answer Content with Interactive Citations */}
              {msg.sender === 'assistant' && msg.responseMeta ? (
                <div className="space-y-3">
                  <CitationTextRenderer
                    text={msg.content}
                    validCitations={msg.responseMeta.valid_citations || []}
                    invalidCitations={msg.responseMeta.invalid_citations || []}
                    onCitationClick={(srcId) => handleOpenCitation(srcId, msg.responseMeta!)}
                  />

                  {/* Insufficient Evidence Notice */}
                  {msg.responseMeta.status === 'insufficient_evidence' && (
                    <div className="mt-2 p-2.5 rounded-lg bg-amber-100/60 border border-amber-300/80 text-amber-800 text-xs flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                      <p>
                        <strong>Refusal Triggered:</strong> The retrieval engine found no candidate chunks exceeding the similarity threshold in this collection.
                      </p>
                    </div>
                  )}

                  {/* Invalid Citation Alert Callout */}
                  {msg.responseMeta.invalid_citations?.length > 0 && (
                    <div className="p-2.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center justify-between">
                      <div className="flex items-center gap-1.5 font-medium">
                        <AlertTriangle className="w-3.5 h-3.5 text-rose-600 shrink-0" />
                        <span>Flagged invalid citation: {msg.responseMeta.invalid_citations.join(', ')}</span>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleOpenSourceDrawer(msg.responseMeta!)}
                        className="text-xs font-semibold text-rose-700 hover:text-rose-900 underline ml-2"
                      >
                        Inspect
                      </button>
                    </div>
                  )}

                  {/* Diagnostics & Citation Explorer Trigger Bar */}
                  <div className="mt-3 pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between gap-2 text-xs">
                    <div className="flex flex-wrap items-center gap-1.5">
                      {/* Activity Trace Visualizer Button */}
                      <button
                        type="button"
                        onClick={() => handleOpenTrace(msg.responseMeta!)}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-purple-50 hover:bg-purple-100 text-purple-700 hover:text-purple-900 border border-purple-200 font-semibold transition-colors cursor-pointer"
                        title="Inspect 9-stage execution timeline and retrieval reasoning"
                      >
                        <Activity className="w-3.5 h-3.5 text-purple-600" />
                        <span>Activity Trace</span>
                        <span className="text-[10px] bg-purple-200/80 text-purple-800 px-1 py-0.2 rounded font-mono font-bold">
                          {msg.responseMeta.execution_trace?.stages
                            ? `${msg.responseMeta.execution_trace.stages.filter((s) => s.status === 'completed').length}/9`
                            : '9/9'}
                        </span>
                      </button>

                      {msg.responseMeta.sources?.length > 0 && (
                        <button
                          type="button"
                          onClick={() => handleOpenSourceDrawer(msg.responseMeta!)}
                          className="inline-flex items-center gap-1 px-2 py-1 rounded bg-slate-100 hover:bg-brand-50 text-slate-700 hover:text-brand-700 border border-slate-200 hover:border-brand-300 font-medium transition-colors cursor-pointer"
                        >
                          <BookOpen className="w-3.5 h-3.5 text-brand-600" />
                          <span>Sources ({msg.responseMeta.sources.length})</span>
                        </button>
                      )}

                      {msg.responseMeta.valid_citations?.length > 0 && (
                        <Badge variant="success" size="sm">
                          <CheckCircle2 className="w-3 h-3" />
                          <span>{msg.responseMeta.valid_citations.length} Verified</span>
                        </Badge>
                      )}

                      {msg.responseMeta.generation_latency_ms ? (
                        <span className="text-[11px] font-mono text-slate-400">
                          {(msg.responseMeta.generation_latency_ms / 1000).toFixed(2)}s
                        </span>
                      ) : null}
                    </div>

                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        onClick={() => handleCopyAnswer(msg.content, msg.id)}
                        className="inline-flex items-center gap-1 p-1 px-2 rounded text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition-colors text-[11px]"
                        title="Copy answer to clipboard"
                      >
                        {copiedId === msg.id ? (
                          <>
                            <Check className="w-3 h-3 text-emerald-600" />
                            <span className="text-emerald-600 font-medium">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3" />
                            <span>Copy</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="whitespace-pre-wrap">{msg.content}</div>
              )}

              <div
                className={`text-[10px] mt-1.5 ${
                  msg.sender === 'user' ? 'text-brand-100 text-right' : 'text-slate-400'
                }`}
              >
                {msg.timestamp}
              </div>
            </div>
          </div>
        ))}

        {/* Loading Indicator */}
        {ragMutation.isPending && (
          <div className="flex items-start gap-3 animate-in fade-in">
            <div className="w-8 h-8 rounded-lg bg-brand-600 text-white flex items-center justify-center shrink-0">
              <Bot className="w-4 h-4" />
            </div>
            <div className="bg-white border border-slate-200/90 rounded-2xl p-4 text-xs text-slate-600 flex items-center gap-2.5 shadow-2xs">
              <div className="flex gap-1">
                <span className="w-2 h-2 rounded-full bg-brand-500 animate-bounce" />
                <span className="w-2 h-2 rounded-full bg-brand-500 animate-bounce [animation-delay:0.2s]" />
                <span className="w-2 h-2 rounded-full bg-brand-500 animate-bounce [animation-delay:0.4s]" />
              </div>
              <span className="font-medium">
                Searching vector store & validating citations...
              </span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Questions Pill Bar (Phase 23) */}
      <div className="flex flex-wrap items-center gap-1.5 py-1.5 px-1 shrink-0">
        <span className="text-[11px] font-semibold text-slate-500 mr-1 flex items-center gap-1 shrink-0">
          <span>{ROLE_PRESETS[userRole].icon}</span>
          <span>{ROLE_PRESETS[userRole].label} Prompts:</span>
        </span>
        {ROLE_PRESETS[userRole].questions.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => setInputQuery(q)}
            className="text-[11px] bg-slate-100 hover:bg-brand-50 text-slate-700 hover:text-brand-700 border border-slate-200 hover:border-brand-300 rounded-md px-2 py-0.5 transition-colors cursor-pointer shrink-0"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Query Input Bar */}
      <form onSubmit={handleSend} className="relative shrink-0">
        <input
          type="text"
          placeholder={
            selectedCollectionId
              ? pipelineMode === 'mock_safe'
                ? 'Type a question or select a scenario above...'
                : hasIndexedDocs
                ? 'Ask a question about the documents in this collection...'
                : 'Upload documents to this collection to ask live questions...'
              : 'Select or create a collection first to enable querying...'
          }
          disabled={!selectedCollectionId || ragMutation.isPending}
          value={inputQuery}
          onChange={(e) => setInputQuery(e.target.value)}
          className="w-full pl-4 pr-24 py-3 bg-white border border-slate-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 shadow-2xs disabled:bg-slate-100 disabled:text-slate-400"
        />
        <Button
          type="submit"
          variant="primary"
          size="sm"
          disabled={!inputQuery.trim() || !selectedCollectionId || ragMutation.isPending}
          className="absolute right-2 top-1/2 -translate-y-1/2 h-8 px-3 text-xs"
        >
          <Send className="w-3.5 h-3.5 mr-1" />
          <span>Ask</span>
        </Button>
      </form>

      {/* Citation Explorer Slide-Over Drawer */}
      <SourceDrawer
        isOpen={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
        responseMeta={drawerResponseMeta}
        selectedSourceId={selectedSourceId}
      />

      {/* Activity Trace & Reasoning Visualizer Slide-Over Drawer */}
      <TraceDrawer
        isOpen={isTraceDrawerOpen}
        onClose={() => setIsTraceDrawerOpen(false)}
        responseMeta={traceResponseMeta}
        onOpenSourceInDrawer={(sourceId) => {
          setIsTraceDrawerOpen(false);
          setSelectedSourceId(sourceId);
          setDrawerResponseMeta(traceResponseMeta);
          setIsDrawerOpen(true);
        }}
      />
    </div>
  );
};
