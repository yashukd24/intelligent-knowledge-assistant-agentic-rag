/**
 * TypeScript API contracts matching backend FastAPI / Pydantic schemas.
 */

export interface ComponentHealth {
  status: 'ok' | 'degraded' | 'down';
  message?: string;
  [key: string]: unknown;
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'down';
  app_name: string;
  version: string;
  timestamp: string;
  components: Record<string, ComponentHealth>;
}

export interface Collection {
  id: string;
  name: string;
  description: string | null;
  document_count: number;
  created_at: string;
  updated_at: string;
}

export interface CollectionListResponse {
  collections: Collection[];
  total: number;
}

export interface CreateCollectionInput {
  name: string;
  description?: string;
}

export type DocumentStatus = 'pending' | 'processing' | 'indexed' | 'failed' | 'deletion_pending';

export interface DocumentItem {
  id: string;
  collection_id: string;
  filename: string;
  file_type: string;
  sha256: string;
  size_bytes: number;
  page_count: number;
  chunk_count: number;
  status: DocumentStatus;
  version: number;
  category?: string | null;
  department?: string | null;
  academic_year?: string | null;
  effective_date?: string | null;
  expiry_date?: string | null;
  section_heading?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  documents: DocumentItem[];
  total: number;
}

export interface ChunkMetadata {
  document_id: string;
  collection_id: string;
  filename: string;
  file_type: string;
  chunk_index: number;
  page_number?: number | null;
  [key: string]: unknown;
}

export interface DocumentChunk {
  chunk_id: string;
  document_id: string;
  collection_id: string;
  chunk_index: number;
  chunk_text: string;
  char_count: number;
  token_count: number;
  page_number?: number | null;
  page_range?: string | null;
  metadata: ChunkMetadata;
  version?: number | null;
}

export interface ProcessedDocument {
  document_id: string;
  collection_id: string;
  filename: string;
  file_type: string;
  page_count: number;
  chunk_count: number;
  total_characters: number;
  chunks: DocumentChunk[];
  status: DocumentStatus;
  error?: string | null;
}

export interface SourceReference {
  source_id: string; // Deterministic identifier: [S1], [S2]
  document_id: string;
  chunk_id: string;
  filename?: string | null;
  page_number?: number | null;
  page_range?: string | null;
  chunk_index: number;
  similarity_score: number;
  char_count: number;
  chunk_text?: string;
  text?: string;
  snippet?: string;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export type RAGAnswerStatus =
  | 'answered'
  | 'insufficient_evidence'
  | 'provider_error'
  | 'timeout'
  | 'rate_limited';

export type TraceStageStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export interface TraceStage {
  id: string;
  name: string;
  status: TraceStageStatus;
  started_at?: string | null;
  completed_at?: string | null;
  duration_ms?: number | null;
  details?: Record<string, unknown>;
  error_message?: string | null;
}

export interface ExecutionTrace {
  trace_id: string;
  stages: TraceStage[];
  total_duration_ms: number;
}

export interface RAGAnswerResponse {
  query: string;
  answer: string;
  status: RAGAnswerStatus;
  collection_id: string;
  sources: SourceReference[];
  valid_citations: string[];
  invalid_citations: string[];
  uncited_sources: string[];
  provider: string;
  model: string;
  usage?: TokenUsage | null;
  generation_latency_ms: number;
  total_latency_ms?: number;
  retrieval_metadata?: Record<string, unknown>;
  error_message?: string | null;
  limitations_note?: string;
  execution_trace?: ExecutionTrace | null;
}

export interface RAGAnswerRequest {
  query: string;
  top_k?: number;
  document_ids?: string[] | null;
  similarity_threshold?: number | null;
  max_context_chars?: number;
  max_output_tokens?: number | null;
  temperature?: number | null;
  collection_id?: string | null;
}

export type AgentStatus =
  | 'INITIALIZING'
  | 'PLANNING'
  | 'RETRIEVING'
  | 'EVALUATING'
  | 'REFINING'
  | 'SYNTHESIZING'
  | 'COMPLETED'
  | 'FAILED';

export interface AgentEvent {
  event_id: string;
  event_type: string;
  timestamp: string;
  step: number;
  payload: Record<string, unknown>;
  message: string;
}

export interface ToolCall {
  tool_name: string;
  arguments: Record<string, unknown>;
  step_number?: number | null;
}

export interface ToolResult {
  call_id: string;
  tool_name: string;
  status: string;
  output?: unknown;
  error_message?: string | null;
  latency_ms: number;
}

export interface RefinedQuery {
  iteration: number;
  strategy: string;
  original_query: string;
  refined_query_text: string;
  targeted_subquery_ids: string[];
  adjusted_top_k?: number | null;
  adjusted_similarity_threshold?: number | null;
  rationale: string;
}

export interface EvidenceEvaluation {
  overall_score: number;
  is_sufficient: boolean;
  reason: string;
  explanation: string;
  coverage_ratio: number;
  total_chunks_evaluated: number;
  qualifying_chunk_count: number;
}

export interface SynthesisVerification {
  is_verified: boolean;
  grounding_score: number;
  total_sentences: number;
  grounded_sentences: number;
  valid_citations: string[];
  invalid_citations: string[];
  uncited_sources: string[];
  hallucination_warning: boolean;
  grounding_explanation: string;
}

export interface AgentRunResponse {
  session_id: string;
  trace_id: string;
  collection_id: string;
  query: string;
  user_role?: string;
  college_intent?: string | null;
  status: AgentStatus;
  answer: string;
  is_sufficient: boolean;
  evidence_score: number;
  evidence_evaluation?: EvidenceEvaluation | null;
  iteration_count: number;
  refinement_history?: RefinedQuery[];
  sources: SourceReference[];
  valid_citations: string[];
  invalid_citations: string[];
  decomposed_queries: string[];
  comparison_objective?: string | null;
  tool_calls: ToolCall[];
  tool_results: ToolResult[];
  events: AgentEvent[];
  execution_trace?: ExecutionTrace | null;
  total_latency_ms: number;
  error_details?: string | null;
  limitations_note?: string;
}

export interface AgentRunRequest {
  query: string;
  user_role?: string;
  category?: string | null;
  department?: string | null;
  top_k?: number;
  document_ids?: string[] | null;
  similarity_threshold?: number | null;
  max_context_chars?: number;
  max_output_tokens?: number | null;
  temperature?: number | null;
  collection_id?: string | null;
}
