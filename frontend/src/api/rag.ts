import { apiClient } from './client';
import type {
  ExecutionTrace,
  RAGAnswerRequest,
  RAGAnswerResponse,
  SourceReference,
  TraceStage,
} from '../types/api';

/**
 * Submit query to conventional RAG answer endpoint.
 * POST /api/collections/{collection_id}/rag/answer
 */
export async function submitRAGQuery(
  collectionId: string,
  request: RAGAnswerRequest
): Promise<RAGAnswerResponse> {
  const response = await apiClient.post<RAGAnswerResponse>(
    `/api/collections/${collectionId}/rag/answer`,
    request
  );
  return response.data;
}

export type MockScenarioType =
  | 'grounded'
  | 'empty_collection'
  | 'no_docs'
  | 'insufficient_evidence'
  | 'invalid_citation'
  | 'provider_error'
  | 'timeout'
  | 'partial_failure';

function buildMockTrace(
  scenario: MockScenarioType,
  query: string,
  collectionId: string,
  sources: SourceReference[],
  validCitations: string[],
  invalidCitations: string[],
  uncitedSources: string[],
  totalLatencyMs: number
): ExecutionTrace {
  const now = Date.now();
  const stages: TraceStage[] = [];

  const timeIso = (offsetMs: number) => new Date(now - totalLatencyMs + offsetMs).toISOString();

  // 1. User Query Received
  stages.push({
    id: 'stage-1-query',
    name: 'User Query Received',
    status: 'completed',
    started_at: timeIso(0),
    completed_at: timeIso(1),
    duration_ms: 1.2,
    details: {
      query_length: query.length,
      query_preview: query.slice(0, 100),
    },
  });

  // 2. Query Validation
  stages.push({
    id: 'stage-2-validation',
    name: 'Query Validation',
    status: 'completed',
    started_at: timeIso(1),
    completed_at: timeIso(4),
    duration_ms: 2.8,
    details: {
      top_k: 5,
      max_context_chars: 12000,
      similarity_threshold: 0.35,
      status: 'valid',
    },
  });

  // 3. Collection Selection
  stages.push({
    id: 'stage-3-collection',
    name: 'Collection Selection',
    status: 'completed',
    started_at: timeIso(4),
    completed_at: timeIso(12),
    duration_ms: 8.1,
    details: {
      collection_id: collectionId,
      ownership_check: 'verified_active',
      sqlite_status: 'indexed',
    },
  });

  // Handle partial failure test scenario
  if (scenario === 'partial_failure') {
    stages.push({
      id: 'stage-4-retrieval',
      name: 'Document Retrieval',
      status: 'completed',
      started_at: timeIso(12),
      completed_at: timeIso(45),
      duration_ms: 32.5,
      details: {
        raw_candidates_count: 5,
        vector_store: 'ChromaDB',
      },
    });
    stages.push({
      id: 'stage-5-filtering',
      name: 'Similarity Filtering',
      status: 'failed',
      started_at: timeIso(45),
      completed_at: timeIso(50),
      duration_ms: 5.0,
      error_message: 'Cosine similarity calculation encountered invalid embedding vector format.',
    });
    stages.push({
      id: 'stage-6-context',
      name: 'Context Construction',
      status: 'skipped',
      details: { reason: 'Prior similarity filtering stage failed.' },
    });
    stages.push({
      id: 'stage-7-generation',
      name: 'LLM Answer Generation',
      status: 'skipped',
      details: { reason: 'No context available due to upstream filtering failure.' },
    });
    stages.push({
      id: 'stage-8-citation',
      name: 'Citation Validation',
      status: 'skipped',
      details: { reason: 'Generation skipped.' },
    });
    stages.push({
      id: 'stage-9-final',
      name: 'Final Answer Returned',
      status: 'failed',
      started_at: timeIso(50),
      completed_at: timeIso(52),
      duration_ms: 2.1,
      error_message: 'Pipeline aborted due to similarity filtering failure.',
    });

    return {
      trace_id: `trace-${Date.now()}`,
      stages,
      total_duration_ms: 52.0,
    };
  }

  // 4. Document Retrieval
  const isZeroCandidates = scenario === 'empty_collection' || scenario === 'no_docs';
  stages.push({
    id: 'stage-4-retrieval',
    name: 'Document Retrieval',
    status: 'completed',
    started_at: timeIso(12),
    completed_at: timeIso(48),
    duration_ms: null,
    details: {
      retrieved_chunk_count: isZeroCandidates ? 0 : 5,
      candidate_count: isZeroCandidates ? 0 : 5,
      vector_db: 'ChromaDB (cosine similarity)',
      dense_model: 'BAAI/bge-small-en-v1.5',
      timing_note: 'Sub-stage duration unmeasured (included in total pipeline latency)',
    },
  });

  // 5. Similarity Filtering
  const isExcluded = scenario === 'insufficient_evidence' || isZeroCandidates;
  stages.push({
    id: 'stage-5-filtering',
    name: 'Similarity Filtering',
    status: 'completed',
    started_at: timeIso(48),
    completed_at: timeIso(60),
    duration_ms: null,
    details: {
      similarity_threshold: 0.35,
      passing_sources_count: isExcluded ? 0 : sources.length,
      excluded_chunks_count: isExcluded ? 3 : 2,
      max_similarity_score: isExcluded ? 0.281 : 0.8842,
      timing_note: 'Sub-stage duration unmeasured (evaluated inside SQLite validation gate)',
    },
  });

  // 6. Context Construction
  const hasEvidence = !isExcluded && sources.length > 0;
  stages.push({
    id: 'stage-6-context',
    name: 'Context Construction',
    status: 'completed',
    started_at: timeIso(60),
    completed_at: timeIso(70),
    duration_ms: null,
    details: {
      character_budget: 12000,
      assembled_context_chars: hasEvidence ? 837 : 0,
      sources_included: hasEvidence ? sources.map((s) => s.source_id) : [],
      has_evidence: hasEvidence,
      timing_note: 'Sub-stage duration unmeasured (deterministic context formatting)',
    },
  });

  // 7. LLM Answer Generation
  if (!hasEvidence) {
    stages.push({
      id: 'stage-7-generation',
      name: 'LLM Answer Generation',
      status: 'skipped',
      started_at: timeIso(70),
      completed_at: timeIso(71),
      duration_ms: 0.5,
      details: {
        reason: 'Bypassed LLM inference because zero relevant evidence passed the similarity threshold.',
        rule_based_refusal: true,
      },
    });
  } else if (scenario === 'provider_error') {
    stages.push({
      id: 'stage-7-generation',
      name: 'LLM Answer Generation',
      status: 'failed',
      started_at: timeIso(70),
      completed_at: timeIso(220),
      duration_ms: 150.0,
      details: {
        provider: 'ollama',
        model: 'qwen2.5:3b',
        attempt: 1,
      },
      error_message: 'Upstream LLM Provider Error (HTTP 502): Model service temporarily unreachable.',
    });
  } else if (scenario === 'timeout') {
    stages.push({
      id: 'stage-7-generation',
      name: 'LLM Answer Generation',
      status: 'failed',
      started_at: timeIso(70),
      completed_at: timeIso(370),
      duration_ms: 300.0,
      details: {
        provider: 'ollama',
        model: 'qwen2.5:3b',
        timeout_seconds: 120,
      },
      error_message: 'LLM generation request timed out after 120 seconds.',
    });
  } else {
    stages.push({
      id: 'stage-7-generation',
      name: 'LLM Answer Generation',
      status: 'completed',
      started_at: timeIso(70),
      completed_at: timeIso(390),
      duration_ms: 320.0,
      details: {
        provider: 'ollama (mocked for RAM safety)',
        model: 'qwen2.5:3b',
        prompt_tokens: 412,
        completion_tokens: 58,
        total_tokens: 470,
        temperature: 0.1,
      },
    });
  }

  // 8. Citation Validation
  if (!hasEvidence || scenario === 'provider_error' || scenario === 'timeout') {
    stages.push({
      id: 'stage-8-citation',
      name: 'Citation Validation',
      status: 'skipped',
      started_at: timeIso(390),
      completed_at: timeIso(391),
      duration_ms: 0.0,
      details: {
        reason: 'Citation validation skipped because answer generation was bypassed or failed.',
      },
    });
  } else {
    stages.push({
      id: 'stage-8-citation',
      name: 'Citation Validation',
      status: 'completed',
      started_at: timeIso(390),
      completed_at: timeIso(395),
      duration_ms: 5.2,
      details: {
        valid_citations: validCitations,
        invalid_citations: invalidCitations,
        uncited_sources: uncitedSources,
        citation_count: validCitations.length + invalidCitations.length,
        integrity_status: invalidCitations.length > 0 ? 'warning' : 'verified',
      },
    });
  }

  // 9. Final Answer Returned
  const isFinalSuccess = scenario !== 'provider_error' && scenario !== 'timeout';
  stages.push({
    id: 'stage-9-final',
    name: 'Final Answer Returned',
    status: isFinalSuccess ? 'completed' : 'failed',
    started_at: timeIso(395),
    completed_at: timeIso(totalLatencyMs),
    duration_ms: 4.8,
    details: {
      status: scenario === 'insufficient_evidence' || isZeroCandidates
        ? 'insufficient_evidence'
        : scenario === 'provider_error'
        ? 'provider_error'
        : scenario === 'timeout'
        ? 'timeout'
        : 'answered',
      total_latency_ms: totalLatencyMs,
      sources_count: sources.length,
    },
    error_message: isFinalSuccess ? null : 'Generation failed during pipeline execution.',
  });

  return {
    trace_id: `trace-${Date.now()}`,
    stages,
    total_duration_ms: totalLatencyMs,
  };
}

/**
 * Safe mocked RAG responses for component testing and demonstration
 * without requiring live Ollama inference or violating the 5 GB RAM safety threshold.
 */
export async function mockRAGQuery(
  scenario: MockScenarioType,
  collectionId: string,
  query: string
): Promise<RAGAnswerResponse> {
  // Simulate natural pipeline execution delay (350ms)
  await new Promise((r) => setTimeout(r, 350));

  const sampleSources: SourceReference[] = [
    {
      source_id: '[S1]',
      document_id: 'doc-college-0001',
      chunk_id: 'doc-college-0001_v1_c0000',
      filename: '01_campus_directory.txt',
      page_number: 1,
      chunk_index: 0,
      similarity_score: 0.8924,
      char_count: 320,
      chunk_text:
        'The Department of Computer Science & Engineering (CSE) HOD Cabin is located in Room 302 on the 3rd Floor of the Main Administrative Block (Block A). Faculty cabins are located in Rooms 303 to 318.',
    },
    {
      source_id: '[S2]',
      document_id: 'doc-college-0002',
      chunk_id: 'doc-college-0002_v1_c0000',
      filename: '02_placement_policy.txt',
      page_number: 1,
      chunk_index: 0,
      similarity_score: 0.8245,
      char_count: 295,
      chunk_text:
        'Placement Eligibility Criteria (Academic Year 2025-2026): Minimum 65.0% cumulative aggregate up to the 6th semester with 0 active backlogs, and minimum 80% attendance in pre-placement training modules.',
    },
    {
      source_id: '[S3]',
      document_id: 'doc-college-0003',
      chunk_id: 'doc-college-0003_v1_c0000',
      filename: '03_internship_guidelines.txt',
      page_number: 1,
      chunk_index: 0,
      similarity_score: 0.7412,
      char_count: 260,
      chunk_text:
        'Mandatory 8-week summer internship for 6th-semester students between May and July. Requires formal No Objection Certificate (NOC) endorsed by the Department HOD and verified training logbook.',
    },
  ];

  if (scenario === 'empty_collection' || scenario === 'no_docs' || scenario === 'insufficient_evidence') {
    const trace = buildMockTrace(scenario, query, collectionId, [], [], [], [], 42.5);
    return {
      query,
      answer:
        "I couldn't find this information in the uploaded college documents. Please upload the relevant document or contact the college administration.",
      status: 'insufficient_evidence',
      collection_id: collectionId,
      sources: [],
      valid_citations: [],
      invalid_citations: [],
      uncited_sources: [],
      provider: 'rule_based',
      model: 'none',
      usage: null,
      generation_latency_ms: 0.0,
      total_latency_ms: 42.5,
      retrieval_metadata: {
        collection_empty: scenario === 'empty_collection',
        threshold_filtered: scenario === 'insufficient_evidence',
        candidate_count: scenario === 'no_docs' ? 0 : 3,
      },
      limitations_note: 'Citation validation verifies reference existence, not semantic entailment.',
      execution_trace: trace,
    };
  }

  if (scenario === 'provider_error') {
    const trace = buildMockTrace(scenario, query, collectionId, sampleSources, [], [], ['[S1]', '[S2]', '[S3]'], 235.0);
    return {
      query,
      answer: 'Failed to generate answer due to an upstream language model error: Model service temporarily unreachable.',
      status: 'provider_error',
      collection_id: collectionId,
      sources: sampleSources,
      valid_citations: [],
      invalid_citations: [],
      uncited_sources: ['[S1]', '[S2]', '[S3]'],
      provider: 'ollama',
      model: 'qwen2.5:3b',
      usage: null,
      generation_latency_ms: 150.0,
      total_latency_ms: 235.0,
      retrieval_metadata: { top_k: 5, candidate_count: 3 },
      error_message: 'Upstream LLM Provider Error (HTTP 502): Model service temporarily unreachable.',
      limitations_note: 'Citation validation verifies reference existence, not semantic entailment.',
      execution_trace: trace,
    };
  }

  if (scenario === 'timeout') {
    const trace = buildMockTrace(scenario, query, collectionId, sampleSources, [], [], ['[S1]', '[S2]', '[S3]'], 380.0);
    return {
      query,
      answer: 'The language model request timed out after 120 seconds. Please try again.',
      status: 'timeout',
      collection_id: collectionId,
      sources: sampleSources,
      valid_citations: [],
      invalid_citations: [],
      uncited_sources: ['[S1]', '[S2]', '[S3]'],
      provider: 'ollama',
      model: 'qwen2.5:3b',
      usage: null,
      generation_latency_ms: 300.0,
      total_latency_ms: 380.0,
      retrieval_metadata: { top_k: 5, candidate_count: 3 },
      error_message: 'LLM generation request timed out after 120 seconds.',
      limitations_note: 'Citation validation verifies reference existence, not semantic entailment.',
      execution_trace: trace,
    };
  }

  if (scenario === 'partial_failure') {
    const trace = buildMockTrace(scenario, query, collectionId, [], [], [], [], 52.0);
    return {
      query,
      answer: 'Pipeline execution halted at stage 5: Similarity filtering failed due to malformed embedding vector.',
      status: 'provider_error',
      collection_id: collectionId,
      sources: [],
      valid_citations: [],
      invalid_citations: [],
      uncited_sources: [],
      provider: 'none',
      model: 'none',
      usage: null,
      generation_latency_ms: 0.0,
      total_latency_ms: 52.0,
      retrieval_metadata: { top_k: 5, candidate_count: 5 },
      error_message: 'Cosine similarity calculation encountered invalid embedding vector format.',
      limitations_note: 'Citation validation verifies reference existence, not semantic entailment.',
      execution_trace: trace,
    };
  }

  if (scenario === 'invalid_citation') {
    const trace = buildMockTrace(
      scenario,
      query,
      collectionId,
      sampleSources,
      ['[S1]'],
      ['[S9]'],
      ['[S2]', '[S3]'],
      412.0
    );
    return {
      query,
      answer:
        'According to the campus directory [S1], the CSE HOD cabin is located in Room 302 on the 3rd Floor of Administrative Block A. Placement exemptions may be approved by the university chancellor [S9].',
      status: 'answered',
      collection_id: collectionId,
      sources: sampleSources,
      valid_citations: ['[S1]'],
      invalid_citations: ['[S9]'],
      uncited_sources: ['[S2]', '[S3]'],
      provider: 'ollama (mocked for RAM safety)',
      model: 'qwen2.5:3b',
      usage: { prompt_tokens: 420, completion_tokens: 48, total_tokens: 468 },
      generation_latency_ms: 380.0,
      total_latency_ms: 412.0,
      retrieval_metadata: { top_k: 5, candidate_count: 3 },
      limitations_note: 'Citation validation verifies reference existence, not semantic entailment.',
      execution_trace: trace,
    };
  }

  // Standard grounded scenario
  const trace = buildMockTrace(
    'grounded',
    query,
    collectionId,
    sampleSources,
    ['[S1]', '[S2]'],
    [],
    ['[S3]'],
    425.0
  );
  return {
    query,
    answer:
      'The Department of Computer Science & Engineering (CSE) HOD Cabin is located in Room 302 on the 3rd Floor of the Main Administrative Block (Block A) [S1]. For campus placements [S2], students must maintain a minimum 65.0% cumulative aggregate score with zero active backlogs and 80% training attendance.',
    status: 'answered',
    collection_id: collectionId,
    sources: sampleSources,
    valid_citations: ['[S1]', '[S2]'],
    invalid_citations: [],
    uncited_sources: ['[S3]'],
    provider: 'ollama (mocked for RAM safety)',
    model: 'qwen2.5:3b',
    usage: { prompt_tokens: 412, completion_tokens: 58, total_tokens: 470 },
    generation_latency_ms: 395.0,
    total_latency_ms: 425.0,
    retrieval_metadata: { top_k: 5, candidate_count: 3 },
    limitations_note: 'Citation validation verifies reference existence, not semantic entailment.',
    execution_trace: trace,
  };
}

/**
 * Submit query to Agentic RAG state machine endpoint.
 * POST /api/collections/{collection_id}/agent/run
 */
export async function submitAgentQuery(
  collectionId: string,
  request: import('../types/api').AgentRunRequest
): Promise<import('../types/api').AgentRunResponse> {
  const response = await apiClient.post<import('../types/api').AgentRunResponse>(
    `/api/collections/${collectionId}/agent/run`,
    request
  );
  return response.data;
}

/**
 * Fetch latest evaluation metrics report.
 * GET /api/evaluation/metrics
 */
export async function fetchEvaluationMetrics(): Promise<Record<string, any>> {
  const response = await apiClient.get<Record<string, any>>('/api/evaluation/metrics');
  return response.data;
}

/**
 * Run comparative evaluation between Conventional and Agentic RAG.
 * POST /api/evaluation/compare
 */
export async function runComparativeEvaluation(): Promise<Record<string, any>> {
  const response = await apiClient.post<Record<string, any>>('/api/evaluation/compare');
  return response.data;
}
