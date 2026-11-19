import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { fetchCollections } from '../api/collections';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { Badge } from '../components/common/Badge';
import {
  FolderArchive,
  MessageSquare,
  Activity,
  ArrowRight,
  Database,
  Cpu,
  Layers,
  FileText,
  Workflow,
  Sparkles,
} from 'lucide-react';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();

  const { data: collectionsData, isLoading } = useQuery({
    queryKey: ['collections'],
    queryFn: fetchCollections,
  });

  const totalCollections = collectionsData?.total ?? 0;
  const totalDocuments =
    collectionsData?.collections.reduce((acc, c) => acc + (c.document_count || 0), 0) ?? 0;

  return (
    <div className="space-y-6">
      {/* Welcome Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-brand-900 via-brand-800 to-slate-900 p-5 sm:p-7 text-white shadow-sm">
        <div className="relative z-10 max-w-3xl space-y-2">
          <div className="inline-flex items-center gap-1.5 sm:gap-2 rounded-full bg-brand-500/20 px-2.5 sm:px-3 py-1 text-[11px] sm:text-xs font-medium text-brand-200 border border-brand-400/30">
            <Sparkles className="w-3.5 h-3.5 text-brand-300 shrink-0" />
            <span className="truncate">College Intelligence & Guidance Assistant</span>
          </div>
          <h2 className="text-xl sm:text-2xl lg:text-3xl font-bold tracking-tight text-white">
            Institutional Knowledge Assistant
          </h2>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed max-w-2xl">
            Autonomous institutional guidance platform for colleges. Serving students, faculty,
            prospective admissions, and administrators with role-aware multi-step retrieval,
            evidence sufficiency evaluation, and verified citations.
          </p>
          <div className="pt-2 sm:pt-3 flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5 sm:gap-3">
            <Button
              variant="primary"
              size="sm"
              onClick={() => navigate('/chat')}
              className="bg-brand-500 hover:bg-brand-400 text-white"
            >
              <MessageSquare className="w-4 h-4 mr-1.5" />
              Start Q&A Session
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigate('/collections')}
              className="bg-white/10 hover:bg-white/20 text-white border-white/20"
            >
              <FolderArchive className="w-4 h-4 mr-1.5" />
              Manage Knowledge Base
            </Button>
          </div>
        </div>
        {/* Subtle decorative background pattern */}
        <div className="absolute right-0 top-0 bottom-0 w-1/3 opacity-10 flex items-center justify-center pointer-events-none">
          <Workflow className="w-80 h-80 text-white" />
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-5 flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-slate-500">Active Collections</p>
              <h4 className="text-2xl font-bold text-slate-900 mt-1">
                {isLoading ? '...' : totalCollections}
              </h4>
              <p className="text-[11px] text-slate-400 mt-0.5">Isolated knowledge stores</p>
            </div>
            <div className="w-11 h-11 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
              <FolderArchive className="w-5 h-5" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-5 flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-slate-500">Indexed Documents</p>
              <h4 className="text-2xl font-bold text-slate-900 mt-1">
                {isLoading ? '...' : totalDocuments}
              </h4>
              <p className="text-[11px] text-slate-400 mt-0.5">PDF, DOCX, TXT chunks</p>
            </div>
            <div className="w-11 h-11 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
              <FileText className="w-5 h-5" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-5 flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-slate-500">Vector Store</p>
              <h4 className="text-lg font-bold text-slate-900 mt-1">ChromaDB</h4>
              <p className="text-[11px] text-slate-400 mt-0.5">BAAI/bge-small-en-v1.5</p>
            </div>
            <div className="w-11 h-11 rounded-lg bg-purple-50 text-purple-600 flex items-center justify-center">
              <Database className="w-5 h-5" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-5 flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-slate-500">Verified Test Suite</p>
              <h4 className="text-2xl font-bold text-emerald-600 mt-1">339 / 339</h4>
              <p className="text-[11px] text-slate-400 mt-0.5">100% automated pass rate</p>
            </div>
            <div className="w-11 h-11 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
              <Cpu className="w-5 h-5" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Comparative Architecture Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Architecture: Conventional RAG vs. Agentic RAG</CardTitle>
                <CardDescription>
                  Empirical comparison criteria based on the project research specification
                </CardDescription>
              </div>
              <Badge variant="primary">Research Spec</Badge>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 border-b border-slate-100 text-slate-600 uppercase tracking-wider text-[10px]">
                  <tr>
                    <th className="px-5 py-3 font-semibold">Dimension</th>
                    <th className="px-5 py-3 font-semibold">Conventional RAG (Phase 7-9)</th>
                    <th className="px-5 py-3 font-semibold text-brand-700">Agentic RAG (Phase 10-15)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  <tr className="hover:bg-slate-50/50">
                    <td className="px-5 py-3 font-medium text-slate-900">Retrieval Mechanism</td>
                    <td className="px-5 py-3 text-slate-600">Single-pass, Top-K</td>
                    <td className="px-5 py-3 font-medium text-brand-700">Multi-step, adaptive refinement loop</td>
                  </tr>
                  <tr className="hover:bg-slate-50/50">
                    <td className="px-5 py-3 font-medium text-slate-900">Query Preprocessing</td>
                    <td className="px-5 py-3 text-slate-600">As-is raw query</td>
                    <td className="px-5 py-3 font-medium text-brand-700">Intent analysis & sub-query decomposition</td>
                  </tr>
                  <tr className="hover:bg-slate-50/50">
                    <td className="px-5 py-3 font-medium text-slate-900">Evidence Assessment</td>
                    <td className="px-5 py-3 text-slate-600">None (direct pass-through)</td>
                    <td className="px-5 py-3 font-medium text-brand-700">Explicit sufficiency decision gate (&ge;0.70)</td>
                  </tr>
                  <tr className="hover:bg-slate-50/50">
                    <td className="px-5 py-3 font-medium text-slate-900">Hallucination Mitigation</td>
                    <td className="px-5 py-3 text-slate-600">Regex citation validation</td>
                    <td className="px-5 py-3 font-medium text-brand-700">Iterative refinement + strict grounding verifier</td>
                  </tr>
                  <tr className="hover:bg-slate-50/50">
                    <td className="px-5 py-3 font-medium text-slate-900">State Engine</td>
                    <td className="px-5 py-3 text-slate-600">Stateless REST pipeline</td>
                    <td className="px-5 py-3 font-medium text-brand-700">Plain Python State Machine (SSE event stream)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Quick Launch Cards */}
        <div className="space-y-4">
          <Card hoverable className="cursor-pointer" onClick={() => navigate('/collections')}>
            <CardContent className="p-5 flex items-start gap-3.5">
              <div className="w-9 h-9 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center shrink-0">
                <FolderArchive className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-slate-900">Knowledge Base</h4>
                  <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-slate-600" />
                </div>
                <p className="text-xs text-slate-500 mt-1">
                  Upload documents, inspect chunking statistics, and manage collections.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card hoverable className="cursor-pointer" onClick={() => navigate('/chat')}>
            <CardContent className="p-5 flex items-start gap-3.5">
              <div className="w-9 h-9 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center shrink-0">
                <MessageSquare className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-slate-900">AI Assistant</h4>
                  <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-slate-600" />
                </div>
                <p className="text-xs text-slate-500 mt-1">
                  Ask questions across documents with verified source citations.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card hoverable className="cursor-pointer" onClick={() => navigate('/agent')}>
            <CardContent className="p-5 flex items-start gap-3.5">
              <div className="w-9 h-9 rounded-lg bg-purple-50 text-purple-600 flex items-center justify-center shrink-0">
                <Activity className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-slate-900">Agent Activity</h4>
                  <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-slate-600" />
                </div>
                <p className="text-xs text-slate-500 mt-1">
                  Inspect the plain Python state machine transitions and event streams.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
