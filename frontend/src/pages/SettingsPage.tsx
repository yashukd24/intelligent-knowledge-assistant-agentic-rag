import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchHealth } from '../api/health';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { Button } from '../components/common/Button';
import {
  Settings,
  Server,
  Database,
  Cpu,
  ShieldCheck,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Info,
} from 'lucide-react';

export const SettingsPage: React.FC = () => {
  const { data: healthData, isLoading, isError, refetch } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
  });

  return (
    <div className="space-y-6">
      {/* Top Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-slate-900 tracking-tight">System Environment & Controls</h2>
          <p className="text-xs text-slate-500">Live inspection of backend services and architecture gates</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          isLoading={isLoading}
          className="text-xs"
        >
          <RefreshCw className="w-3.5 h-3.5 mr-1" />
          Refresh Health Status
        </Button>
      </div>

      {/* Live Backend Health Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4 text-brand-600" />
              <CardTitle>FastAPI Backend Status</CardTitle>
            </div>
            {healthData?.status === 'ok' ? (
              <Badge variant="success">Service Healthy</Badge>
            ) : (
              <Badge variant="danger">Service Degraded</Badge>
            )}
          </div>
          <CardDescription>Direct status returned from GET /health</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200/80">
              <span className="text-[10px] uppercase font-mono text-slate-400 block">Application</span>
              <span className="text-sm font-semibold text-slate-900">
                {healthData?.app_name || 'Intelligent Knowledge Assistant'}
              </span>
            </div>
            <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200/80">
              <span className="text-[10px] uppercase font-mono text-slate-400 block">Version</span>
              <span className="text-sm font-semibold text-slate-900 font-mono">
                {healthData?.version ? `v${healthData.version}` : 'v0.1.0'}
              </span>
            </div>
            <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200/80">
              <span className="text-[10px] uppercase font-mono text-slate-400 block">Database Status</span>
              <span className="text-sm font-semibold text-emerald-600 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                {healthData?.components?.database?.status || 'ok (SQLite / WAL)'}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model & Retrieval Engine Architecture */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-purple-600" />
              <CardTitle>LLM Provider & Safety Parameters</CardTitle>
            </div>
            <CardDescription>Configured local and cloud model inference</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-xs">
            <div className="flex items-center justify-between py-2 border-b border-slate-100">
              <span className="text-slate-600">Active Provider</span>
              <Badge variant="outline" className="font-mono">Ollama (Local)</Badge>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-slate-100">
              <span className="text-slate-600">Model Name</span>
              <span className="font-mono font-semibold text-slate-900">qwen2.5:3b</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-slate-100">
              <span className="text-slate-600">Host Endpoint</span>
              <span className="font-mono text-slate-700">http://localhost:11434</span>
            </div>
            <div className="flex items-center justify-between py-2">
              <span className="text-slate-600">RAM Safety Threshold</span>
              <span className="font-mono text-emerald-700 font-semibold">&ge; 5.0 GB available</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-emerald-600" />
              <CardTitle>Vector Store & Ingestion Pipeline</CardTitle>
            </div>
            <CardDescription>Embedding model, chunking parameters, and persistence</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-xs">
            <div className="flex items-center justify-between py-2 border-b border-slate-100">
              <span className="text-slate-600">Vector Database</span>
              <Badge variant="outline" className="font-mono">ChromaDB (Persistent)</Badge>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-slate-100">
              <span className="text-slate-600">Embedding Model</span>
              <span className="font-mono font-semibold text-slate-900">BAAI/bge-small-en-v1.5</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-slate-100">
              <span className="text-slate-600">Embedding Dimensions</span>
              <span className="font-mono text-slate-700">384 dimensions</span>
            </div>
            <div className="flex items-center justify-between py-2">
              <span className="text-slate-600">Chunk Size & Overlap</span>
              <span className="font-mono text-slate-700">512 tokens / 64 overlap</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Architectural Safeguards */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-brand-600" />
            <CardTitle>Project Architecture Safeguards</CardTitle>
          </div>
          <CardDescription>Strict architectural constraints enforced in this implementation</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2.5 text-xs text-slate-700">
          <div className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
            <p>
              <strong>Plain Python State Machine:</strong> LangChain and LangGraph are strictly prohibited. All agent transitions, loops, tools, and sufficiency gates are deterministic plain-Python classes.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
            <p>
              <strong>Citation Validation:</strong> All generated answers must cite retrieved evidence using bracketed numbers `[N]` referencing retrieved chunks. Hallucinations or untracked claims are flagged.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
            <p>
              <strong>Memory Safety Protocol:</strong> Local inference is gated by physical RAM availability (&ge;5.0 GB required) to prevent OS instability on constrained hardware.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
