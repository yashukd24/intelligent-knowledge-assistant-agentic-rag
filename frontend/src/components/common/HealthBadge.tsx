import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchHealth } from '../../api/health';
import { Badge } from './Badge';
import { Activity, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';

export const HealthBadge: React.FC = () => {
  const { data, isError, isLoading } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 15000, // Poll health every 15s
    retry: 1,
  });

  if (isLoading) {
    return (
      <Badge variant="outline" className="text-slate-500 animate-pulse text-xs shrink-0">
        <Activity className="w-3 h-3 animate-spin" />
        <span className="hidden sm:inline">Connecting...</span>
      </Badge>
    );
  }

  if (isError || !data || data.status !== 'ok') {
    return (
      <Badge variant="danger" className="text-xs flex items-center gap-1.5 shrink-0" title="Backend unreachable on port 8000">
        <XCircle className="w-3.5 h-3.5 text-rose-500" />
        <span className="font-semibold">
          <span className="hidden sm:inline">Backend </span>Offline
        </span>
      </Badge>
    );
  }

  const dbStatus = data.components?.database?.status;
  const isHealthy = data.status === 'ok' && (!dbStatus || dbStatus === 'ok');

  return (
    <Badge
      variant={isHealthy ? 'success' : 'warning'}
      className="text-xs flex items-center gap-1.5 cursor-pointer hover:bg-emerald-100/60 transition-colors shrink-0"
      title={`FastAPI ${data.version} | DB: ${dbStatus || 'ok'} | Engine: OK`}
    >
      {isHealthy ? (
        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
      ) : (
        <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
      )}
      <span className="font-medium whitespace-nowrap">
        <span className="hidden sm:inline">Backend </span>{isHealthy ? 'Online' : 'Degraded'}
      </span>
      <span className="text-[10px] text-slate-400 font-mono hidden sm:inline">
        v{data.version}
      </span>
    </Badge>
  );
};
