import React, { useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchCollection } from '../api/collections';
import {
  fetchCollectionDocuments,
  uploadDocument,
  processDocument,
  deleteDocument,
} from '../api/documents';
import { Card, CardContent } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { Badge } from '../components/common/Badge';
import {
  ArrowLeft,
  Upload,
  FileText,
  Trash2,
  AlertCircle,
  CheckCircle2,
  Clock,
  XCircle,
  Layers,
  MessageSquare,
  Play,
  RotateCw,
  AlertTriangle,
  X,
  FileUp,
} from 'lucide-react';
import type { ApiError } from '../api/client';
import type { DocumentItem, DocumentStatus } from '../types/api';

const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50 MB limit enforced by backend
const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt'];

export const CollectionDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Replacement conflict state
  const [conflictFile, setConflictFile] = useState<File | null>(null);

  // Deletion modal state
  const [deletingDoc, setDeletingDoc] = useState<DocumentItem | null>(null);

  // Filters (Phase 22)
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');

  // Fetch collection metadata
  const { data: collection, isLoading: colLoading } = useQuery({
    queryKey: ['collection', id],
    queryFn: () => fetchCollection(id!),
    enabled: !!id,
  });

  // Fetch documents with auto-polling while processing
  const {
    data: docsData,
    isLoading: docsLoading,
    isError: docsError,
    error,
    refetch: refetchDocs,
  } = useQuery({
    queryKey: ['documents', id],
    queryFn: () => fetchCollectionDocuments(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      const hasActive = query.state.data?.documents.some(
        (d) => d.status === 'processing' || d.status === 'pending'
      );
      return hasActive ? 2000 : false;
    },
  });

  // Process mutation
  const processMutation = useMutation({
    mutationFn: (documentId: string) => processDocument(documentId, true),
    onSuccess: (processedDoc) => {
      setErrorMessage(null);
      setSuccessMessage(
        `Successfully indexed "${processedDoc.filename}" (${processedDoc.chunk_count} chunks embedded into ChromaDB).`
      );
      queryClient.invalidateQueries({ queryKey: ['documents', id] });
      queryClient.invalidateQueries({ queryKey: ['collection', id] });
      queryClient.invalidateQueries({ queryKey: ['collections'] });
    },
    onError: (err: ApiError) => {
      setErrorMessage(`Processing failed: ${err.message}`);
      queryClient.invalidateQueries({ queryKey: ['documents', id] });
    },
  });

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: async ({ file, replace }: { file: File; replace: boolean }) => {
      const uploadedDoc = await uploadDocument(id!, file, replace);
      // Automatically trigger processing and embedding immediately after upload
      try {
        await processMutation.mutateAsync(uploadedDoc.id);
      } catch {
        // processMutation's onError handles reporting
      }
      return uploadedDoc;
    },
    onSuccess: () => {
      setConflictFile(null);
      queryClient.invalidateQueries({ queryKey: ['documents', id] });
      queryClient.invalidateQueries({ queryKey: ['collection', id] });
      queryClient.invalidateQueries({ queryKey: ['collections'] });
    },
    onError: (err: ApiError, variables) => {
      if (err.status === 409 && err.message.toLowerCase().includes('already exists')) {
        // Offer replacement option to user
        setConflictFile(variables.file);
      } else {
        setErrorMessage(err.message);
      }
    },
  });

  // Delete mutation
  const deleteDocMutation = useMutation({
    mutationFn: (docId: string) => deleteDocument(docId),
    onSuccess: () => {
      setDeletingDoc(null);
      setSuccessMessage('Document and its ChromaDB embeddings were successfully deleted.');
      queryClient.invalidateQueries({ queryKey: ['documents', id] });
      queryClient.invalidateQueries({ queryKey: ['collection', id] });
      queryClient.invalidateQueries({ queryKey: ['collections'] });
    },
    onError: (err: ApiError) => {
      alert(`Failed to delete document: ${err.message}`);
      setDeletingDoc(null);
    },
  });

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset input
    e.target.value = '';
    setErrorMessage(null);
    setSuccessMessage(null);
    setConflictFile(null);

    // 1. Validate file extension
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      setErrorMessage(
        `Unsupported file type "${extension}". Please upload a .pdf, .docx, or .txt document.`
      );
      return;
    }

    // 2. Validate 50 MB limit
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setErrorMessage(
        `File size (${formatBytes(file.size)}) exceeds the maximum allowed limit of 50 MB.`
      );
      return;
    }

    if (file.size === 0) {
      setErrorMessage('Uploaded file is empty (0 bytes).');
      return;
    }

    // Submit upload
    uploadMutation.mutate({ file, replace: false });
  };

  const handleConfirmReplace = () => {
    if (!conflictFile) return;
    setErrorMessage(null);
    setConflictFile(null);
    uploadMutation.mutate({ file: conflictFile, replace: true });
  };

  const getStatusBadge = (status: DocumentStatus) => {
    switch (status) {
      case 'indexed':
        return (
          <Badge variant="success" size="sm">
            <CheckCircle2 className="w-3 h-3" />
            <span>Indexed</span>
          </Badge>
        );
      case 'processing':
        return (
          <Badge variant="primary" size="sm">
            <RotateCw className="w-3 h-3 animate-spin" />
            <span>Processing</span>
          </Badge>
        );
      case 'pending':
        return (
          <Badge variant="warning" size="sm">
            <Clock className="w-3 h-3" />
            <span>Pending</span>
          </Badge>
        );
      case 'failed':
        return (
          <Badge variant="danger" size="sm">
            <XCircle className="w-3 h-3" />
            <span>Failed</span>
          </Badge>
        );
      default:
        return (
          <Badge variant="default" size="sm">
            <span>{status}</span>
          </Badge>
        );
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  const isOperationBusy = uploadMutation.isPending || processMutation.isPending;

  return (
    <div className="space-y-6">
      {/* Back button & Collection title header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="icon"
            onClick={() => navigate('/collections')}
            title="Back to Collections"
            className="h-9 w-9 shrink-0"
          >
            <ArrowLeft className="w-4 h-4" />
          </Button>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-slate-900 tracking-tight">
                {colLoading ? 'Loading...' : collection?.name}
              </h2>
              <Badge variant="outline" className="font-mono text-[10px]">
                {id?.slice(0, 8)}
              </Badge>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              {collection?.description || 'No description provided.'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 self-stretch sm:self-auto">
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate('/chat')}
            className="text-xs"
          >
            <MessageSquare className="w-3.5 h-3.5 mr-1" />
            Query This Collection
          </Button>

          <Button
            variant="primary"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            isLoading={isOperationBusy}
            disabled={isOperationBusy}
            className="text-xs"
          >
            <Upload className="w-3.5 h-3.5 mr-1" />
            Upload Document
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>
      </div>

      {/* Success Notification Banner */}
      {successMessage && (
        <div className="p-3.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center justify-between animate-in fade-in">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>{successMessage}</span>
          </div>
          <button
            type="button"
            onClick={() => setSuccessMessage(null)}
            className="text-emerald-700 hover:text-emerald-900 font-semibold ml-2"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Error Notification Banner */}
      {errorMessage && (
        <div className="p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center justify-between animate-in fade-in">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
            <span>{errorMessage}</span>
          </div>
          <button
            type="button"
            onClick={() => setErrorMessage(null)}
            className="text-rose-600 hover:text-rose-800 font-semibold ml-2"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Replace Confirmation Prompt (Conflict Resolution) */}
      {conflictFile && (
        <Card className="border-amber-300 bg-amber-50/40 shadow-sm animate-in fade-in">
          <CardContent className="p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <div className="flex items-start gap-2.5">
              <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-xs font-semibold text-slate-900">
                  Duplicate Filename Detected
                </h4>
                <p className="text-xs text-slate-600 mt-0.5">
                  A document named <strong className="text-slate-900">"{conflictFile.name}"</strong> already exists in this collection.
                  Would you like to replace the existing file and re-embed its contents?
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0 self-end sm:self-auto">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setConflictFile(null)}
                className="text-xs"
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={handleConfirmReplace}
                isLoading={uploadMutation.isPending}
                className="text-xs bg-amber-600 hover:bg-amber-700 text-white"
              >
                Replace & Re-Index
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Document List Table */}
      <Card>
        <div className="p-5 border-b border-slate-100 flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Documents in Collection</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              PDF, DOCX, and TXT institutional documents parsed, chunked, and embedded into ChromaDB.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {/* Search Input */}
            <input
              type="text"
              placeholder="Search document name..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-brand-500 font-medium text-slate-800 w-44"
            />
            {/* Category Filter */}
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-brand-500 font-medium text-slate-800 max-w-[170px]"
            >
              <option value="all">All Categories</option>
              {Array.from(new Set((docsData?.documents || []).map((d) => d.category).filter(Boolean))).map((cat) => (
                <option key={cat!} value={cat!}>{cat}</option>
              ))}
            </select>
            <Button
              variant="outline"
              size="sm"
              onClick={() => refetchDocs()}
              className="text-xs h-7 px-2"
              title="Refresh document status"
            >
              <RotateCw className="w-3 h-3 mr-1" />
              Refresh
            </Button>
            <Badge variant="primary" size="sm">
              {docsData?.total ?? 0} {docsData?.total === 1 ? 'doc' : 'docs'}
            </Badge>
          </div>
        </div>

        <CardContent className="p-0">
          {docsLoading && (
            <div className="p-10 text-center text-xs text-slate-400">Loading documents...</div>
          )}

          {docsError && (
            <div className="p-6 text-xs text-rose-600">
              {(error as Error)?.message || 'Failed to fetch collection documents.'}
            </div>
          )}

          {!docsLoading && !docsError && (docsData?.documents.length === 0 || !docsData) && (
            <div className="p-12 text-center">
              <div className="w-12 h-12 rounded-full bg-slate-100 text-slate-400 mx-auto flex items-center justify-center mb-3">
                <FileUp className="w-6 h-6" />
              </div>
              <h4 className="text-sm font-semibold text-slate-800">No documents in this collection</h4>
              <p className="text-xs text-slate-500 max-w-sm mx-auto mt-1 mb-4">
                Upload a PDF, DOCX, or TXT document (up to 50 MB) to index and embed its text for semantic search.
              </p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
                isLoading={isOperationBusy}
                disabled={isOperationBusy}
              >
                <Upload className="w-4 h-4 mr-1.5" />
                Select File (.pdf, .docx, .txt)
              </Button>
            </div>
          )}

          {!docsLoading && docsData && docsData.documents.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 border-b border-slate-100 text-slate-600 uppercase tracking-wider text-[10px]">
                  <tr>
                    <th className="px-5 py-3 font-semibold">Document</th>
                    <th className="px-5 py-3 font-semibold">Format</th>
                    <th className="px-5 py-3 font-semibold">Size</th>
                    <th className="px-5 py-3 font-semibold">Chunks & Pages</th>
                    <th className="px-5 py-3 font-semibold">Status</th>
                    <th className="px-5 py-3 font-semibold text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {docsData.documents
                    .filter((doc) => {
                      if (categoryFilter !== 'all' && (doc.category || 'Uncategorized') !== categoryFilter) return false;
                      if (searchTerm && !doc.filename.toLowerCase().includes(searchTerm.toLowerCase())) return false;
                      return true;
                    })
                    .map((doc) => (
                    <tr key={doc.id} className="hover:bg-slate-50/50">
                      {/* Filename & SHA */}
                      <td className="px-5 py-3 font-medium text-slate-900">
                        <div className="flex items-center gap-2.5">
                          <FileText className="w-4 h-4 text-slate-400 shrink-0" />
                          <div className="min-w-0">
                            <span className="font-semibold block truncate max-w-xs sm:max-w-md">
                              {doc.filename}
                            </span>
                            <div className="flex flex-wrap items-center gap-1.5 text-[10px] text-slate-500 font-mono mt-0.5">
                              <span className="font-semibold text-slate-700">v{doc.version}</span>
                              {doc.category && (
                                <span className="bg-brand-50 text-brand-700 border border-brand-200/80 px-1.5 py-0.2 rounded font-sans font-medium">
                                  {doc.category}
                                </span>
                              )}
                              {doc.department && (
                                <span className="bg-slate-100 text-slate-700 border border-slate-200 px-1.5 py-0.2 rounded font-sans font-medium">
                                  {doc.department}
                                </span>
                              )}
                              {doc.academic_year && <span className="text-slate-400 font-sans">AY {doc.academic_year}</span>}
                              <span>•</span>
                              <span>sha256: {doc.sha256.slice(0, 8)}...</span>
                            </div>
                          </div>
                        </div>
                      </td>

                      {/* Format */}
                      <td className="px-5 py-3 uppercase font-mono text-[11px] text-slate-600">
                        <Badge variant="outline" size="sm" className="font-mono text-[10px]">
                          {doc.file_type}
                        </Badge>
                      </td>

                      {/* File Size */}
                      <td className="px-5 py-3 text-slate-600 font-mono">
                        {formatBytes(doc.size_bytes)}
                      </td>

                      {/* Chunks & Pages */}
                      <td className="px-5 py-3">
                        <div className="flex items-center gap-1.5 text-slate-700">
                          <Layers className="w-3.5 h-3.5 text-brand-600" />
                          <span className="font-medium">{doc.chunk_count} chunks</span>
                          {doc.page_count ? (
                            <span className="text-slate-400 text-[10px]">
                              ({doc.page_count} {doc.page_count === 1 ? 'page' : 'pages'})
                            </span>
                          ) : null}
                        </div>
                      </td>

                      {/* Status */}
                      <td className="px-5 py-3">
                        <div className="flex flex-col gap-1 items-start">
                          {getStatusBadge(doc.status)}
                          {doc.error && (
                            <span
                              className="text-[10px] text-rose-600 line-clamp-1 max-w-xs"
                              title={doc.error}
                            >
                              {doc.error}
                            </span>
                          )}
                        </div>
                      </td>

                      {/* Actions */}
                      <td className="px-5 py-3 text-right">
                        <div className="flex items-center justify-end gap-1">
                          {/* Manual Process Button if not indexed */}
                          {doc.status !== 'indexed' && doc.status !== 'processing' && (
                            <button
                              type="button"
                              onClick={() => processMutation.mutate(doc.id)}
                              disabled={processMutation.isPending}
                              title="Trigger text extraction and ChromaDB embedding"
                              className="p-1.5 text-brand-600 hover:text-brand-800 hover:bg-brand-50 rounded-md transition-colors"
                            >
                              <Play className="w-3.5 h-3.5" />
                            </button>
                          )}

                          {/* Delete Document Button */}
                          <button
                            type="button"
                            onClick={() => setDeletingDoc(doc)}
                            disabled={deleteDocMutation.isPending}
                            title="Delete document and vector embeddings"
                            className="p-1.5 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-md transition-colors"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Confirmation Modal for Document Deletion */}
      {deletingDoc && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs animate-in fade-in"
          role="dialog"
          aria-modal="true"
        >
          <div className="bg-white rounded-xl max-w-md w-full p-6 shadow-2xl border border-slate-200 space-y-4">
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 rounded-full bg-rose-100 text-rose-600 flex items-center justify-center shrink-0">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <div className="space-y-1">
                <h3 className="text-base font-semibold text-slate-900">
                  Delete Document?
                </h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Are you sure you want to delete <strong className="text-slate-900">"{deletingDoc.filename}"</strong>?
                  <span className="text-rose-600 block mt-1">
                    Its stored file on disk and all {deletingDoc.chunk_count} vector embeddings in ChromaDB will be permanently removed.
                  </span>
                </p>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2.5 pt-2">
              <Button
                variant="outline"
                size="sm"
                disabled={deleteDocMutation.isPending}
                onClick={() => setDeletingDoc(null)}
              >
                Cancel
              </Button>
              <Button
                variant="danger"
                size="sm"
                isLoading={deleteDocMutation.isPending}
                onClick={() => deleteDocMutation.mutate(deletingDoc.id)}
              >
                Confirm Delete
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
