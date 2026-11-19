import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchCollections, createCollection, deleteCollection } from '../api/collections';
import { Card, CardContent } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { Badge } from '../components/common/Badge';
import {
  FolderArchive,
  Plus,
  Trash2,
  FileText,
  Calendar,
  Search,
  AlertCircle,
  ExternalLink,
  Loader2,
  X,
  AlertTriangle,
} from 'lucide-react';
import type { ApiError } from '../api/client';
import type { Collection } from '../types/api';

export const CollectionsPage: React.FC = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [searchQuery, setSearchQuery] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Deletion modal state
  const [deletingCollection, setDeletingCollection] = useState<Collection | null>(null);

  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['collections'],
    queryFn: fetchCollections,
  });

  const createMutation = useMutation({
    mutationFn: createCollection,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['collections'] });
      setIsCreating(false);
      setName('');
      setDescription('');
      setErrorMessage(null);
    },
    onError: (err: ApiError) => {
      setErrorMessage(err.message);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteCollection(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['collections'] });
      setDeletingCollection(null);
    },
    onError: (err: ApiError) => {
      alert(`Failed to delete collection: ${err.message}`);
      setDeletingCollection(null);
    },
  });

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedName = name.trim();
    if (!trimmedName) {
      setErrorMessage('Collection name cannot be empty or only whitespace.');
      return;
    }
    setErrorMessage(null);
    createMutation.mutate({
      name: trimmedName,
      description: description.trim() || undefined,
    });
  };

  const filteredCollections = (data?.collections || []).filter((col) =>
    col.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (col.description && col.description.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="space-y-6">
      {/* Top Action Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search collections..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-white border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500 shadow-2xs"
          />
        </div>

        <Button
          variant="primary"
          onClick={() => {
            setIsCreating(true);
            setErrorMessage(null);
          }}
          disabled={isCreating}
          className="shrink-0"
        >
          <Plus className="w-4 h-4 mr-1.5" />
          Create Collection
        </Button>
      </div>

      {/* Creation Modal / Inline Drawer */}
      {isCreating && (
        <Card className="border-brand-200 bg-brand-50/20 shadow-sm animate-in fade-in duration-200">
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-slate-900">Create New Knowledge Collection</h3>
              <button
                type="button"
                onClick={() => setIsCreating(false)}
                className="text-slate-400 hover:text-slate-600"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {errorMessage && (
              <div className="mb-3.5 p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
                  <span>{errorMessage}</span>
                </div>
                <button
                  type="button"
                  onClick={() => setErrorMessage(null)}
                  className="text-rose-600 hover:text-rose-800 font-semibold ml-2"
                >
                  Dismiss
                </button>
              </div>
            )}

            <form onSubmit={handleCreate} className="space-y-3.5">
              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Collection Name <span className="text-rose-500">*</span>
                </label>
                <input
                  type="text"
                  required
                  maxLength={255}
                  placeholder="e.g., Financial Reports 2025"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={createMutation.isPending}
                  className="w-full px-3 py-2 bg-white border border-slate-200 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 disabled:bg-slate-100"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Description <span className="text-slate-400 font-normal">(Optional)</span>
                </label>
                <input
                  type="text"
                  placeholder="Summary of document scope and knowledge domain"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  disabled={createMutation.isPending}
                  className="w-full px-3 py-2 bg-white border border-slate-200 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 disabled:bg-slate-100"
                />
              </div>
              <div className="flex items-center justify-end gap-2 pt-1">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  disabled={createMutation.isPending}
                  onClick={() => setIsCreating(false)}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  variant="primary"
                  size="sm"
                  isLoading={createMutation.isPending}
                  disabled={!name.trim() || createMutation.isPending}
                >
                  Create Collection
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      )}

      {/* Loading & Error States */}
      {isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-5 space-y-3">
                <div className="h-5 bg-slate-200 rounded w-1/2" />
                <div className="h-4 bg-slate-100 rounded w-3/4" />
                <div className="h-4 bg-slate-100 rounded w-1/3" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {isError && (
        <div className="p-6 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold">Unable to load collections</h4>
              <p className="text-xs text-rose-700 mt-1">
                {(error as Error)?.message || 'Make sure the FastAPI backend is running on port 8000.'}
              </p>
            </div>
          </div>
          <Button variant="outline" size="sm" onClick={() => refetch()} className="text-xs shrink-0">
            Retry
          </Button>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && !isError && filteredCollections.length === 0 && (
        <Card className="border-dashed p-10 text-center">
          <div className="w-12 h-12 rounded-full bg-slate-100 text-slate-400 mx-auto flex items-center justify-center mb-3">
            <FolderArchive className="w-6 h-6" />
          </div>
          <h3 className="text-sm font-semibold text-slate-800">No knowledge collections found</h3>
          <p className="text-xs text-slate-500 max-w-sm mx-auto mt-1 mb-4">
            {searchQuery
              ? 'No collections match your search filter.'
              : 'Create your first knowledge collection to upload PDFs, DOCX, and TXT documents for semantic indexing.'}
          </p>
          <Button
            variant="primary"
            size="sm"
            onClick={() => {
              setIsCreating(true);
              setErrorMessage(null);
            }}
          >
            <Plus className="w-4 h-4 mr-1.5" />
            Create First Collection
          </Button>
        </Card>
      )}

      {/* Collections Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredCollections.map((col) => (
          <Card
            key={col.id}
            hoverable
            className="flex flex-col justify-between cursor-pointer group"
            onClick={() => navigate(`/collections/${col.id}`)}
          >
            <CardContent className="p-5 space-y-3">
              <div className="flex items-start justify-between gap-2">
                <div className="w-9 h-9 rounded-lg bg-brand-50 text-brand-600 flex items-center justify-center shrink-0">
                  <FolderArchive className="w-4 h-4" />
                </div>
                <div className="flex items-center gap-1.5">
                  <Badge variant="primary" size="sm">
                    <FileText className="w-3 h-3 mr-1" />
                    <span>{col.document_count} {col.document_count === 1 ? 'doc' : 'docs'}</span>
                  </Badge>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setDeletingCollection(col);
                    }}
                    title="Delete collection"
                    className="p-1.5 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-md transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-semibold text-slate-900 group-hover:text-brand-600 transition-colors flex items-center gap-1">
                  <span>{col.name}</span>
                  <ExternalLink className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                </h3>
                <p className="text-xs text-slate-500 line-clamp-2 mt-1 min-h-[2rem]">
                  {col.description || 'No description provided.'}
                </p>
              </div>

              <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
                <span className="flex items-center gap-1">
                  <Calendar className="w-3 h-3" />
                  <span>{new Date(col.created_at).toLocaleDateString()}</span>
                </span>
                <span className="font-mono text-[10px] text-slate-400 truncate max-w-[120px]">
                  {col.id.slice(0, 8)}...
                </span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Confirmation Modal for Collection Deletion */}
      {deletingCollection && (
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
                  Delete Collection?
                </h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Are you sure you want to delete <strong className="text-slate-900">"{deletingCollection.name}"</strong>?
                  {deletingCollection.document_count > 0 && (
                    <span className="text-rose-600 block mt-1 font-medium">
                      Warning: All {deletingCollection.document_count} documents, disk files, and ChromaDB vector embeddings will be permanently removed.
                    </span>
                  )}
                </p>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2.5 pt-2">
              <Button
                variant="outline"
                size="sm"
                disabled={deleteMutation.isPending}
                onClick={() => setDeletingCollection(null)}
              >
                Cancel
              </Button>
              <Button
                variant="danger"
                size="sm"
                isLoading={deleteMutation.isPending}
                onClick={() => deleteMutation.mutate(deletingCollection.id)}
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
