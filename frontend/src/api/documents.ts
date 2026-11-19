import { apiClient } from './client';
import type {
  DocumentItem,
  DocumentListResponse,
  ProcessedDocument,
} from '../types/api';

/**
 * Fetch documents for a collection.
 */
export async function fetchCollectionDocuments(
  collectionId: string,
  skip = 0,
  limit = 100
): Promise<DocumentListResponse> {
  const response = await apiClient.get<DocumentListResponse>(
    `/api/collections/${collectionId}/documents`,
    {
      params: { skip, limit },
    }
  );
  return response.data;
}

/**
 * Retrieve metadata and status of a single document.
 */
export async function fetchDocument(documentId: string): Promise<DocumentItem> {
  const response = await apiClient.get<DocumentItem>(`/api/documents/${documentId}`);
  return response.data;
}

/**
 * Upload a document to a collection.
 * Supports replace=true to overwrite an existing document with the same filename.
 */
export async function uploadDocument(
  collectionId: string,
  file: File,
  replace = false
): Promise<DocumentItem> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<DocumentItem>(
    `/api/collections/${collectionId}/documents`,
    formData,
    {
      params: { replace },
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
}

/**
 * Trigger text extraction, cleaning, chunking, and embedding into ChromaDB.
 */
export async function processDocument(
  documentId: string,
  embed = true
): Promise<ProcessedDocument> {
  const response = await apiClient.post<ProcessedDocument>(
    `/api/documents/${documentId}/process`,
    {},
    {
      params: { embed },
    }
  );
  return response.data;
}

/**
 * Delete a document from SQLite, ChromaDB, and disk storage.
 */
export async function deleteDocument(
  documentId: string
): Promise<{ status: string; id: string; filename?: string; message?: string }> {
  const response = await apiClient.delete<{
    status: string;
    id: string;
    filename?: string;
    message?: string;
  }>(`/api/documents/${documentId}`);
  return response.data;
}
