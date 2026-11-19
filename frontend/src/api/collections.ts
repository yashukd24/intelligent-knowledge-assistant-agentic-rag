import { apiClient } from './client';
import type {
  Collection,
  CollectionListResponse,
  CreateCollectionInput,
} from '../types/api';

export async function fetchCollections(): Promise<CollectionListResponse> {
  const response = await apiClient.get<CollectionListResponse>('/api/collections');
  return response.data;
}

export async function fetchCollection(id: string): Promise<Collection> {
  const response = await apiClient.get<Collection>(`/api/collections/${id}`);
  return response.data;
}

export async function createCollection(input: CreateCollectionInput): Promise<Collection> {
  const response = await apiClient.post<Collection>('/api/collections', input);
  return response.data;
}

export async function deleteCollection(id: string): Promise<{ message: string }> {
  const response = await apiClient.delete<{ message: string }>(`/api/collections/${id}`);
  return response.data;
}
