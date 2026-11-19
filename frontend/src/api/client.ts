import axios, { AxiosError } from 'axios';

/**
 * Enterprise Axios REST client.
 * In development, requests without leading hosts route through Vite proxy.
 */
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 30000, // 30s timeout for embedding processing
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

export interface ApiError extends Error {
  status?: number;
  detail?: string;
}

// Response interceptor for consistent error extraction
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ detail?: string | { msg?: string }[] }>) => {
    let message = error.message;

    if (error.response?.data?.detail) {
      const detail = error.response.data.detail;
      if (typeof detail === 'string') {
        message = detail;
      } else if (Array.isArray(detail) && detail.length > 0 && detail[0].msg) {
        message = detail.map((d) => d.msg).join(', ');
      }
    } else if (error.code === 'ECONNABORTED') {
      message = 'Request timed out after 30 seconds.';
    } else if (!error.response) {
      message = 'Cannot connect to backend server. Ensure FastAPI is running on port 8000.';
    }

    const apiErr: ApiError = new Error(message);
    apiErr.status = error.response?.status;
    apiErr.detail = typeof error.response?.data?.detail === 'string' ? error.response.data.detail : message;

    return Promise.reject(apiErr);
  }
);
