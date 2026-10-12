import { apiClient } from './client';

export const submissionsService = {
  uploadMedia: async (file: File, modality: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('modality', modality);
    const response = await apiClient.post('/submissions/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  getHistory: async () => {
    const response = await apiClient.get('/submissions/');
    return response.data;
  },

  getSubmissionStatus: async (id: string) => {
    const response = await apiClient.get(`/submissions/${id}`);
    return response.data;
  },

  getSubmission: async (id: string) => {
    const response = await apiClient.get(`/submissions/${id}`);
    return response.data;
  },

  getReport: async (id: string) => {
    const response = await apiClient.get(`/submissions/${id}`);
    return response.data;
  },

  uploadText: async (text: string) => {
    const formData = new FormData();
    formData.append('text_content', text);
    formData.append('modality', 'text');
    const response = await apiClient.post('/submissions/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  deleteSubmission: async (id: string) => {
    const response = await apiClient.delete(`/submissions/${id}`);
    return response.data;
  }
};
