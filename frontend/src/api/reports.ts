import { apiClient } from './client';

export const reportsService = {
  getReport: async (id: string) => {
    const response = await apiClient.get(`/reports/${id}`);
    return response.data;
  }
};
