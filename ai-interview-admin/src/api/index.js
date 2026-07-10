import api from './request'

export const authApi = {
  login: (email, password) => api.post('/auth/login', { email, password }),
  me: () => api.get('/auth/me'),
  logout: (refresh_token) => api.post('/auth/logout', { refresh_token })
}

export const userApi = {
  list: (params) => api.get('/users', { params }),
  stats: () => api.get('/users/stats'),
  toggleActive: (id) => api.put(`/users/${id}/toggle-active`)
}

export const interviewApi = {
  list: (params) => api.get('/interviews', { params }),
  detail: (id) => api.get(`/interviews/${id}`),
  delete: (id) => api.delete(`/interviews/${id}`)
}

// ── 题库管理 ─────────────────────────────────────────────────────
export const questionBankApi = {
  list: (params) => api.get('/question-bank', { params }),
  detail: (id) => api.get(`/question-bank/${id}`),
  create: (data) => api.post('/question-bank', data),
  update: (id, data) => api.put(`/question-bank/${id}`, data),
  delete: (id) => api.delete(`/question-bank/${id}`),
  toggle: (id, is_active) => api.patch(`/question-bank/${id}/toggle`, { is_active }),
  batchImport: (items) => api.post('/question-bank/batch-import', { items }),
  reindexAll: () => api.post('/question-bank/reindex-all'),
  testRetrieve: (data) => api.post('/question-bank/test-retrieve', data),
  stats: () => api.get('/question-bank/stats')
}

// ── 管理员管理 ───────────────────────────────────────────────────
export const adminApi = {
  list: (params) => api.get('/admins', { params }),
  detail: (id) => api.get(`/admins/${id}`),
  create: (data) => api.post('/admins', data),
  update: (id, data) => api.put(`/admins/${id}`, data),
  delete: (id) => api.delete(`/admins/${id}`),
  changePassword: (id, data) => api.post(`/admins/${id}/change-password`, data),
  resetPassword: (id, data) => api.post(`/admins/${id}/reset-password`, data)
}

// ── 岗位模板管理 ─────────────────────────────────────────────────
export const positionTemplateApi = {
  list: (params) => api.get('/position-templates', { params }),
  detail: (id) => api.get(`/position-templates/${id}`),
  create: (data) => api.post('/position-templates', data),
  update: (id, data) => api.put(`/position-templates/${id}`, data),
  delete: (id) => api.delete(`/position-templates/${id}`),
  toggle: (id, is_active) => api.patch(`/position-templates/${id}/toggle`, { is_active })
}

// ── 知识文档管理 ─────────────────────────────────────────────────
export const knowledgeApi = {
  uploadDocument: (formData) => api.post('/knowledge/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  list: (params) => api.get('/knowledge/documents', { params }),
  detail: (id) => api.get(`/knowledge/documents/${id}`),
  delete: (id) => api.delete(`/knowledge/documents/${id}`),
  toggle: (id, is_active) => api.patch(`/knowledge/documents/${id}/toggle`, { is_active }),
  reindex: (id) => api.post(`/knowledge/documents/${id}/reindex`),
  chunks: (id, params) => api.get(`/knowledge/documents/${id}/chunks`, { params }),
  testRetrieve: (data) => api.post('/knowledge/test-retrieve', data)
}
