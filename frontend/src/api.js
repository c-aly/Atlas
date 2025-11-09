/**
 * API client for backend communication
 */
import axios from 'axios'
import { supabase } from './lib/supabase'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minute timeout for heavy operations (CLIP embedding can be slow)
})

// Add auth token to requests
api.interceptors.request.use(async (config) => {
  const { data: { session } } = await supabase.auth.getSession()
  if (session?.access_token) {
    config.headers.Authorization = `Bearer ${session.access_token}`
  }
  return config
})

// Upload multiple images
export const uploadImagesBatch = async (files) => {
  const formData = new FormData()
  files.forEach(file => {
    formData.append('files', file)
  })
  
  const response = await api.post('/embed/batch', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

// Search by text
export const searchByText = async (query, k = 10) => {
  const response = await api.post('/search', {
    query,
    k
  })
  return response.data
}

// Search by image
export const searchByImage = async (file, k = 10) => {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await api.post(`/search/image?k=${k}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

// Export all data for visualization
export const exportData = async () => {
  const response = await api.get('/data/export')
  return response.data
}

// Get statistics
export const getStats = async () => {
  const response = await api.get('/stats')
  return response.data
}

// Health check
export const healthCheck = async () => {
  const response = await api.get('/health')
  return response.data
}

// Recompute clusters
export const recomputeClusters = async () => {
  const response = await api.post('/cluster/recompute')
  return response.data
}

// Generate image description using Google Gemini
export const describeImage = async (imageId) => {
  const response = await api.post(`/describe/image/${imageId}`)
  return response.data
}

// Get narration audio URL for an image
export const getNarrationUrl = (imageId) => {
  return `${import.meta.env.VITE_API_URL || 'http://localhost:8001'}/narrate/image/${imageId}`
}

// Get fresh signed URL for an image (useful when URLs expire)
export const getImageUrl = async (imageId, expiresIn = 3600) => {
  const response = await api.get(`/image/${imageId}/url`, {
    params: { expires_in: expiresIn }
  })
  return response.data
}

export default api

