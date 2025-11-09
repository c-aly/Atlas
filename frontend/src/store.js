/**
 * Zustand store for global state management
 */
import { create } from 'zustand'

export const useStore = create((set, get) => ({
  // Image data
  images: [], // Array of {id, coords, thumb, filename, labels}
  imageMap: new Map(), // Quick lookup by id
  
  // Graph data
  edges: [], // Array of {source, target, weight}
  
  // UI state
  selectedNode: null,
  hoveredNode: null,
  searchResults: [],
  isLoading: false,
  loadingMessage: '',
  
  // Auto-play state
  isAutoPlaying: false,
  
  // Auth state
  user: null,
  session: null,
  isAuthenticated: false,
  
  // Actions
  setImages: (images) => {
    const imageMap = new Map()
    images.forEach(img => imageMap.set(img.id, img))
    set({ images, imageMap })
  },
  
  setEdges: (edges) => set({ edges }),
  
  setSelectedNode: (nodeId) => set({ selectedNode: nodeId }),
  
  setHoveredNode: (nodeId) => set({ hoveredNode: nodeId }),
  
  setSearchResults: (results) => set({ searchResults: results }),
  
  setLoading: (isLoading, message = '') => 
    set({ isLoading, loadingMessage: message }),
  
  setUser: (user) => set({ user, isAuthenticated: !!user }),
  
  setSession: (session) => set({ session, isAuthenticated: !!session }),
  
  setIsAutoPlaying: (isAutoPlaying) => set({ isAutoPlaying }),
  
  getImageById: (id) => {
    return get().imageMap.get(id)
  },
  
  getNeighbors: (nodeId) => {
    const edges = get().edges
    const neighbors = []
    
    // Check both directions (source -> target and target -> source)
    edges.forEach(e => {
      if (e.source === nodeId) {
        const image = get().imageMap.get(e.target)
        if (image) {
          neighbors.push({
            id: e.target,
            weight: e.weight,
            image: image
          })
        }
      } else if (e.target === nodeId) {
        const image = get().imageMap.get(e.source)
        if (image) {
          neighbors.push({
            id: e.source,
            weight: e.weight,
            image: image
          })
        }
      }
    })
    
    // Sort by similarity (highest first) and return
    return neighbors.sort((a, b) => b.weight - a.weight)
  },
}))

