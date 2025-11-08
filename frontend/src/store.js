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
  
  // First-person navigation state
  isFirstPerson: false,
  firstPersonNode: null,
  
  // Webcam state
  isWebcamActive: false,
  webcamPosition: null,
  
  // Cluster state
  clusters: [], // Array of {id, title, description, members}
  
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
  
  setWebcamActive: (isActive) => set({ isWebcamActive: isActive }),
  
  setWebcamPosition: (position) => set({ webcamPosition: position }),
  
  setClusters: (clusters) => set({ clusters }),
  
  // First-person navigation
  enterFirstPerson: (nodeId) => set({ 
    isFirstPerson: true, 
    firstPersonNode: nodeId,
    selectedNode: nodeId 
  }),
  
  exitFirstPerson: () => set({ 
    isFirstPerson: false, 
    firstPersonNode: null 
  }),
  
  navigateToNode: (nodeId) => {
    if (get().isFirstPerson) {
      set({ firstPersonNode: nodeId, selectedNode: nodeId })
    }
  },
  
  getImageById: (id) => {
    return get().imageMap.get(id)
  },
  
  getNeighbors: (nodeId) => {
    const edges = get().edges
    return edges
      .filter(e => e.source === nodeId)
      .map(e => ({
        id: e.target,
        weight: e.weight,
        image: get().imageMap.get(e.target)
      }))
      .filter(n => n.image) // Filter out missing images
      .sort((a, b) => b.weight - a.weight) // Sort by similarity
  },
}))

