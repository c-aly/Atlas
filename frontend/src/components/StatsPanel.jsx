/**
 * Stats Panel Component
 * Displays statistics about the current dataset
 */
import { useState, useEffect } from 'react'
import { useStore } from '../store'
import { getStats, recomputeClusters, exportData } from '../api'

export default function StatsPanel() {
  const { images, edges, setImages, setEdges } = useStore()
  const [backendStats, setBackendStats] = useState(null)
  const [isReclustering, setIsReclustering] = useState(false)
  
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const stats = await getStats()
        setBackendStats(stats)
      } catch (error) {
        console.error('Failed to fetch stats:', error)
      }
    }
    
    fetchStats()
    const interval = setInterval(fetchStats, 10000) // Update every 10s
    
    return () => clearInterval(interval)
  }, [])
  
  const handleRecluster = async () => {
    if (isReclustering) return
    
    try {
      setIsReclustering(true)
      const result = await recomputeClusters()
      console.log('Re-clustering result:', result)
      
      // Reload data to get new cluster assignments
      const data = await exportData()
      if (data.coords && data.coords.points) {
        const newImages = data.coords.points.map(point => ({
          id: point.id,
          coords: [point.x, point.y, point.z],
          thumb: data.meta?.[point.id]?.thumb || null,
          filename: data.meta?.[point.id]?.filename || 'Unknown',
          labels: data.meta?.[point.id]?.labels || [],
          cluster: data.meta?.[point.id]?.cluster || 0
        }))
        setImages(newImages)
      }
      
      if (data.graph && data.graph.edges) {
        setEdges(data.graph.edges)
      }
      
      alert(`Re-clustered ${result.n_images} images into ${result.n_clusters} clusters!`)
    } catch (error) {
      console.error('Error re-clustering:', error)
      alert('Error re-clustering: ' + (error.response?.data?.detail || error.message))
    } finally {
      setIsReclustering(false)
    }
  }
  
  return (
    <div className="bg-neural-card/80 backdrop-blur-sm rounded-lg px-4 py-2 border border-gray-700">
      <div className="flex items-center space-x-6 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-neural-accent rounded-full animate-pulse-glow"></div>
          <span className="text-gray-400">Images:</span>
          <span className="text-white font-semibold">{images.length}</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse-glow"></div>
          <span className="text-gray-400">Edges:</span>
          <span className="text-white font-semibold">{edges.length}</span>
        </div>
        
        {backendStats && (
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse-glow"></div>
            <span className="text-gray-400">Embedding:</span>
            <span className="text-white font-semibold">CLIP</span>
          </div>
        )}
        
        {images.length > 1 && (
          <button
            onClick={handleRecluster}
            disabled={isReclustering}
            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white text-xs rounded transition-colors"
            title="Re-cluster all images with updated settings"
          >
            {isReclustering ? 'Clustering...' : '🔀 Re-cluster'}
          </button>
        )}
      </div>
    </div>
  )
}

