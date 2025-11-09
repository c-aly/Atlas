/**
 * Node Detail Panel Component
 * Shows details about selected node and its neighbors, or cluster legend when no node is selected
 */
import { useState, useEffect, useRef } from 'react'
import { useStore } from '../store'
import { describeImage, getNarrationUrl, getImageUrl } from '../api'
import { supabase } from '../lib/supabase'

// Cluster color palette (matching Scene3D.jsx)
const CLUSTER_COLORS = [
  '#3b82f6', // Bright Blue
  '#ef4444', // Bright Red
  '#10b981', // Bright Green
  '#f59e0b', // Amber/Orange
  '#8b5cf6', // Purple
  '#ec4899', // Pink
  '#06b6d4', // Cyan
  '#f97316', // Orange
  '#84cc16', // Lime Green
  '#14b8a6', // Teal
  '#a855f7', // Violet
  '#f43f5e', // Rose
  '#22c55e', // Emerald
  '#eab308', // Yellow
  '#6366f1', // Indigo
  '#d946ef', // Fuchsia
  '#06b6d4', // Sky Blue
  '#f97316', // Orange Red
  '#10b981', // Green
  '#3b82f6', // Blue
]

export default function NodeDetailPanel() {
  const { selectedNode, getImageById, getNeighbors, setSelectedNode, images } = useStore()
  const [description, setDescription] = useState(null)
  const [isGeneratingDescription, setIsGeneratingDescription] = useState(false)
  const [audioUrl, setAudioUrl] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [narrationError, setNarrationError] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [neighborUrls, setNeighborUrls] = useState({}) // Cache URLs for neighbor images
  const audioRef = useRef(null)
  
  // Generate description and refresh image URL when node is selected
  useEffect(() => {
    if (selectedNode) {
      const image = getImageById(selectedNode)
      
      // Set initial image URL from metadata
      setImageUrl(image?.thumb || null)
      
      // Refresh image URL if it exists (in case it expired)
      if (image?.thumb) {
        // Try to get a fresh signed URL
        getImageUrl(selectedNode, 86400) // 24 hour expiration
          .then(result => {
            if (result.success && result.url) {
              setImageUrl(result.url)
              console.log('Refreshed image URL for', selectedNode)
            }
          })
          .catch(error => {
            console.warn('Could not refresh image URL, using stored URL:', error)
            // Keep using the stored URL
          })
      }
      
      // Check if image already has description in metadata
      if (image?.description) {
        setDescription(image.description)
        setIsGeneratingDescription(false)
        // Set up audio URL for narration
        setAudioUrl(getNarrationUrl(selectedNode))
      } else {
        // Generate description
        setIsGeneratingDescription(true)
        setDescription(null)
        setAudioUrl(null)
        describeImage(selectedNode)
          .then(result => {
            if (result.success) {
              setDescription(result.description)
              // Set up audio URL for narration
              setAudioUrl(getNarrationUrl(selectedNode))
            }
          })
          .catch(error => {
            console.error('Error generating description:', error)
            setDescription(null)
            setAudioUrl(null)
          })
          .finally(() => {
            setIsGeneratingDescription(false)
          })
      }
    } else {
      setDescription(null)
      setIsGeneratingDescription(false)
      setAudioUrl(null)
      setIsPlaying(false)
      setNarrationError(null)
      setImageUrl(null)
    }
    
    // Cleanup audio when component unmounts or node changes
    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current = null
      }
      setIsPlaying(false)
    }
  }, [selectedNode, getImageById])
  
  // Handle image load error - try to refresh URL
  const handleImageError = async (e) => {
    console.warn('Image failed to load, attempting to refresh URL for', selectedNode)
    e.target.onerror = null // Prevent infinite loop
    
    if (selectedNode) {
      try {
        const result = await getImageUrl(selectedNode, 86400)
        if (result.success && result.url) {
          setImageUrl(result.url)
          // Update the image src
          e.target.src = result.url
          console.log('Refreshed and updated image URL')
        } else {
          // Hide image and show placeholder
          e.target.style.display = 'none'
          if (e.target.nextSibling) {
            e.target.nextSibling.style.display = 'flex'
          }
        }
      } catch (error) {
        console.error('Failed to refresh image URL:', error)
        // Hide image and show placeholder
        e.target.style.display = 'none'
        if (e.target.nextSibling) {
          e.target.nextSibling.style.display = 'flex'
        }
      }
    }
  }
  
  // Handle audio playback
  const handlePlayNarration = async () => {
    if (!audioUrl) return
    
    // Stop any currently playing audio
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }
    
    // Fetch audio with auth token
    try {
      const { data: { session } } = await supabase.auth.getSession()
      
      const response = await fetch(audioUrl, {
        headers: {
          'Authorization': `Bearer ${session?.access_token || ''}`
        }
      })
      
      if (!response.ok) {
        // Try to get error message from response
        let errorMessage = `Failed to fetch audio (${response.status})`
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorMessage
        } catch {
          // If response isn't JSON, use status text
          errorMessage = response.statusText || errorMessage
        }
        throw new Error(errorMessage)
      }
      
      const blob = await response.blob()
      const audioObjectUrl = URL.createObjectURL(blob)
      
      const audio = new Audio(audioObjectUrl)
      audioRef.current = audio
      
      audio.onplay = () => setIsPlaying(true)
      audio.onended = () => {
        setIsPlaying(false)
        URL.revokeObjectURL(audioObjectUrl)
        audioRef.current = null
      }
      audio.onerror = (e) => {
        setIsPlaying(false)
        console.error('Error playing narration audio:', e)
        URL.revokeObjectURL(audioObjectUrl)
        audioRef.current = null
      }
      
      audio.play().catch(error => {
        console.error('Error playing audio:', error)
        setIsPlaying(false)
        setNarrationError('Failed to play audio. Please try again.')
        URL.revokeObjectURL(audioObjectUrl)
        audioRef.current = null
      })
    } catch (error) {
      console.error('Error loading narration audio:', error)
      setIsPlaying(false)
      setNarrationError(error.message || 'Failed to load narration. Please check your ElevenLabs API key.')
    }
  }
  
  const handleStopNarration = () => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      audioRef.current = null
      setIsPlaying(false)
    }
  }
  
  // Show legend when no node is selected
  if (!selectedNode) {
    // Get all unique clusters
    const clusterSet = new Set()
    images.forEach(img => {
      if (img.cluster !== undefined && img.cluster !== null) {
        clusterSet.add(img.cluster)
      }
    })
    const clusters = Array.from(clusterSet).sort((a, b) => a - b)
    
    // Count images per cluster
    const clusterCounts = new Map()
    images.forEach(img => {
      const cluster = img.cluster || 0
      clusterCounts.set(cluster, (clusterCounts.get(cluster) || 0) + 1)
    })
    
    return (
      <div className="absolute top-24 right-6 z-10 w-80 max-h-[calc(100vh-8rem)] overflow-y-auto">
        <div className="bg-neural-card rounded-lg shadow-xl border border-gray-700 animate-fade-in">
          {/* Header */}
          <div className="p-4 border-b border-gray-700">
            <h3 className="text-lg font-semibold text-white">Cluster Legend</h3>
            <p className="text-sm text-gray-400 mt-1">Click a node to see image details</p>
          </div>
          
          {/* Cluster List */}
          <div className="p-4 space-y-2">
            {clusters.length === 0 ? (
              <p className="text-gray-400 text-sm text-center py-4">No clusters found</p>
            ) : (
              clusters.map((clusterId) => {
                const color = CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length]
                const count = clusterCounts.get(clusterId) || 0
                return (
                  <div
                    key={clusterId}
                    className="flex items-center justify-between p-3 bg-neural-bg rounded-lg border border-gray-700"
                  >
                    <div className="flex items-center space-x-3">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: color }}
                      />
                      <div>
                        <p className="text-sm font-medium text-white">Cluster {clusterId + 1}</p>
                        <p className="text-xs text-gray-400">{count} image{count !== 1 ? 's' : ''}</p>
                      </div>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>
      </div>
    )
  }
  
  const image = getImageById(selectedNode)
  const neighbors = getNeighbors(selectedNode)
  
  if (!image) return null
  
  return (
    <div className="absolute top-24 right-6 z-10 w-80 max-h-[calc(100vh-8rem)] overflow-y-auto">
      <div className="bg-neural-card rounded-lg shadow-xl border border-gray-700 animate-fade-in">
        {/* Header */}
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white">Image Details</h3>
          <button
            onClick={() => setSelectedNode(null)}
            className="text-gray-400 hover:text-white transition-colors"
          >
            ✕
          </button>
        </div>
        
        {/* Image Info */}
        <div className="p-4 space-y-3">
          {/* Actual Image */}
          <div className="bg-neural-bg rounded-lg overflow-hidden">
            <img 
              src={imageUrl || image.thumb || ''}
              alt={image.filename}
              className="w-full h-auto object-contain max-h-64"
              onError={handleImageError}
            />
            <div className="hidden flex-col items-center justify-center p-8">
              <div className="text-5xl mb-2">🖼️</div>
              <p className="text-sm text-gray-400">Image not available</p>
            </div>
            <p className="text-sm text-gray-400 text-center py-2 border-t border-gray-700">{image.filename}</p>
          </div>
          
          {/* Metadata */}
          <div className="space-y-2">
            <div className="text-xs">
              <span className="text-gray-400">ID:</span>
              <span className="text-gray-300 ml-2 font-mono text-xs">
                {selectedNode.slice(0, 8)}...
              </span>
            </div>
            
            <div className="text-xs">
              <span className="text-gray-400">Position:</span>
              <span className="text-gray-300 ml-2 font-mono text-xs">
                ({image.coords[0].toFixed(2)}, {image.coords[1].toFixed(2)}, {image.coords[2].toFixed(2)})
              </span>
            </div>
            
            {image.labels && image.labels.length > 0 && (
              <div>
                <span className="text-gray-400 text-xs">Labels:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {image.labels.map((label, i) => (
                    <span
                      key={i}
                      className="px-2 py-1 bg-neural-accent/20 text-neural-glow text-xs rounded"
                    >
                      {label.name}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {/* Image Description */}
          <div className="mt-4 pt-4 border-t border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-semibold text-white">Description</h4>
              {description && audioUrl && !narrationError && (
                <div className="flex items-center space-x-2">
                  {isPlaying ? (
                    <button
                      onClick={handleStopNarration}
                      className="px-2 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors flex items-center space-x-1"
                      title="Stop narration"
                    >
                      <span>⏹</span>
                      <span>Stop</span>
                    </button>
                  ) : (
                    <button
                      onClick={handlePlayNarration}
                      className="px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors flex items-center space-x-1"
                      title="Play narration"
                    >
                      <span>▶</span>
                      <span>Listen</span>
                    </button>
                  )}
                </div>
              )}
            </div>
            {narrationError && (
              <div className="mt-2 p-2 bg-red-900/30 border border-red-700 rounded text-xs text-red-300">
                {narrationError}
              </div>
            )}
            {isGeneratingDescription ? (
              <div className="flex items-center space-x-2 text-gray-400 text-sm">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                <span>Generating description...</span>
              </div>
            ) : description ? (
              <p className="text-sm text-gray-300 leading-relaxed">{description}</p>
            ) : (
              <p className="text-sm text-gray-500 italic">No description available</p>
            )}
          </div>
        </div>
        
        {/* Neighbors */}
        {neighbors.length > 0 && (
          <>
            <div className="px-4 py-2 border-t border-gray-700">
              <h4 className="text-sm font-semibold text-white">
                Similar Images ({neighbors.length})
              </h4>
            </div>
            
            <div className="p-4 space-y-2 max-h-64 overflow-y-auto">
              {neighbors.slice(0, 8).map((neighbor) => {
                // Get fresh URL for this neighbor if not already cached
                const neighborUrl = neighborUrls[neighbor.id] || neighbor.image.thumb
                
                // Fetch fresh URL on mount if not cached
                if (!neighborUrls[neighbor.id] && neighbor.image.thumb) {
                  getImageUrl(neighbor.id, 86400)
                    .then(result => {
                      if (result.success && result.url) {
                        setNeighborUrls(prev => ({ ...prev, [neighbor.id]: result.url }))
                      }
                    })
                    .catch(error => {
                      console.warn(`Could not refresh URL for neighbor ${neighbor.id}:`, error)
                    })
                }
                
                return (
                  <div
                    key={neighbor.id}
                    onClick={() => setSelectedNode(neighbor.id)}
                    className="
                      flex items-center justify-between p-2 
                      bg-neural-bg hover:bg-gray-700 rounded-lg 
                      cursor-pointer transition-colors
                    "
                  >
                    <div className="flex items-center space-x-2">
                      <div className="w-8 h-8 bg-neural-accent/20 rounded overflow-hidden flex-shrink-0">
                        <img 
                          src={neighborUrl || ''}
                          alt={neighbor.image.filename}
                          className="w-full h-full object-cover"
                          onError={async (e) => {
                            e.target.onerror = null;
                            // Try to refresh URL
                            try {
                              const result = await getImageUrl(neighbor.id, 86400)
                              if (result.success && result.url) {
                                setNeighborUrls(prev => ({ ...prev, [neighbor.id]: result.url }))
                                e.target.src = result.url
                              } else {
                                e.target.style.display = 'none';
                                const fallback = document.createElement('div');
                                fallback.className = 'w-8 h-8 flex items-center justify-center text-lg';
                                fallback.textContent = '🖼️';
                                e.target.parentNode.appendChild(fallback);
                              }
                            } catch (error) {
                              e.target.style.display = 'none';
                              const fallback = document.createElement('div');
                              fallback.className = 'w-8 h-8 flex items-center justify-center text-lg';
                              fallback.textContent = '🖼️';
                              e.target.parentNode.appendChild(fallback);
                            }
                          }}
                        />
                      </div>
                      <div>
                        <p className="text-sm text-gray-300 truncate max-w-[150px]">
                          {neighbor.image.filename}
                        </p>
                        <p className="text-xs text-gray-500">
                          {(neighbor.weight * 100).toFixed(0)}% similar
                        </p>
                      </div>
                    </div>
                    <div className="text-gray-400">→</div>
                  </div>
                )
              })}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

