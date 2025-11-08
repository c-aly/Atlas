/**
 * Node Detail Panel Component
 * Shows details about selected node and its neighbors, or cluster legend when no node is selected
 */
import { useStore } from '../store'

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
              src={image.thumb || `http://localhost:8001/uploads/${selectedNode}`}
              alt={image.filename}
              className="w-full h-auto object-contain max-h-64"
              onError={(e) => {
                e.target.onerror = null;
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'flex';
              }}
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
              {neighbors.slice(0, 8).map((neighbor) => (
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
                        src={neighbor.image.thumb || `http://localhost:8001/uploads/${neighbor.id}`}
                        alt={neighbor.image.filename}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          e.target.onerror = null;
                          e.target.style.display = 'none';
                          const fallback = document.createElement('div');
                          fallback.className = 'w-8 h-8 flex items-center justify-center text-lg';
                          fallback.textContent = '🖼️';
                          e.target.parentNode.appendChild(fallback);
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
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

