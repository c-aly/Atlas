/**
 * First-Person Navigation Controls
 * Provides UI for entering/exiting first-person mode and navigating between nodes
 */
import { useStore } from '../store'

export default function FirstPersonControls() {
  const {
    isFirstPerson,
    firstPersonNode,
    selectedNode,
    enterFirstPerson,
    exitFirstPerson,
    getNeighbors,
    navigateToNode,
    getImageById
  } = useStore()
  
  const neighbors = firstPersonNode ? getNeighbors(firstPersonNode) : []
  const currentImage = firstPersonNode ? getImageById(firstPersonNode) : null
  
  // If not in first-person and a node is selected, show "Enter" button
  if (!isFirstPerson && selectedNode) {
    return (
      <div className="absolute bottom-24 right-6 z-10">
        <button
          onClick={() => enterFirstPerson(selectedNode)}
          className="
            px-6 py-3 bg-green-600 hover:bg-green-700
            text-white font-semibold rounded-lg shadow-xl
            transition-all duration-200 flex items-center space-x-2
          "
        >
          <span>👁️</span>
          <span>Enter First-Person View</span>
        </button>
      </div>
    )
  }
  
  // If in first-person, show navigation controls
  if (isFirstPerson && firstPersonNode) {
    return (
      <>
        {/* Exit button */}
        <div className="absolute top-24 left-1/2 -translate-x-1/2 z-10">
          <button
            onClick={exitFirstPerson}
            className="
              px-6 py-3 bg-red-600 hover:bg-red-700
              text-white font-semibold rounded-lg shadow-xl
              transition-all duration-200 flex items-center space-x-2
            "
          >
            <span>↩️</span>
            <span>Exit First-Person</span>
          </button>
        </div>
        
        {/* Current node info */}
        <div className="absolute top-24 left-6 z-10 bg-neural-card/90 backdrop-blur-sm rounded-lg p-4 shadow-xl border border-gray-700 max-w-xs">
          <div className="flex items-center space-x-3 mb-2">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-white font-semibold">First-Person Mode</span>
          </div>
          {currentImage && (
            <p className="text-sm text-gray-300 truncate">{currentImage.filename}</p>
          )}
          <p className="text-xs text-gray-500 mt-2">Click on nodes to navigate</p>
        </div>
        
        {/* Navigation arrows - nearby nodes */}
        {neighbors.length > 0 && (
          <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10">
            <div className="bg-neural-card/90 backdrop-blur-sm rounded-lg p-4 shadow-xl border border-gray-700">
              <p className="text-xs text-gray-400 text-center mb-3">Nearby Nodes ({neighbors.length})</p>
              <div className="grid grid-cols-4 gap-2">
                {neighbors.slice(0, 8).map((neighbor, idx) => (
                  <button
                    key={neighbor.id}
                    onClick={() => navigateToNode(neighbor.id)}
                    className="
                      w-16 h-16 bg-neural-bg hover:bg-neural-accent/20
                      rounded-lg border-2 border-gray-600 hover:border-neural-accent
                      transition-all duration-200 overflow-hidden
                      group relative
                    "
                    title={neighbor.image.filename}
                  >
                    {/* Thumbnail image */}
                    <img 
                      src={`http://localhost:8000/images/${neighbor.id}`}
                      alt={neighbor.image.filename}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.style.display = 'none';
                      }}
                    />
                    {/* Similarity badge */}
                    <div className="
                      absolute bottom-0 left-0 right-0 
                      bg-black/70 text-white text-xs py-0.5 text-center
                    ">
                      {(neighbor.weight * 100).toFixed(0)}%
                    </div>
                    {/* Tooltip */}
                    <div className="
                      absolute bottom-full mb-2 left-1/2 -translate-x-1/2
                      bg-neural-card px-2 py-1 rounded text-xs whitespace-nowrap
                      opacity-0 group-hover:opacity-100 transition-opacity
                      pointer-events-none z-50
                    ">
                      {neighbor.image.filename.slice(0, 25)}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {/* Instructions */}
        <div className="absolute bottom-6 right-6 z-10 bg-neural-card/80 backdrop-blur-sm rounded-lg p-3 shadow-xl border border-gray-700 max-w-xs space-y-1.5">
          <p className="text-xs font-semibold text-neural-glow mb-2">First-Person Controls:</p>
          <p className="text-xs text-gray-400">
            <span className="text-neural-glow">🖱️ Drag:</span> Look around in 360°
          </p>
          <p className="text-xs text-gray-400">
            <span className="text-neural-glow">👆 Click Node:</span> Travel to that image
          </p>
          <p className="text-xs text-gray-400">
            <span className="text-neural-glow">📊 Panel Below:</span> Quick navigation to nearby nodes
          </p>
        </div>
      </>
    )
  }
  
  return null
}

