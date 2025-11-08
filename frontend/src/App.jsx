import { useState, useEffect } from 'react'
import Scene3D from './components/Scene3D'
import UploadPanel from './components/UploadPanel'
import SearchBar from './components/SearchBar'
import StatsPanel from './components/StatsPanel'
import NodeDetailPanel from './components/NodeDetailPanel'
import WebcamMode from './components/WebcamMode'
import LoadingOverlay from './components/LoadingOverlay'
import FirstPersonControls from './components/FirstPersonControls'
import LandingPage from './components/LandingPage'
import { useStore } from './store'
import { exportData, healthCheck } from './api'

function App() {
  const [isInitialized, setIsInitialized] = useState(false)
  const [showLanding, setShowLanding] = useState(true)
  const { setImages, setEdges, isLoading, loadingMessage, images } = useStore()

  useEffect(() => {
    // Initialize app
    const init = async () => {
      try {
        // Check backend health
        const health = await healthCheck()
        console.log('Backend status:', health)
        
        // Try to load existing data
        if (health.total_images > 0) {
          const data = await exportData()
          
          // Transform data for frontend
          if (data.coords && data.coords.points) {
            const images = data.coords.points.map(point => ({
              id: point.id,
              coords: [point.x, point.y, point.z],
              thumb: data.meta?.[point.id]?.thumb || null,
              filename: data.meta?.[point.id]?.filename || 'Unknown',
              labels: data.meta?.[point.id]?.labels || []
            }))
            setImages(images)
            
            // Skip landing page if we have data
            setShowLanding(false)
          }
          
          // Load graph edges
          if (data.graph && data.graph.edges) {
            setEdges(data.graph.edges)
          }
        }
        
        setIsInitialized(true)
      } catch (error) {
        console.error('Initialization error:', error)
        setIsInitialized(true) // Continue anyway
      }
    }
    
    init()
  }, [setImages, setEdges])

  if (!isInitialized) {
    return (
      <div className="flex items-center justify-center h-screen bg-gradient-to-br from-blue-950 via-blue-900 to-slate-900">
        <div className="text-center">
          <div className="animate-pulse-glow text-6xl mb-4">🧭</div>
          <p className="text-gray-300 text-lg">Initializing Atlas...</p>
        </div>
      </div>
    )
  }
  
  // Show landing page if no images
  if (showLanding) {
    return <LandingPage onComplete={() => setShowLanding(false)} />
  }

  return (
    <div className="relative w-screen h-screen bg-neural-bg overflow-hidden fixed inset-0">
      {/* 3D Scene */}
      <Scene3D />
      
      {/* Header */}
      <header className="absolute top-0 left-0 right-0 z-10 p-6 bg-gradient-to-b from-neural-bg to-transparent">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="text-3xl">🧭</div>
            <div>
              <h1 className="text-2xl font-bold text-white">Atlas of Images</h1>
              <p className="text-sm text-gray-400">3D Neural Map of Visual Space</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setShowLanding(true)}
              className="
                px-4 py-2 bg-gray-700 hover:bg-gray-600
                text-white text-sm font-medium rounded-lg
                transition-colors duration-200
              "
              title="Return to landing page"
            >
              ← Upload More
            </button>
            <StatsPanel />
          </div>
        </div>
      </header>
      
      {/* Left Sidebar - Upload & Controls */}
      <div className="absolute top-24 left-6 z-10 space-y-4 w-80">
        <UploadPanel />
        <SearchBar />
        <WebcamMode />
      </div>
      
      {/* Right Sidebar - Node Details */}
      <NodeDetailPanel />
      
      {/* First-Person Navigation Controls */}
      <FirstPersonControls />
      
      {/* Loading Overlay */}
      {isLoading && <LoadingOverlay message={loadingMessage} />}
      
      {/* Instructions */}
      <div className="absolute bottom-6 left-6 z-10 text-sm text-gray-500 space-y-1">
        <p>🖱️ <span className="text-gray-400">Drag to rotate • Scroll to zoom</span></p>
        <p>🎯 <span className="text-gray-400">Click nodes to explore • Hover for preview</span></p>
        <p>👁️ <span className="text-gray-400">Enter first-person mode to navigate inside</span></p>
      </div>
    </div>
  )
}

export default App

