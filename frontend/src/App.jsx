import { useState, useEffect } from 'react'
import Scene3D from './components/Scene3D'
import UploadPanel from './components/UploadPanel'
import SearchBar from './components/SearchBar'
import StatsPanel from './components/StatsPanel'
import NodeDetailPanel from './components/NodeDetailPanel'
import LoadingOverlay from './components/LoadingOverlay'
import LandingPage from './components/LandingPage'
import { useStore } from './store'
import { exportData, healthCheck, recomputeClusters } from './api'
import { supabase } from './lib/supabase'

function App() {
  const [isInitialized, setIsInitialized] = useState(false)
  const [showLanding, setShowLanding] = useState(true)
  const { setImages, setEdges, isLoading, loadingMessage, images, setUser, setSession, isAuthenticated } = useStore()

  // Initialize auth state
  useEffect(() => {
    const initAuth = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession()
        if (session) {
          setSession(session)
          setUser(session.user)
        }
        
        // Listen for auth changes
        const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
          if (session) {
            setSession(session)
            setUser(session.user)
          } else {
            setSession(null)
            setUser(null)
          }
        })
        
        return () => subscription.unsubscribe()
      } catch (error) {
        console.error('Auth initialization error:', error)
      }
    }
    
    initAuth()
  }, [setUser, setSession])

  useEffect(() => {
    // Only initialize if authenticated
    if (!isAuthenticated) {
      setIsInitialized(true)
      return
    }
    
    // Initialize app
    const init = async () => {
      try {
        // Check backend health to see if user has images
        const health = await healthCheck()
        console.log('Backend status:', health)
        
        // If user has at least 3 images, load them and go straight to map
        if (health.total_images >= 3) {
          // Check if clusters need to be recomputed (if all have cluster 0 or missing)
          const data = await exportData()
          
          // Check if clustering is needed
          const hasClusters = data.coords?.points?.some(point => {
            const cluster = data.meta?.[point.id]?.cluster
            return cluster !== undefined && cluster !== null && cluster !== 0
          })
          
          // If no clusters found, try to recompute them
          if (!hasClusters && data.coords?.points?.length > 1) {
            console.log('No clusters found, recomputing...')
            try {
              await recomputeClusters()
              // Reload data after clustering
              const newData = await exportData()
              if (newData.coords && newData.coords.points) {
                const images = newData.coords.points.map(point => ({
                  id: point.id,
                  coords: [point.x, point.y, point.z],
                  thumb: newData.meta?.[point.id]?.thumb || null,
                  filename: newData.meta?.[point.id]?.filename || 'Unknown',
                  labels: newData.meta?.[point.id]?.labels || [],
                  cluster: newData.meta?.[point.id]?.cluster || 0
                }))
                setImages(images)
                
                if (newData.graph && newData.graph.edges) {
                  setEdges(newData.graph.edges)
                }
                setShowLanding(false)
                setIsInitialized(true)
                return
              }
            } catch (error) {
              console.error('Error recomputing clusters:', error)
              // Continue with existing data
            }
          }
          
          // Transform data for frontend
          if (data.coords && data.coords.points) {
            const images = data.coords.points.map(point => ({
              id: point.id,
              coords: [point.x, point.y, point.z],
              thumb: data.meta?.[point.id]?.thumb || null,
              filename: data.meta?.[point.id]?.filename || 'Unknown',
              labels: data.meta?.[point.id]?.labels || [],
              cluster: data.meta?.[point.id]?.cluster || 0
            }))
            setImages(images)
            
            // Skip landing page - go straight to map
            setShowLanding(false)
          }
          
          // Load graph edges
          if (data.graph && data.graph.edges) {
            console.log(`Loaded ${data.graph.edges.length} edges from export`)
            setEdges(data.graph.edges)
          } else {
            console.warn('No graph edges in export data:', data)
          }
        } else {
          // User has less than 3 images, show landing page to upload more
          setShowLanding(true)
        }
        
        setIsInitialized(true)
      } catch (error) {
        console.error('Initialization error:', error)
        // If there's an error, show landing page to allow upload
        setShowLanding(true)
        setIsInitialized(true)
      }
    }
    
    init()
  }, [setImages, setEdges, isAuthenticated])

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
  
  // Show landing page if not authenticated or if no images
  if (!isAuthenticated || showLanding) {
    return <LandingPage onComplete={() => setShowLanding(false)} />
  }

  return (
    <div className="relative w-screen h-screen overflow-hidden fixed inset-0" style={{
      background: 'radial-gradient(ellipse at center, #2d1b4e 0%, #1a0d2e 30%, #0f0519 60%, #050210 100%)'
    }}>
      {/* 3D Scene */}
      <Scene3D />
      
      {/* Header */}
      <header className="absolute top-0 left-0 z-10 p-6 bg-gradient-to-b from-black/50 to-transparent backdrop-blur-sm">
        <div className="flex items-center space-x-3">
          <div className="text-3xl">🧭</div>
          <div>
            <h1 className="text-2xl font-bold text-white">Atlas of Images</h1>
            <p className="text-sm text-gray-400">3D Neural Map of Visual Space</p>
          </div>
        </div>
      </header>
      
      {/* Top Right Controls */}
      <div className="absolute top-0 right-0 z-10 p-6">
        <StatsPanel />
      </div>
      
      {/* Left Sidebar - Upload & Controls */}
      <div className="absolute top-24 left-6 z-10 space-y-4 w-80 max-h-[calc(100vh-8rem)] overflow-y-auto pr-2">
        <UploadPanel />
        <SearchBar />
      </div>
      
      {/* Right Sidebar - Node Details */}
      <NodeDetailPanel />
      
      {/* Loading Overlay */}
      {isLoading && <LoadingOverlay message={loadingMessage} />}
      
      {/* Instructions */}
      <div className="absolute bottom-6 left-6 z-10 text-sm text-gray-500 space-y-1">
        <p>🖱️ <span className="text-gray-400">Drag to rotate • Scroll to zoom</span></p>
        <p>🎯 <span className="text-gray-400">Click nodes to explore • Hover for preview</span></p>
      </div>
    </div>
  )
}

export default App

