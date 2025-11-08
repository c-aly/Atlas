/**
 * Landing Page Component
 * Beautiful, clean landing page for uploading images before entering the 3D visualization
 */
import { useState, useRef } from 'react'
import { uploadImagesBatch, projectTo3D, buildGraph, exportData } from '../api'
import { useStore } from '../store'

export default function LandingPage({ onComplete }) {
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingStep, setProcessingStep] = useState('')
  const [progress, setProgress] = useState(0)
  const fileInputRef = useRef(null)
  const { setImages, setEdges } = useStore()
  
  const handleFiles = async (files) => {
    if (!files || files.length === 0) return
    
    const fileArray = Array.from(files)
    setUploadedFiles(fileArray)
    
    // Wait a moment to show the preview
    setTimeout(() => processImages(fileArray), 500)
  }
  
  const processImages = async (files) => {
    try {
      setIsProcessing(true)
      setProgress(10)
      setProcessingStep('Analyzing images with AI...')
      
      // Upload and embed
      await uploadImagesBatch(files)
      setProgress(40)
      
      setProcessingStep('Creating 3D coordinates...')
      // Project to 3D
      await projectTo3D()
      setProgress(70)
      
      setProcessingStep('Building similarity network...')
      // Build graph
      await buildGraph(8)
      setProgress(90)
      
      setProcessingStep('Preparing visualization...')
      // Fetch data
      const data = await exportData()
      
      if (data.coords && data.coords.points) {
        const images = data.coords.points.map(point => ({
          id: point.id,
          coords: [point.x, point.y, point.z],
          thumb: data.meta?.[point.id]?.thumb || null,
          filename: data.meta?.[point.id]?.filename || 'Unknown',
          labels: data.meta?.[point.id]?.labels || []
        }))
        setImages(images)
      }
      
      if (data.graph && data.graph.edges) {
        setEdges(data.graph.edges)
      }
      
      setProgress(100)
      setProcessingStep('Complete!')
      
      // Redirect to main app
      setTimeout(() => {
        onComplete()
      }, 1000)
      
    } catch (error) {
      console.error('Processing error:', error)
      alert('Error processing images: ' + error.message)
      setIsProcessing(false)
      setUploadedFiles([])
    }
  }
  
  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    handleFiles(e.dataTransfer.files)
  }
  
  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }
  
  const handleDragLeave = () => {
    setIsDragging(false)
  }
  
  const handleFileSelect = (e) => {
    handleFiles(e.target.files)
  }
  
  if (isProcessing) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center p-6">
        <div className="max-w-2xl w-full">
          <div className="bg-slate-800/50 backdrop-blur-xl border border-blue-500/30 rounded-3xl shadow-2xl p-12 text-center">
            {/* Animated icon */}
            <div className="mb-8">
              <div className="inline-block text-8xl animate-pulse">🧭</div>
            </div>
            
            {/* Progress */}
            <h2 className="text-3xl font-bold text-white mb-4">
              {processingStep}
            </h2>
            
            <div className="w-full bg-slate-700 rounded-full h-3 mb-6 overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-blue-400 via-blue-600 to-cyan-500 transition-all duration-500 ease-out shadow-lg"
                style={{ width: `${progress}%` }}
              />
            </div>
            
            <p className="text-gray-300">
              Processing {uploadedFiles.length} image{uploadedFiles.length !== 1 ? 's' : ''}...
            </p>
            
            {/* File previews */}
            <div className="mt-8 flex flex-wrap gap-2 justify-center max-h-32 overflow-hidden">
              {uploadedFiles.slice(0, 6).map((file, i) => (
                <div key={i} className="text-xs text-gray-300 bg-slate-700/50 px-3 py-1 rounded-full border border-blue-500/30">
                  {file.name.slice(0, 20)}...
                </div>
              ))}
              {uploadedFiles.length > 6 && (
                <div className="text-xs text-gray-300 bg-slate-700/50 px-3 py-1 rounded-full border border-blue-500/30">
                  +{uploadedFiles.length - 6} more
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-950 via-blue-900 to-slate-900 overflow-y-auto">
      {/* Header */}
      <header className="p-8">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="text-4xl">🧭</div>
            <div>
              <h1 className="text-2xl font-bold text-white">Atlas of Images</h1>
              <p className="text-sm text-blue-300">Explore Visual Space</p>
            </div>
          </div>
        </div>
      </header>
      
      {/* Hero Section */}
      <div className="max-w-6xl mx-auto px-8 py-16">
        <div className="text-center mb-16">
          <h2 className="text-6xl font-bold text-white mb-6 leading-tight">
            Explore Images in<br/>
            <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent">
              3D Semantic Space
            </span>
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Upload your images and watch them transform into an interactive 3D neural network. 
            Discover hidden connections, search with natural language, and navigate through visual concepts.
          </p>
        </div>
        
        {/* Upload Area */}
        <div className="max-w-3xl mx-auto">
          <div
            className={`
              relative bg-slate-800/50 backdrop-blur-xl rounded-3xl shadow-2xl p-16 
              border-4 border-dashed transition-all duration-300
              ${isDragging 
                ? 'border-blue-400 bg-blue-900/30 scale-105 shadow-blue-500/50' 
                : 'border-blue-500/50 hover:border-blue-400 hover:shadow-blue-500/30 hover:shadow-3xl'
              }
            `}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            {/* Upload Icon */}
            <div className="text-center mb-8">
              <div className="inline-block mb-6">
                <div className="text-8xl mb-2 animate-bounce">📸</div>
              </div>
              
              <h3 className="text-3xl font-bold text-white mb-4">
                Drop Your Images Here
              </h3>
              
              <p className="text-lg text-gray-300 mb-8">
                or click to browse your files
              </p>
              
              <button
                onClick={() => fileInputRef.current?.click()}
                className="
                  px-8 py-4 bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500
                  text-white text-lg font-semibold rounded-full
                  shadow-lg shadow-blue-500/50 hover:shadow-xl hover:shadow-blue-500/70 
                  transform hover:scale-105
                  transition-all duration-200
                "
              >
                Choose Images
              </button>
              
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>
            
            {/* Info */}
            <div className="grid grid-cols-3 gap-6 mt-12 pt-8 border-t border-blue-500/30">
              <div className="text-center">
                <div className="text-3xl mb-2">🎨</div>
                <p className="text-sm font-semibold text-white">AI-Powered</p>
                <p className="text-xs text-gray-400 mt-1">CLIP embeddings</p>
              </div>
              <div className="text-center">
                <div className="text-3xl mb-2">🌐</div>
                <p className="text-sm font-semibold text-white">3D Visualization</p>
                <p className="text-xs text-gray-400 mt-1">Interactive exploration</p>
              </div>
              <div className="text-center">
                <div className="text-3xl mb-2">⚡</div>
                <p className="text-sm font-semibold text-white">Fast</p>
                <p className="text-xs text-gray-400 mt-1">Instant search</p>
              </div>
            </div>
          </div>
          
          {/* Tips */}
          <div className="mt-8 text-center text-sm text-gray-400">
            <p className="mb-2">💡 <strong className="text-blue-300">Pro tip:</strong> Upload 10-20+ diverse images for the best visualization</p>
            <p className="text-gray-500">Supported formats: JPG, PNG, WebP • Max 10MB per image</p>
          </div>
        </div>
      </div>
      
      {/* Features Section */}
      <div className="max-w-6xl mx-auto px-8 py-16">
        <div className="grid md:grid-cols-3 gap-8">
          <div className="bg-slate-800/50 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-8 shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40 transition-all duration-300">
            <div className="text-4xl mb-4">🔍</div>
            <h3 className="text-xl font-bold text-white mb-3">Natural Language Search</h3>
            <p className="text-gray-300">
              Search for "cozy desk with coffee" and find semantically similar images instantly.
            </p>
          </div>
          
          <div className="bg-slate-800/50 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-8 shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40 transition-all duration-300">
            <div className="text-4xl mb-4">👁️</div>
            <h3 className="text-xl font-bold text-white mb-3">First-Person Navigation</h3>
            <p className="text-gray-300">
              Enter any image node and explore the 3D space from inside. Look around in 360°.
            </p>
          </div>
          
          <div className="bg-slate-800/50 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-8 shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40 transition-all duration-300">
            <div className="text-4xl mb-4">📹</div>
            <h3 className="text-xl font-bold text-white mb-3">Live Webcam Mode</h3>
            <p className="text-gray-300">
              Watch your position update in real-time as you show different objects to your camera.
            </p>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <footer className="text-center py-12 text-gray-400 text-sm border-t border-blue-500/20">
        <p className="text-blue-300">Powered by CLIP • AWS Bedrock (Claude) • Three.js</p>
        <p className="mt-2">Built for HackUMass 2025</p>
      </footer>
    </div>
  )
}

