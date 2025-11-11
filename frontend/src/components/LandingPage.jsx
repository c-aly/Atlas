/**
 * Landing Page Component
 * Shows auth first, then upload interface with at least 3 images requirement
 */
import { useState, useRef } from 'react'
import { uploadImagesBatch, exportData } from '../api'
import { useStore } from '../store'
import AuthModal from './AuthModal'
import { supabase } from '../lib/supabase'

export default function LandingPage({ onComplete }) {
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingStep, setProcessingStep] = useState('')
  const [progress, setProgress] = useState(0)
  const fileInputRef = useRef(null)
  const { setImages, setEdges, isAuthenticated, user } = useStore()
  
  // Show auth if not authenticated
  if (!isAuthenticated) {
    return <AuthModal onAuthSuccess={() => {}} />
  }
  
  const handleFiles = async (files) => {
    if (!files || files.length === 0) return
    
    const fileArray = Array.from(files).filter(file => file.type.startsWith('image/'))
    setUploadedFiles(prev => [...prev, ...fileArray])
  }
  
  const removeFile = (index) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index))
  }
  
  const handleStart = async () => {
    if (uploadedFiles.length < 3) {
      alert('Please upload at least 3 images to generate the 3D map')
      return
    }
    
    try {
      setIsProcessing(true)
      setProgress(20)
      setProcessingStep('Uploading and processing images...')
      
      // Upload, embed, and project to 3D (all in one call)
      await uploadImagesBatch(uploadedFiles)
      setProgress(80)
      
      setProcessingStep('Preparing visualization...')
      // Fetch data
      const data = await exportData()
      
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
      console.error('Full error:', error.response?.data)
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error'
      alert('Error processing images: ' + errorMessage)
      setIsProcessing(false)
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
    e.target.value = '' // Reset input to allow selecting same file again
  }
  
  const handleSignOut = async () => {
    await supabase.auth.signOut()
    window.location.reload()
  }
  
  if (isProcessing) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center p-6">
        <div className="max-w-2xl w-full">
          <div className="bg-slate-800/50 backdrop-blur-xl border border-blue-500/30 rounded-3xl shadow-2xl p-12 text-center">
            <div className="mb-8">
              <div className="inline-block text-8xl animate-pulse">🧭</div>
            </div>
            
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
              <p className="text-sm text-blue-300">Welcome, {user?.email}</p>
            </div>
          </div>
          <button
            onClick={handleSignOut}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded-lg transition-colors"
          >
            Sign Out
          </button>
        </div>
      </header>
      
      {/* Hero Section */}
      <div className="max-w-6xl mx-auto px-8 py-16">
        <div className="text-center mb-16">
          <h2 className="text-6xl font-bold text-white mb-6 leading-tight">
            Upload Your Images
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Upload at least 3 images to generate your 3D semantic map. 
            Discover hidden connections and explore your visual space.
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
          </div>
          
          {/* File List */}
          {uploadedFiles.length > 0 && (
            <div className="mt-8 bg-slate-800/50 backdrop-blur-xl rounded-2xl p-6 border border-blue-500/30">
              <h3 className="text-xl font-bold text-white mb-4">
                Uploaded Images ({uploadedFiles.length})
                {uploadedFiles.length < 3 && (
                  <span className="ml-2 text-sm text-yellow-400">
                    (Need {3 - uploadedFiles.length} more)
                  </span>
                )}
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-h-64 overflow-y-auto">
                {uploadedFiles.map((file, index) => (
                  <div key={index} className="relative group">
                    <div className="aspect-square bg-slate-700 rounded-lg overflow-hidden">
                      <img
                        src={URL.createObjectURL(file)}
                        alt={file.name}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="absolute top-1 right-1 bg-red-600 hover:bg-red-700 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      ×
                    </button>
                    <p className="text-xs text-gray-400 mt-1 truncate">{file.name}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Start Button */}
          {uploadedFiles.length >= 3 && (
            <div className="mt-8 text-center">
              <button
                onClick={handleStart}
                className="
                  px-12 py-4 bg-gradient-to-r from-green-600 via-green-500 to-emerald-500
                  text-white text-xl font-bold rounded-full
                  shadow-lg shadow-green-500/50 hover:shadow-xl hover:shadow-green-500/70 
                  transform hover:scale-105
                  transition-all duration-200
                "
              >
                🚀 Start Exploring
              </button>
              <p className="text-sm text-gray-400 mt-4">
                This will process your images and generate your 3D map
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
