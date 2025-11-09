/**
 * Auto-Play Button Component
 * Persistent button at bottom center that shuffles through nodes and plays audio
 */
import { useEffect, useRef } from 'react'
import { useStore } from '../store'
import { getNarrationUrl } from '../api'
import { supabase } from '../lib/supabase'

export default function AutoPlayButton() {
  const { 
    images, 
    selectedNode, 
    setSelectedNode,
    isAutoPlaying,
    setIsAutoPlaying,
    getImageById
  } = useStore()
  
  const playedNodesRef = useRef(new Set())
  const audioRef = useRef(null)
  const timeoutRef = useRef(null)
  
  // Helper to get current isAutoPlaying state
  const getIsAutoPlaying = () => useStore.getState().isAutoPlaying
  
  // Get random unplayed node
  const getRandomUnplayedNode = () => {
    if (images.length === 0) return null
    
    const unplayedNodes = images.filter(img => !playedNodesRef.current.has(img.id))
    
    if (unplayedNodes.length === 0) {
      // All nodes played, reset and return random from all
      playedNodesRef.current.clear()
      const randomIndex = Math.floor(Math.random() * images.length)
      return images[randomIndex].id
    }
    
    // Return random unplayed node
    const randomIndex = Math.floor(Math.random() * unplayedNodes.length)
    return unplayedNodes[randomIndex].id
  }
  
  // Play audio for a node
  const playAudioForNode = async (nodeId) => {
    const image = getImageById(nodeId)
    if (!image) return false
    
    // Get audio URL
    const audioUrl = getNarrationUrl(nodeId)
    
    try {
      // Stop any currently playing audio
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current = null
      }
      
      // Fetch audio with auth token
      const { data: { session } } = await supabase.auth.getSession()
      
      const response = await fetch(audioUrl, {
        headers: {
          'Authorization': `Bearer ${session?.access_token || ''}`
        }
      })
      
      if (!response.ok) {
        console.warn(`Failed to fetch audio for ${nodeId}: ${response.status}`)
        return false
      }
      
      const blob = await response.blob()
      const audioObjectUrl = URL.createObjectURL(blob)
      
      const audio = new Audio(audioObjectUrl)
      audioRef.current = audio
      
      return new Promise((resolve) => {
        audio.onended = () => {
          URL.revokeObjectURL(audioObjectUrl)
          audioRef.current = null
          playedNodesRef.current.add(nodeId)
          console.log('Auto-play: Audio finished for', nodeId)
          // If auto-playing, move to next node
          if (getIsAutoPlaying()) {
            timeoutRef.current = setTimeout(() => {
              if (getIsAutoPlaying()) {
                playNextNode()
              }
            }, 800)
          }
          resolve(true)
        }
        
        audio.onerror = () => {
          console.error(`Error playing audio for ${nodeId}`)
          URL.revokeObjectURL(audioObjectUrl)
          audioRef.current = null
          playedNodesRef.current.add(nodeId) // Mark as played even if error
          resolve(false)
        }
        
        audio.play().catch(error => {
          console.error('Error playing audio:', error)
          URL.revokeObjectURL(audioObjectUrl)
          audioRef.current = null
          playedNodesRef.current.add(nodeId)
          resolve(false)
        })
      })
    } catch (error) {
      console.error(`Error loading audio for ${nodeId}:`, error)
      playedNodesRef.current.add(nodeId)
      return false
    }
  }
  
  // Play next node in shuffle
  const playNextNode = async () => {
    if (!getIsAutoPlaying()) {
      return
    }
    
    const nextNodeId = getRandomUnplayedNode()
    if (!nextNodeId) {
      console.log('All nodes have been played, resetting...')
      setIsAutoPlaying(false)
      playedNodesRef.current.clear()
      return
    }
    
    console.log('Auto-play: Selecting node', nextNodeId)
    // Select the node (this will trigger description generation in NodeDetailPanel)
    setSelectedNode(nextNodeId)
    
    // Wait for description to be generated and then play audio
    // NodeDetailPanel will generate description automatically when node is selected
    // We'll wait a reasonable amount of time (3-5 seconds for generation + 1 second buffer)
    await new Promise(resolve => setTimeout(resolve, 4000))
    
    if (!getIsAutoPlaying()) {
      return
    }
    
    // Try to play audio - it will work if description was generated
    console.log('Auto-play: Attempting to play audio for', nextNodeId)
    const played = await playAudioForNode(nextNodeId)
    
    if (!getIsAutoPlaying()) {
      return
    }
    
    if (played) {
      // Audio played successfully, wait for it to finish (handled in playAudioForNode)
      // The onended handler will call playNextNode again
      console.log('Auto-play: Audio playing for', nextNodeId)
    } else {
      // Audio failed (probably no description yet), skip to next node
      console.log('Auto-play: Audio failed for', nextNodeId, ', skipping to next')
      playedNodesRef.current.add(nextNodeId)
      timeoutRef.current = setTimeout(() => {
        if (getIsAutoPlaying()) {
          playNextNode()
        }
      }, 1000)
    }
  }
  
  // Handle auto-play toggle
  const handleToggleAutoPlay = () => {
    console.log('Auto-play button clicked, current state:', isAutoPlaying)
    if (isAutoPlaying) {
      // Stop auto-play
      console.log('Stopping auto-play')
      setIsAutoPlaying(false)
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current = null
      }
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
    } else {
      // Start auto-play
      console.log('Starting auto-play, images count:', images.length)
      setIsAutoPlaying(true)
      playedNodesRef.current.clear()
      // If current node is selected, mark it as played
      if (selectedNode) {
        playedNodesRef.current.add(selectedNode)
        console.log('Marked current node as played:', selectedNode)
      }
      // Start playing
      console.log('Calling playNextNode...')
      playNextNode()
    }
  }
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current = null
      }
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])
  
  // Don't show if no images
  if (images.length === 0) {
    console.log('AutoPlayButton: No images, not rendering')
    return null
  }
  
  console.log('AutoPlayButton: Rendering button, images:', images.length, 'isAutoPlaying:', isAutoPlaying)
  
  return (
    <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-20 pointer-events-auto">
      <button
        onClick={(e) => {
          e.preventDefault()
          e.stopPropagation()
          console.log('Button clicked!')
          handleToggleAutoPlay()
        }}
        className={`
          px-6 py-3 rounded-full font-semibold text-white
          transition-all duration-200 shadow-lg cursor-pointer
          flex items-center space-x-2
          ${isAutoPlaying 
            ? 'bg-purple-600 hover:bg-purple-700 animate-pulse shadow-purple-500/50' 
            : 'bg-purple-700 hover:bg-purple-600 shadow-purple-500/30'
          }
        `}
        title={isAutoPlaying ? "Stop auto-play tour" : "Start auto-play tour (shuffle through all images)"}
      >
        <span className="text-xl">{isAutoPlaying ? '⏸' : '🔀'}</span>
        <span>{isAutoPlaying ? 'Auto-Play Active' : 'Start Auto-Play'}</span>
      </button>
    </div>
  )
}

