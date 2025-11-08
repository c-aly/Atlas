/**
 * Main 3D Scene Component
 * Renders the neural network visualization with nodes and edges
 */
import { useRef, useMemo, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useStore } from '../store'

// Cluster color palette - vibrant, distinct colors
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

// Helper function to brighten a hex color
function brightenColor(hex, amount = 0.3) {
  const num = parseInt(hex.replace('#', ''), 16)
  const r = Math.min(255, ((num >> 16) & 0xff) + Math.floor(255 * amount))
  const g = Math.min(255, ((num >> 8) & 0xff) + Math.floor(255 * amount))
  const b = Math.min(255, (num & 0xff) + Math.floor(255 * amount))
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')}`
}

// Node component - represents a single image
function Node({ position, id, isSelected, isHovered, isSearchResult, cluster }) {
  const meshRef = useRef()
  const { setSelectedNode, setHoveredNode } = useStore()
  
  // Get base color from cluster
  const baseClusterColor = CLUSTER_COLORS[(cluster || 0) % CLUSTER_COLORS.length]
  
  // Determine color and size based on state
  // Priority: Selected > Search Result > Hovered > Cluster Color
  const color = isSelected 
    ? '#60a5fa'  // Bright blue when selected
    : isSearchResult 
    ? '#fbbf24'  // Yellow for search results
    : isHovered 
    ? '#93c5fd'  // Light blue on hover
    : baseClusterColor  // Cluster-specific color
  
  const scale = isSelected ? 0.3 : isHovered ? 0.25 : 0.2
  
  // Animation
  useFrame((state) => {
    if (meshRef.current) {
      // Gentle breathing animation
      const scaleVariation = 1 + Math.sin(state.clock.elapsedTime * 2 + position[0]) * 0.05
      meshRef.current.scale.setScalar(scale * scaleVariation)
      
      // Rotate slightly if selected
      if (isSelected) {
        meshRef.current.rotation.y += 0.02
      }
    }
  })
  
  return (
    <mesh
      ref={meshRef}
      position={position}
      scale={scale}
      onClick={(e) => {
        e.stopPropagation()
        setSelectedNode(id)
      }}
      onPointerOver={(e) => {
        e.stopPropagation()
        setHoveredNode(id)
        document.body.style.cursor = 'pointer'
      }}
      onPointerOut={() => {
        setHoveredNode(null)
        document.body.style.cursor = 'default'
      }}
    >
      <sphereGeometry args={[1, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={isSelected ? 0.8 : isHovered ? 0.5 : 0.3}
        roughness={0.3}
        metalness={0.8}
      />
      
      {/* Glow effect */}
      <mesh scale={1.5}>
        <sphereGeometry args={[1, 16, 16]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={isSelected ? 0.3 : isHovered ? 0.2 : 0.1}
        />
      </mesh>
    </mesh>
  )
}

// Edge component - represents similarity connection between nodes
function Edge({ start, end, weight, sourceCluster, targetCluster }) {
  const meshRef = useRef()
  
  const startVec = useMemo(() => new THREE.Vector3(...start), [start])
  const endVec = useMemo(() => new THREE.Vector3(...end), [end])
  
  // Calculate position (midpoint)
  const position = useMemo(() => {
    const midPoint = new THREE.Vector3().addVectors(startVec, endVec).multiplyScalar(0.5)
    return [midPoint.x, midPoint.y, midPoint.z]
  }, [startVec, endVec])
  
  // Calculate length
  const length = useMemo(() => {
    return startVec.distanceTo(endVec)
  }, [startVec, endVec])
  
  // Calculate rotation to align cylinder from start to end
  const rotation = useMemo(() => {
    const direction = new THREE.Vector3().subVectors(endVec, startVec)
    const len = direction.length()
    
    if (len < 0.001) {
      return [0, 0, 0]
    }
    
    direction.normalize()
    const up = new THREE.Vector3(0, 1, 0)
    const quaternion = new THREE.Quaternion().setFromUnitVectors(up, direction)
    const euler = new THREE.Euler().setFromQuaternion(quaternion)
    
    return [euler.x, euler.y, euler.z]
  }, [startVec, endVec])
  
  // Determine edge color based on clusters (blend if different clusters)
  const edgeColor = useMemo(() => {
    if (sourceCluster === targetCluster) {
      // Same cluster - use cluster color
      return CLUSTER_COLORS[(sourceCluster || 0) % CLUSTER_COLORS.length]
    } else {
      // Different clusters - use a neutral spacey color
      return '#8b5cf6' // Purple/space color
    }
  }, [sourceCluster, targetCluster])
  
  // Edge opacity and thickness
  const opacity = Math.max(0.3, Math.min(0.7, (weight || 0.5) * 0.3 + 0.3))
  const radius = Math.max(0.03, (weight || 0.5) * 0.05)
  
  return (
    <mesh ref={meshRef} position={position} rotation={rotation}>
      <cylinderGeometry args={[radius, radius, length, 8]} />
      <meshStandardMaterial
        color={edgeColor}
        emissive={edgeColor}
        emissiveIntensity={0.6}
        transparent
        opacity={opacity}
        roughness={0.1}
        metalness={0.5}
      />
    </mesh>
  )
}

// Main scene content
function SceneContent() {
  const { 
    images, 
    edges, 
    selectedNode, 
    hoveredNode, 
    searchResults
  } = useStore()
  
  // Scale factor to spread out normalized coordinates visually
  const COORD_SCALE = 15 // Multiply [-1,1] coordinates by this to spread them out
  
  const searchResultIds = useMemo(() => 
    new Set(searchResults.map(r => r.image_id)), 
    [searchResults]
  )
  
  // Filter edges to show only those connected to selected/hovered node
  // Also prioritize edges within the same cluster for better visualization
  const visibleEdges = useMemo(() => {
    if (edges.length === 0) {
      return []
    }
    
    if (!selectedNode && !hoveredNode) {
      // When nothing selected, show only edges within clusters (more meaningful)
      const imageMapForCluster = new Map()
      images.forEach(img => imageMapForCluster.set(img.id, img.cluster || 0))
      
      const clusterEdges = edges.filter(e => {
        const sourceCluster = imageMapForCluster.get(e.source) || 0
        const targetCluster = imageMapForCluster.get(e.target) || 0
        return sourceCluster === targetCluster
      })
      
      // Limit to avoid clutter - show up to 500 cluster edges
      return clusterEdges.slice(0, 500)
    }
    
    // When a node is selected/hovered, show its connections
    const focusNode = selectedNode || hoveredNode
    return edges.filter(e => 
      e.source === focusNode || e.target === focusNode
    )
  }, [edges, selectedNode, hoveredNode, images])
  
  // Get image map for edge rendering
  const imageMap = useMemo(() => {
    const map = new Map()
    images.forEach(img => map.set(img.id, img))
    return map
  }, [images])
  
  return (
    <>
      {/* Ambient lighting */}
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
      
      {/* Nodes */}
      {images.map(image => (
        <Node
          key={image.id}
          id={image.id}
          position={image.coords.map(c => c * COORD_SCALE)}
          isSelected={selectedNode === image.id}
          isHovered={hoveredNode === image.id}
          isSearchResult={searchResultIds.has(image.id)}
          cluster={image.cluster || 0}
        />
      ))}
      
      {/* Edges */}
      {visibleEdges.length > 0 && (
        <>
          {visibleEdges.map((edge, i) => {
            const sourceImg = imageMap.get(edge.source)
            const targetImg = imageMap.get(edge.target)
            
            if (!sourceImg || !targetImg) {
              console.warn(`Missing image for edge:`, edge)
              return null
            }
            
            return (
              <Edge
                key={`${edge.source}-${edge.target}-${i}`}
                start={sourceImg.coords.map(c => c * COORD_SCALE)}
                end={targetImg.coords.map(c => c * COORD_SCALE)}
                weight={edge.weight || 0.5}
                sourceCluster={sourceImg.cluster || 0}
                targetCluster={targetImg.cluster || 0}
              />
            )
          })}
        </>
      )}
      
      {/* No fog - let cosmic background show through */}
    </>
  )
}

// Camera controller that focuses on selected node
function CameraController() {
  const { camera } = useThree()
  const controlsRef = useRef()
  const { selectedNode, images } = useStore()
  const COORD_SCALE = 15
  
  // Find selected node position
  const targetPosition = useMemo(() => {
    if (!selectedNode) {
      return new THREE.Vector3(0, 0, 0) // Default to origin
    }
    
    const selectedImage = images.find(img => img.id === selectedNode)
    if (!selectedImage || !selectedImage.coords) {
      return new THREE.Vector3(0, 0, 0)
    }
    
    return new THREE.Vector3(
      selectedImage.coords[0] * COORD_SCALE,
      selectedImage.coords[1] * COORD_SCALE,
      selectedImage.coords[2] * COORD_SCALE
    )
  }, [selectedNode, images])
  
  // Update controls target when selection changes
  useEffect(() => {
    if (controlsRef.current) {
      // Smoothly transition to new target
      const currentTarget = controlsRef.current.target.clone()
      const newTarget = targetPosition.clone()
      
      // Animate the transition
      const duration = 500 // ms
      const startTime = Date.now()
      const startTarget = currentTarget.clone()
      
      const animate = () => {
        const elapsed = Date.now() - startTime
        const progress = Math.min(elapsed / duration, 1)
        const easeProgress = 1 - Math.pow(1 - progress, 3) // Ease out cubic
        
        currentTarget.lerpVectors(startTarget, newTarget, easeProgress)
        controlsRef.current.target.copy(currentTarget)
        
        if (progress < 1) {
          requestAnimationFrame(animate)
        } else {
          controlsRef.current.target.copy(newTarget)
        }
      }
      
      animate()
    }
  }, [targetPosition])
  
  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping
      dampingFactor={0.05}
      rotateSpeed={0.5}
      zoomSpeed={0.8}
      minDistance={5}
      maxDistance={200} // Allow zooming out further for large datasets
    />
  )
}

// Main Scene3D component
export default function Scene3D() {
  return (
    <Canvas
      style={{ width: '100%', height: '100%', position: 'relative', zIndex: 0 }}
      gl={{ 
        antialias: true, 
        alpha: true,
        powerPreference: 'high-performance'
      }}
    >
      <PerspectiveCamera makeDefault position={[0, 0, 30]} fov={60} />
      
      <CameraController />
      
      <SceneContent />
    </Canvas>
  )
}
