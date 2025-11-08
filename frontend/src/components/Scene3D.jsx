/**
 * Main 3D Scene Component
 * Renders the neural network visualization with nodes and edges
 * Supports both third-person orbital view and first-person navigation
 */
import { useRef, useMemo, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useStore } from '../store'

// Node component - represents a single image
function Node({ position, id, isSelected, isHovered, isSearchResult, isCurrentFP }) {
  const meshRef = useRef()
  const { setSelectedNode, setHoveredNode, isFirstPerson, navigateToNode, enterFirstPerson } = useStore()
  
  // Animation
  useFrame((state) => {
    if (meshRef.current) {
      // Gentle breathing animation
      const scale = 1 + Math.sin(state.clock.elapsedTime * 2 + position[0]) * 0.05
      meshRef.current.scale.setScalar(scale)
      
      // Rotate slightly if selected
      if (isSelected) {
        meshRef.current.rotation.y += 0.02
      }
    }
  })
  
  // Determine color and size based on state
  const color = isCurrentFP
    ? '#10b981'  // Green for current first-person node
    : isSelected 
    ? '#60a5fa'  // Bright blue when selected
    : isSearchResult 
    ? '#fbbf24'  // Yellow for search results
    : isHovered 
    ? '#93c5fd'  // Light blue on hover
    : '#3b82f6'  // Default blue
  
  const scale = isCurrentFP ? 0.4 : isSelected ? 0.3 : isHovered ? 0.25 : 0.2
  
  // Don't render the current first-person node (we're inside it)
  if (isCurrentFP && isFirstPerson) {
    return null
  }
  
  return (
    <mesh
      ref={meshRef}
      position={position}
      scale={scale}
      onClick={(e) => {
        e.stopPropagation()
        if (isFirstPerson) {
          navigateToNode(id)
        } else {
          setSelectedNode(id)
        }
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
function Edge({ start, end, weight }) {
  const lineRef = useRef()
  
  const points = useMemo(() => {
    return [
      new THREE.Vector3(...start),
      new THREE.Vector3(...end)
    ]
  }, [start, end])
  
  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry().setFromPoints(points)
    return geom
  }, [points])
  
  // Edge opacity based on similarity weight
  const opacity = Math.max(0.1, weight * 0.5)
  
  return (
    <line ref={lineRef} geometry={geometry}>
      <lineBasicMaterial
        color="#60a5fa"
        transparent
        opacity={opacity}
        linewidth={1}
      />
    </line>
  )
}

// First-person camera controller with free look
function FirstPersonCamera({ targetPosition }) {
  const { camera, gl } = useThree()
  const targetRef = useRef(new THREE.Vector3(...targetPosition))
  const rotationRef = useRef({ x: 0, y: 0 })
  const isDragging = useRef(false)
  const previousMouse = useRef({ x: 0, y: 0 })
  
  useEffect(() => {
    targetRef.current.set(...targetPosition)
  }, [targetPosition])
  
  // Mouse controls for looking around
  useEffect(() => {
    const canvas = gl.domElement
    
    const handleMouseDown = (e) => {
      isDragging.current = true
      previousMouse.current = { x: e.clientX, y: e.clientY }
      canvas.style.cursor = 'grabbing'
    }
    
    const handleMouseMove = (e) => {
      if (!isDragging.current) return
      
      const deltaX = e.clientX - previousMouse.current.x
      const deltaY = e.clientY - previousMouse.current.y
      
      // Update rotation (horizontal and vertical look)
      rotationRef.current.y -= deltaX * 0.003 // Horizontal rotation (yaw)
      rotationRef.current.x -= deltaY * 0.003 // Vertical rotation (pitch)
      
      // Clamp vertical rotation to prevent flipping
      rotationRef.current.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotationRef.current.x))
      
      previousMouse.current = { x: e.clientX, y: e.clientY }
    }
    
    const handleMouseUp = () => {
      isDragging.current = false
      canvas.style.cursor = 'grab'
    }
    
    canvas.addEventListener('mousedown', handleMouseDown)
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    canvas.style.cursor = 'grab'
    
    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown)
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
      canvas.style.cursor = 'default'
    }
  }, [gl])
  
  useFrame(() => {
    // Smoothly interpolate camera position to target
    camera.position.lerp(targetRef.current, 0.1)
    
    // Apply rotation for free look
    const lookDirection = new THREE.Vector3(
      Math.sin(rotationRef.current.y) * Math.cos(rotationRef.current.x),
      Math.sin(rotationRef.current.x),
      Math.cos(rotationRef.current.y) * Math.cos(rotationRef.current.x)
    )
    
    const lookAt = new THREE.Vector3()
    lookAt.addVectors(camera.position, lookDirection.multiplyScalar(10))
    camera.lookAt(lookAt)
  })
  
  return null
}

// Main scene content
function SceneContent() {
  const { 
    images, 
    edges, 
    selectedNode, 
    hoveredNode, 
    searchResults,
    isFirstPerson,
    firstPersonNode,
    getImageById
  } = useStore()
  
  const searchResultIds = useMemo(() => 
    new Set(searchResults.map(r => r.image_id)), 
    [searchResults]
  )
  
  // Filter edges to show only those connected to selected/hovered node
  const visibleEdges = useMemo(() => {
    if (!selectedNode && !hoveredNode) {
      // Show a subset of edges to avoid clutter
      return edges.filter((_, i) => i % 5 === 0).slice(0, 200)
    }
    
    const focusNode = selectedNode || hoveredNode
    return edges.filter(e => 
      e.source === focusNode || e.target === focusNode
    )
  }, [edges, selectedNode, hoveredNode])
  
  // Get image map for edge rendering
  const imageMap = useMemo(() => {
    const map = new Map()
    images.forEach(img => map.set(img.id, img))
    return map
  }, [images])
  
  // Get current first-person position
  const fpPosition = useMemo(() => {
    if (isFirstPerson && firstPersonNode) {
      const img = getImageById(firstPersonNode)
      return img ? img.coords : [0, 0, 30]
    }
    return [0, 0, 30]
  }, [isFirstPerson, firstPersonNode, getImageById])
  
  return (
    <>
      {/* Ambient lighting */}
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
      
      {/* First-person camera controller */}
      {isFirstPerson && <FirstPersonCamera targetPosition={fpPosition} />}
      
      {/* Nodes */}
      {images.map(image => (
        <Node
          key={image.id}
          id={image.id}
          position={image.coords}
          isSelected={selectedNode === image.id}
          isHovered={hoveredNode === image.id}
          isSearchResult={searchResultIds.has(image.id)}
          isCurrentFP={firstPersonNode === image.id}
        />
      ))}
      
      {/* Edges - show connections from current node in first-person */}
      {visibleEdges.map((edge, i) => {
        const sourceImg = imageMap.get(edge.source)
        const targetImg = imageMap.get(edge.target)
        
        if (!sourceImg || !targetImg) return null
        
        return (
          <Edge
            key={`${edge.source}-${edge.target}-${i}`}
            start={sourceImg.coords}
            end={targetImg.coords}
            weight={edge.weight}
          />
        )
      })}
      
      {/* Background fog */}
      <fog attach="fog" args={['#0a0e1a', 10, 50]} />
    </>
  )
}

// Main Scene3D component
export default function Scene3D() {
  const { isFirstPerson } = useStore()
  
  return (
    <Canvas
      style={{ width: '100%', height: '100%' }}
      gl={{ 
        antialias: true, 
        alpha: false,
        powerPreference: 'high-performance'
      }}
    >
      <PerspectiveCamera makeDefault position={[0, 0, 30]} fov={60} />
      
      {/* Only show orbit controls in third-person mode */}
      {!isFirstPerson && (
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          rotateSpeed={0.5}
          zoomSpeed={0.8}
          minDistance={5}
          maxDistance={100}
        />
      )}
      
      <SceneContent />
    </Canvas>
  )
}
