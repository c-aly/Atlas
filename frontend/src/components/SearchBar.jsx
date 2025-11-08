/**
 * Search Bar Component
 * Natural language search functionality
 */
import { useState } from 'react'
import { searchByText } from '../api'
import { useStore } from '../store'

export default function SearchBar() {
  const [query, setQuery] = useState('')
  const [isSearching, setIsSearching] = useState(false)
  const { setSearchResults, searchResults, setSelectedNode } = useStore()
  
  const handleSearch = async (e) => {
    e.preventDefault()
    
    if (!query.trim()) {
      setSearchResults([])
      return
    }
    
    try {
      setIsSearching(true)
      const results = await searchByText(query, 10)
      setSearchResults(results.results)
      console.log('Search results:', results)
    } catch (error) {
      console.error('Search error:', error)
      alert('Search failed: ' + error.message)
    } finally {
      setIsSearching(false)
    }
  }
  
  const clearSearch = () => {
    setQuery('')
    setSearchResults([])
  }
  
  return (
    <div className="bg-neural-card rounded-lg p-4 shadow-xl border border-gray-700">
      <h3 className="text-lg font-semibold mb-3 text-white flex items-center">
        <span className="mr-2">🔍</span>
        Search
      </h3>
      
      <form onSubmit={handleSearch} className="space-y-3">
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., cozy desk with coffee..."
            className="
              w-full px-4 py-2 bg-neural-bg border border-gray-600 
              rounded-lg text-white placeholder-gray-500
              focus:outline-none focus:ring-2 focus:ring-neural-accent
              transition-all duration-200
            "
          />
          
          {query && (
            <button
              type="button"
              onClick={clearSearch}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-white"
            >
              ✕
            </button>
          )}
        </div>
        
        <button
          type="submit"
          disabled={isSearching || !query.trim()}
          className="
            w-full py-2 bg-neural-accent hover:bg-neural-glow
            text-white font-medium rounded-lg
            transition-colors duration-200
            disabled:opacity-50 disabled:cursor-not-allowed
          "
        >
          {isSearching ? 'Searching...' : 'Search'}
        </button>
      </form>
      
      {searchResults.length > 0 && (
        <div className="mt-3 p-3 bg-neural-bg rounded-lg border border-yellow-500/20">
          <p className="text-sm text-gray-300 mb-2">
            Found {searchResults.length} similar images
          </p>
          <div className="flex flex-wrap gap-1.5">
            {searchResults.slice(0, 5).map((result, i) => (
              <button
                key={result.image_id}
                onClick={() => setSelectedNode(result.image_id)}
                className="
                  px-2 py-1 bg-yellow-500/20 text-yellow-300 text-xs rounded
                  hover:bg-yellow-500/30 hover:text-yellow-200
                  transition-all duration-200 cursor-pointer
                  border border-yellow-500/0 hover:border-yellow-500/50
                  transform hover:scale-105
                "
                title={`Click to view (${(result.similarity * 100).toFixed(0)}% match)`}
              >
                #{i + 1}: {(result.similarity * 100).toFixed(0)}% match
              </button>
            ))}
          </div>
        </div>
      )}
      
      <div className="mt-3 text-xs text-gray-500">
        <p>💡 Try: "sunset over mountains", "indoor plant", "laptop workspace"</p>
      </div>
    </div>
  )
}

