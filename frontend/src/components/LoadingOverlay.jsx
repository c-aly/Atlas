/**
 * Loading Overlay Component
 * Full-screen loading indicator
 */

export default function LoadingOverlay({ message = 'Processing...' }) {
  return (
    <div className="fixed inset-0 z-50 bg-neural-bg/90 backdrop-blur-sm flex items-center justify-center">
      <div className="text-center">
        <div className="inline-block">
          <div className="text-6xl mb-4 animate-pulse-glow">🧭</div>
        </div>
        <p className="text-xl text-white font-semibold mb-2">{message}</p>
        <div className="flex items-center justify-center space-x-2">
          <div className="w-2 h-2 bg-neural-accent rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-neural-accent rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-2 h-2 bg-neural-accent rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    </div>
  )
}

