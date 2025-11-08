/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        neural: {
          bg: '#0a0e1a',
          card: '#1a1f2e',
          accent: '#3b82f6',
          glow: '#60a5fa',
        }
      }
    },
  },
  plugins: [],
}

