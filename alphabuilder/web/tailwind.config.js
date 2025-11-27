/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // The Void (60%)
        void: '#050505',
        deep: '#0A0A0A',
        
        // The Structure (30%)
        matter: '#121212',
        steel: '#2A2A2A',
        text: '#E0E0E0',
        
        // The Energy (10%)
        cyan: '#00F0FF',
        magenta: '#FF0055',
        purple: '#7000FF',
        green: '#00FF9D',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ['Space Grotesk', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
