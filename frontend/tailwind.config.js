/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      // Tailwind's default opacity scale jumps 5 -> 10 -> 20. The security
      // surfaces need tints subtler than /10 for panel washes that must not
      // compete with text, so the intermediate steps are added here rather
      // than approximated with arbitrary-value syntax at every call site.
      opacity: { 6: '0.06', 8: '0.08', 12: '0.12', 15: '0.15', 35: '0.35', 45: '0.45' },
      colors: {
        ink: {
          950: '#0a0c10', 900: '#0f1218', 850: '#141821',
          800: '#1a1f2b', 700: '#252b3a', 600: '#343c4f',
          500: '#4a5468', 400: '#6b7689', 300: '#98a2b3',
          200: '#c7cdd9', 100: '#e6e9ef',
        },
        accent: { DEFAULT: '#5b8def', dim: '#3d6fd4', glow: '#7ea6ff' },
        safe: '#2ea56b',
        warn: '#d99a2b',
        danger: '#e05252',
        critical: '#b5379a',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.25s ease-out',
        'slide-up': 'slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
        'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: { '0%': { opacity: '0' }, '100%': { opacity: '1' } },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSoft: { '0%,100%': { opacity: '1' }, '50%': { opacity: '0.45' } },
      },
    },
  },
  plugins: [],
};
