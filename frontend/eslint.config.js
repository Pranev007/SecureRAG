// ESLint flat config (ESLint 9).
//
// TypeScript already catches type errors via `npm run typecheck`, so this
// deliberately does not duplicate that. It covers what the compiler cannot:
// React Hook rules, unused code, and a small set of correctness patterns.

import js from '@eslint/js';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist', 'node_modules', 'eslint.config.js', 'vite.config.ts'] },
  {
    files: ['**/*.{ts,tsx}'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,

      // Fast Refresh only works when a module exports components alone. The
      // shared `ui.tsx` intentionally also exports helpers, so this warns
      // rather than errors.
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],

      // Underscore-prefixed arguments are a deliberate "unused on purpose"
      // marker used throughout the codebase.
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],

      // The API client deals in `unknown` from `fetch`, and narrowing it is the
      // point; an explicit `any` is not allowed to creep in instead.
      '@typescript-eslint/no-explicit-any': 'error',
    },
  },
);
