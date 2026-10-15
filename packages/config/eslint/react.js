/** @type {import('eslint').Linter.Config} */
module.exports = {
  extends: ['./base.js', 'plugin:react/recommended', 'plugin:react-hooks/recommended'],
  plugins: ['react', 'react-hooks'],
  settings: {
    react: {
      version: 'detect',
    },
  },
  rules: {
    'react/react-in-jsx-scope': 'off',
    'no-restricted-imports': [
      'error',
      {
        patterns: [
          {
            group: ['**/features/*/**', '!**/features/*/index', '!**/features/*/index.ts'],
            message: 'Cross-feature imports must only import from the feature public index.ts.',
          },
          {
            group: ['apps/mobile/**'],
            message: 'Admin DOM app cannot import mobile code.',
          },
        ],
      },
    ],
  },
};
