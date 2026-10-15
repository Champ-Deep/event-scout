/** @type {import('eslint').Linter.Config} */
module.exports = {
  extends: ['./base.js', 'plugin:react/recommended', 'plugin:react-hooks/recommended'],
  plugins: ['react', 'react-hooks', 'react-native'],
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
            group: ['apps/admin/**'],
            message: 'Mobile RN app cannot import admin DOM code.',
          },
        ],
      },
    ],
  },
};
