const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

const projectRoot = __dirname;
const monorepoRoot = path.resolve(projectRoot, '../..');

const config = getDefaultConfig(projectRoot);

// 1. Watch only mobile and shared workspace packages (avoid root node_modules/.pnpm watcher collisions)
config.watchFolders = [
  projectRoot,
  path.resolve(monorepoRoot, 'packages/types'),
];

// 2. Let Metro resolve packages from both local and hoisted node_modules
config.resolver.nodeModulesPaths = [
  path.resolve(projectRoot, 'node_modules'),
  path.resolve(monorepoRoot, 'node_modules'),
];

// 3. Allow hierarchical lookup so nested pnpm dependencies resolve
config.resolver.disableHierarchicalLookup = false;

// On-device OCR models (assets/ocr/*.onnx) ship as app assets.
config.resolver.assetExts = [...config.resolver.assetExts, 'onnx'];

// 4. Force single React 18 instance across every package in the bundle
const reactPath = path.resolve(projectRoot, 'node_modules/react');
const reactDomPath = path.resolve(projectRoot, 'node_modules/react-dom');

config.resolver.extraNodeModules = {
  react: reactPath,
  'react-dom': reactDomPath,
};

config.resolver.resolveRequest = (context, moduleName, platform) => {
  if (moduleName === 'react') {
    return {
      filePath: path.resolve(reactPath, 'index.js'),
      type: 'sourceFile',
    };
  }
  if (moduleName === 'react/jsx-runtime') {
    return {
      filePath: path.resolve(reactPath, 'jsx-runtime.js'),
      type: 'sourceFile',
    };
  }
  if (moduleName === 'react/jsx-dev-runtime') {
    return {
      filePath: path.resolve(reactPath, 'jsx-dev-runtime.js'),
      type: 'sourceFile',
    };
  }
  if (moduleName === 'react-dom') {
    return {
      filePath: path.resolve(reactDomPath, 'index.js'),
      type: 'sourceFile',
    };
  }
  if (moduleName === 'react-dom/client') {
    return {
      filePath: path.resolve(reactDomPath, 'client.js'),
      type: 'sourceFile',
    };
  }
  if (moduleName.startsWith('react/')) {
    const sub = moduleName.slice('react/'.length);
    const subFile = sub.endsWith('.js') ? sub : `${sub}.js`;
    return {
      filePath: path.resolve(reactPath, subFile),
      type: 'sourceFile',
    };
  }
  if (moduleName.startsWith('react-dom/')) {
    const sub = moduleName.slice('react-dom/'.length);
    const subFile = sub.endsWith('.js') ? sub : `${sub}.js`;
    return {
      filePath: path.resolve(reactDomPath, subFile),
      type: 'sourceFile',
    };
  }

  return context.resolveRequest(context, moduleName, platform);
};

// 5. Dev only: forward /backend/* to the local API. Office networks often let the phone reach
// Metro (:8081) but not the API port, so point EXPO_PUBLIC_API_BASE_URL at http://<pc-ip>:8081/backend.
const http = require('http');
const BACKEND_PREFIX = '/backend';
const BACKEND_PORT = Number(process.env.EVENTSCOUT_BACKEND_PORT || 8000);
config.server = config.server || {};
const enhance = config.server.enhanceMiddleware;
config.server.enhanceMiddleware = (middleware, server) => {
  const base = enhance ? enhance(middleware, server) : middleware;
  return (req, res, next) => {
    if (req.url !== BACKEND_PREFIX && !req.url.startsWith(`${BACKEND_PREFIX}/`)) return base(req, res, next);
    const upstream = http.request(
      {
        host: '127.0.0.1',
        port: BACKEND_PORT,
        method: req.method,
        path: req.url.slice(BACKEND_PREFIX.length) || '/',
        // Forwarded-* let the backend build links (share pages, card QR) the phone can open:
        // http://<pc-ip>:8081/backend/s/… instead of 127.0.0.1:8000.
        headers: {
          ...req.headers,
          host: `127.0.0.1:${BACKEND_PORT}`,
          'x-forwarded-host': req.headers.host || '',
          'x-forwarded-proto': 'http',
          'x-forwarded-prefix': BACKEND_PREFIX,
        },
      },
      (up) => {
        res.writeHead(up.statusCode || 502, up.headers);
        up.pipe(res); // streams SSE replies as they arrive
      },
    );
    upstream.on('error', (err) => {
      if (res.headersSent) return res.end();
      res.writeHead(502, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ detail: `Backend not reachable on port ${BACKEND_PORT}: ${err.message}` }));
    });
    req.pipe(upstream);
  };
};

module.exports = config;
