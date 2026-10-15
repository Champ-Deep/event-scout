// Stands in for react-native / expo modules in unit tests (see vitest.config.ts). The modules under
// test must not need them: touching anything here fails the test with a clear message.
const fail = (): never => {
  throw new Error('A unit-tested module reached a native (react-native / expo) module — pass an adapter instead.');
};

export const Platform = { OS: 'test', select: fail };
export default new Proxy({}, { get: fail });
