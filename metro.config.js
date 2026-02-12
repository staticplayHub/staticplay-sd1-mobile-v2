// eslint-disable-next-line @typescript-eslint/no-var-requires
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

config.resolver.assetExts = Array.from(new Set([...(config.resolver.assetExts || []), 'txt']));

module.exports = config;

