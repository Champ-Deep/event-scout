const os = require('os');
const { spawn } = require('child_process');

function getLanIp() {
  const nets = os.networkInterfaces();
  const candidates = [];

  for (const name of Object.keys(nets)) {
    const isVirtual = /vEthernet|VirtualBox|VMware|WSL|Loopback/i.test(name);
    for (const net of nets[name] || []) {
      const isIPv4 = net.family === 'IPv4' || net.family === 4;
      if (isIPv4 && !net.internal && !net.address.startsWith('127.')) {
        candidates.push({ name, address: net.address, isVirtual });
      }
    }
  }

  // 1. Prefer physical Wi-Fi or Ethernet
  const preferred = candidates.find(c => !c.isVirtual && /wi-fi|wireless|wlan|ethernet/i.test(c.name));
  if (preferred) return preferred.address;

  // 2. Prefer any non-virtual
  const nonVirtual = candidates.find(c => !c.isVirtual);
  if (nonVirtual) return nonVirtual.address;

  // 3. Any found candidate
  if (candidates.length > 0) return candidates[0].address;

  return null;
}

const isTunnel = process.argv.includes('--tunnel');

if (isTunnel) {
  console.log(`[EventScout] Starting in Tunnel mode (ngrok)...`);
} else {
  const lanIp = getLanIp();
  if (lanIp) {
    console.log(`[EventScout] Detected local Wi-Fi / LAN IP: ${lanIp}`);
    process.env.REACT_NATIVE_PACKAGER_HOSTNAME = lanIp;
  } else {
    console.log(`[EventScout] Could not automatically detect LAN IP, falling back to Expo default.`);
  }
}

// expo-dev-client is installed (for the EAS development build), which makes `expo start` default
// to dev-build mode — its QR code only opens that custom app, so Expo Go can't use it.
// Default to Expo Go; pass --dev-client to target the EAS development build instead.
const args = process.argv.slice(2);
const wantsDevClient = args.includes('--dev-client');
if (!wantsDevClient && !args.includes('--go')) args.push('--go');
console.log(`[EventScout] Target: ${wantsDevClient ? 'EAS development build (--dev-client)' : 'Expo Go (--go)'}`);

const isWindows = process.platform === 'win32';
const expoCmd = isWindows ? 'npx.cmd' : 'npx';
const child = spawn(expoCmd, ['expo', 'start', ...args], {
  stdio: 'inherit',
  env: process.env,
  shell: true,
});

child.on('exit', (code) => {
  process.exit(code ?? 0);
});
