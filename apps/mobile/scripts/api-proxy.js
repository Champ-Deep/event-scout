// Dev-only TCP forwarder: LAN :8002 -> backend 127.0.0.1:8000.
// Use when Windows Firewall blocks python.exe inbound but allows node.exe.
const net = require('net');

const LISTEN_PORT = Number(process.env.PROXY_PORT || 8002);
const TARGET_PORT = Number(process.env.TARGET_PORT || 8000);

net
  .createServer((client) => {
    const upstream = net.connect(TARGET_PORT, '127.0.0.1');
    client.pipe(upstream).pipe(client);
    const close = () => { client.destroy(); upstream.destroy(); };
    client.on('error', close);
    upstream.on('error', close);
    console.log(`[proxy] ${client.remoteAddress} -> :${TARGET_PORT}`);
  })
  .listen(LISTEN_PORT, '0.0.0.0', () => {
    console.log(`[proxy] 0.0.0.0:${LISTEN_PORT} -> 127.0.0.1:${TARGET_PORT}`);
  });
