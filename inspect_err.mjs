import http from 'http';

http.get('http://127.0.0.1:9222/json', (res) => {
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    const tabs = JSON.parse(data);
    const target = tabs.find(t => t.url.includes('3000'));
    if (target && target.webSocketDebuggerUrl) {
      const ws = new WebSocket(target.webSocketDebuggerUrl);
      ws.onopen = () => {
        ws.send(JSON.stringify({ id: 0, method: 'Console.enable' }));
        ws.send(JSON.stringify({ id: 0, method: 'Runtime.enable' }));
        ws.send(JSON.stringify({
          id: 1,
          method: 'Runtime.evaluate',
          params: {
            expression: `(async () => {
              return new Promise((resolve) => {
                const img = new Image();
                img.onload = () => resolve({ loaded: true, width: img.width, height: img.height });
                img.onerror = (e) => resolve({ loaded: false });
                img.src = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 80 80' width='80' height='80'><defs><linearGradient id='g1' x1='0%' y1='0%' x2='100%' y2='100%'><stop offset='0%' stop-color='%23059669'/><stop offset='100%' stop-color='%2334d399'/></linearGradient></defs><rect width='80' height='80' rx='40' fill='url(%23g1)'/><circle cx='40' cy='32' r='14' fill='white' fill-opacity='0.9'/><path d='M18 70 C18 53, 30 48, 40 48 C50 48, 62 53, 62 70 Z' fill='white' fill-opacity='0.9'/><text x='40' y='36' text-anchor='middle' font-family='sans-serif' font-size='10' font-weight='bold' fill='%23064e3b'>AM</text></svg>";
              });
            })()`,
            awaitPromise: true,
            returnByValue: true
          }
        }));
      };
      ws.onmessage = (msg) => {
        const res = JSON.parse(msg.data);
        if (res.method === 'Console.messageAdded') {
          console.log('CONSOLE:', res.params?.message?.text);
        }
        if (res.method === 'Runtime.exceptionThrown') {
          console.log('EXCEPTION:', JSON.stringify(res.params?.exceptionDetails));
        }
        if (res.id === 1) {
          console.log('PAGE:', JSON.stringify(res.result?.result?.value, null, 2));
          setTimeout(() => process.exit(0), 1000);
        }
      };
    }
  });
});
