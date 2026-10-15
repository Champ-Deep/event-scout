import http from 'http';
import fs from 'fs';

http.get('http://127.0.0.1:9222/json', (res) => {
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    const tabs = JSON.parse(data);
    const target = tabs.find(t => t.url.includes('3000'));
    if (target && target.webSocketDebuggerUrl) {
      const ws = new WebSocket(target.webSocketDebuggerUrl);
      ws.onopen = () => {
        // Just capture immediately
        ws.send(JSON.stringify({
          id: 1,
          method: 'Page.captureScreenshot',
          params: { format: 'png' }
        }));
      };
      ws.onmessage = (msg) => {
        const res = JSON.parse(msg.data);
        if (res.id === 1) {
          const buf = Buffer.from(res.result.data, 'base64');
          fs.writeFileSync('C:/Users/harsha.c/.gemini/antigravity-ide/brain/dcf61721-0fb4-42d7-863e-6a02481ec2f6/live_contacts.png', buf);
          console.log('Saved live_contacts.png');
          process.exit(0);
        }
      };
    }
  });
});
