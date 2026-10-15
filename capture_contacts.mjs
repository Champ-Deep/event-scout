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
        ws.send(JSON.stringify({
          id: 1,
          method: 'Page.navigate',
          params: { url: 'http://localhost:3000/contacts' }
        }));
      };
      ws.onmessage = (msg) => {
        const res = JSON.parse(msg.data);
        if (res.id === 1) {
          setTimeout(() => {
            ws.send(JSON.stringify({
              id: 2,
              method: 'Page.captureScreenshot',
              params: { format: 'png' }
            }));
          }, 1500);
        }
        if (res.id === 2) {
          const buf = Buffer.from(res.result.data, 'base64');
          fs.writeFileSync('C:/Users/harsha.c/.gemini/antigravity-ide/brain/dcf61721-0fb4-42d7-863e-6a02481ec2f6/contacts_with_avatars.png', buf);
          console.log('Saved contacts_with_avatars.png');
          process.exit(0);
        }
      };
    }
  });
});
