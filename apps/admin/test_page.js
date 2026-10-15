const http = require('http');

http.get('http://localhost:3000/seats', (res) => {
  let body = '';
  res.on('data', chunk => body += chunk);
  res.on('end', () => {
    console.log('Status:', res.statusCode);
    console.log('HTML:\n', body);
  });
});
