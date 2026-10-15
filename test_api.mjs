import http from 'http';

http.get('http://127.0.0.1:8000/api/contacts', (res) => {
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    try {
      const json = JSON.parse(data);
      console.log('First contact avatarUrl:', json.contacts?.[0]?.avatarUrl);
      console.log('First contact photoUrl:', json.contacts?.[0]?.photoUrl);
      console.log('Full first contact:', JSON.stringify(json.contacts?.[0], null, 2));
    } catch (e) {
      console.error(e.message, data);
    }
  });
});
