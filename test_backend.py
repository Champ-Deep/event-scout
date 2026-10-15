import requests

try:
    resp = requests.post(
        'http://127.0.0.1:8000/add_contact_from_image/?user_id=mock-user-id',
        headers={'X-API-Key': '1234'},
        files={'file': ('test.txt', b'hello world'*200)}
    )
    print(resp.status_code)
    print(resp.text)
except Exception as e:
    print(e)
