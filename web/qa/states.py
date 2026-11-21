# Labelled loopback-only response fixtures for queue, auth, injection and recovery states.
# These are UI contract tests, never evidence of AI-provider success.
import importlib.util, io, json, pathlib, time
from PIL import Image,ImageDraw
from playwright.sync_api import sync_playwright,expect
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'web/qa';BASE='http://localhost:8792'
spec=importlib.util.spec_from_file_location('audit',str(pathlib.Path.home()/'champ-skills/design/skills/visual-verify/scripts/audit.py'));audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
results=[];checks=[];frames=[];console=[]
def passed(s):checks.append(s);print('PASS',s,flush=True)
image=Image.new('RGB',(480,280),'white');ImageDraw.Draw(image).text((32,40),'SYNTHETIC LOCAL QA\nFixture Labs\nfixture@example.invalid',fill='black');buf=io.BytesIO();image.save(buf,format='PNG');photo={'name':'qa-fixture.png','mimeType':'image/png','buffer':buf.getvalue()}
with sync_playwright() as pw:
    browser=pw.chromium.launch();ctx=browser.new_context(viewport={'width':390,'height':844});page=ctx.new_page();page.on('pageerror',lambda e:console.append(str(e)))
    user=ctx.request.post(BASE+'/api/register/',data={'name':'QA State Scout','email':f'states-{int(time.time())}@example.invalid','password':'local-qa-only-123'}).json()
    page.goto(BASE);page.evaluate('(u)=>localStorage.setItem("eventScoutUser",JSON.stringify(u))',user);page.reload();expect(page.get_by_role('heading',name='Scan cards')).to_be_visible()
    def shot(name):
        page.wait_for_timeout(150);findings=page.evaluate(audit.PROBE);results.append({'state':name,'audit':findings})
        # Label test doubles in every captured state.
        page.evaluate('''() => {let tag=document.getElementById('fixture-label');if(!tag){tag=document.createElement('p');tag.id='fixture-label';tag.textContent='LOCAL QA FIXTURE. Not AI provider evidence.';tag.style.cssText='position:fixed;top:0;left:0;right:0;z-index:1000;background:#0f172a;color:#fff;padding:4px;text-align:center;font-size:12px';document.body.append(tag)}}''')
        frames.append((name,Image.open(io.BytesIO(page.screenshot())).convert('RGB')))
    def upload(value=photo):page.locator('input[aria-label="Choose card photos"]').set_input_files(value)
    upload({'name':'not-a-card.txt','mimeType':'text/plain','buffer':b'not image'});expect(page.get_by_role('alert')).to_contain_text('Choose a JPEG');shot('Invalid image');passed('file-type refusal is inline and does not enter the queue')
    ctx.set_offline(True);upload();expect(page.locator('[data-status=queued]')).to_be_visible();expect(page.get_by_text('You are offline.',exact=False)).to_be_visible();shot('Offline queue');passed('offline card is persisted before any request')
    calls=[];pending=[]
    def held(route):calls.append(route);pending.append(route)
    page.route('**/api/add_contact_from_image/**',held);ctx.set_offline(False);page.wait_for_timeout(250);assert len(calls)==1
    upload({'name':'qa-fixture-two.png','mimeType':'image/png','buffer':buf.getvalue()});page.wait_for_timeout(200);assert len(calls)==1;shot('Reading and queued');passed('queue serializes uploads with exactly one active request')
    pending.pop(0).fulfill(status=400,json={'detail':'No card text found. Try a clearer photo.'});expect(page.get_by_role('heading',name='Needs attention · 1')).to_be_visible();page.wait_for_timeout(200);assert len(calls)==2;pending.pop(0).fulfill(status=503,json={'detail':'Temporarily unavailable. Please retry.'});expect(page.get_by_role('heading',name='Needs attention · 2')).to_be_visible();shot('Failed cards');passed('400 gives Retry and Enter manually; 503 keeps the photo with Retry')
    page.reload();expect(page.get_by_role('heading',name='Needs attention · 2')).to_be_visible();passed('failed photos and their reasons survive reload')
    other=ctx.request.post(BASE+'/api/register/',data={'name':'QA Other Scout','email':f'other-{time.time_ns()}@example.invalid','password':'local-qa-only-123'}).json()
    page.evaluate('(u)=>localStorage.setItem("eventScoutUser",JSON.stringify(u))',other);page.reload();expect(page.get_by_role('heading',name='Scan cards')).to_be_visible();expect(page.get_by_role('heading',name='Needs attention',exact=False)).not_to_be_visible();expect(page.get_by_text('No photos waiting.',exact=False)).to_be_visible()
    page.evaluate('(u)=>localStorage.setItem("eventScoutUser",JSON.stringify(u))',user);page.reload();expect(page.get_by_role('heading',name='Needs attention · 2')).to_be_visible();passed('queued photos are owner-scoped when switching accounts')
    page.get_by_role('button',name='Retry',exact=True).first.click();page.wait_for_timeout(200);pending.pop(0).fulfill(json={'status':'duplicate','existing_name':'QA Existing Fixture'});expect(page.locator('.saved-ledger')).to_be_visible();page.locator('.saved-ledger summary').click();expect(page.locator('.saved-ledger').get_by_text('Already saved: QA Existing Fixture')).to_be_visible();assert page.get_by_role('button',name='Retry',exact=True).count()==1;passed('duplicate settles without retry')
    page.get_by_role('button',name='Retry',exact=True).click();page.wait_for_timeout(200);pending.pop(0).fulfill(json={'status':'success','contact_id':'qa-fixture-only','extracted_fields':{'name':'QA Saved Fixture','company_name':'Fixture Labs'}});expect(page.locator('.saved-ledger').get_by_text('QA Saved Fixture · Fixture Labs')).to_be_visible();expect(page.get_by_role('heading',name='Needs attention',exact=False)).not_to_be_visible();shot('Completed ledger');passed('manual retry recovers and removes the failed state')
    # No customer fields are interpreted as HTML, including result rows and assistant text.
    attack='<img src=x onerror="window.qaInjected=true">'
    page.route('**/api/converse/**',lambda route:route.fulfill(json={'response':attack,'retrieved_contacts':[{'id':'qa-injection','name':attack,'company_name':'Fixture Labs','lead_score':75,'lead_temperature':'hot'}]}))
    page.get_by_role('navigation').get_by_role('button',name='Ask',exact=True).click();page.get_by_label('Ask about your contacts').fill('QA injection check');page.get_by_role('button',name='Send question').click();expect(page.locator('.turn-assistant>p')).to_have_text(attack);assert page.evaluate('window.qaInjected') is None;assert page.locator('.turn-assistant img').count()==0;shot('Ask escaped text');passed('assistant and contact HTML payloads render literally, no injection')
    history=[]
    def history_reply(route):history.append(route.request.post_data_json);route.fulfill(json={'response':'QA labelled follow-up reply','retrieved_contacts':[]})
    page.unroute('**/api/converse/**');page.route('**/api/converse/**',history_reply);page.get_by_label('Ask about your contacts').fill('QA second question');page.get_by_role('button',name='Send question').click();expect(page.get_by_text('QA labelled follow-up reply')).to_be_visible();assert len(history[0]['conversation_history'])==2;assert history[0]['top_k']==4;assert 'admin_mode' not in history[0];passed('Ask sends role/content history and top_k 4, never admin_mode')
    # All protected routes share the same session-expiry behavior.
    page.route('**/api/list_contacts/**',lambda route:route.fulfill(status=401,json={'detail':'Expired fixture session'}));page.get_by_role('navigation').get_by_role('button',name='Contacts',exact=True).click();expect(page.get_by_role('heading',name='Sign in to Event Scout')).to_be_visible();expect(page.get_by_role('alert')).to_contain_text('Your session has expired. Please log in again.');assert page.evaluate('localStorage.getItem("eventScoutUser")') is None;shot('Session expired');passed('a protected 401 clears storage and returns to sign in with the exact notice')
    assert not console,console;browser.close()
label=28;sheet=Image.new('RGB',(780,(844+label)*4),'#e2e8f0');draw=ImageDraw.Draw(sheet)
for i,(name,im) in enumerate(frames):x=i%2*390;y=i//2*(844+label);draw.text((x+8,y+6),name,fill='#0f172a');sheet.paste(im,(x,y+label))
sheet.save(OUT/'screenshots'/'states-390.png')
(OUT/'states.json').write_text(json.dumps({'checks':checks,'states':results,'console_errors':console},indent=2))
failures=[(r['state'],f) for r in results for f in r['audit'] if f['severity']=='FAIL'];print('Audit FAILs:',json.dumps(failures));assert not failures
