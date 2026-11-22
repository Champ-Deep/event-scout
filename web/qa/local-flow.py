# Run from the repo root with a Playwright-enabled Python against wrangler dev --local.
# Creates synthetic data only. Never run against a deployed host. No session tokens are written to disk.
import json, os, pathlib, time, urllib.parse, urllib.request
from playwright.sync_api import sync_playwright, expect
ROOT=pathlib.Path(__file__).resolve().parents[2]
BASE=os.environ.get('SCOUT_QA_URL','http://localhost:8792')
assert urllib.parse.urlparse(BASE).hostname in ('localhost','127.0.0.1'), 'Local QA only'
OUT=ROOT/'web/qa'; SHOTS=OUT/'screenshots';SHOTS.mkdir(exist_ok=True)
from audit_probe import audit
report={'base':BASE,'checks':[],'states':[],'ocr':None,'console_errors':[],'browser_console':[]}
import atexit
atexit.register(lambda: (OUT/'playwright-real.json').write_text(json.dumps(report,indent=2)))
def passed(label): report['checks'].append(label);print('PASS',label,flush=True)
def measure(page,state,shot=True):
    page.wait_for_timeout(200)
    findings=page.evaluate(audit.PROBE)
    geometry=page.evaluate('''() => ({width:innerWidth,scroll:document.documentElement.scrollWidth,dashes:(document.body.innerText.match(/[\\u2013\\u2014]/g)||[]).length,small:[...document.querySelectorAll('button,input,textarea,select,a,summary')].filter(e=>{const r=e.getBoundingClientRect();const s=getComputedStyle(e);return r.width>0&&r.height>0&&s.visibility!=='hidden'&&!e.closest('[hidden]')&&(r.width<44||r.height<44)}).map(e=>({tag:e.tagName,text:e.innerText||e.getAttribute('aria-label'),width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height})),overflow:[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return r.width&&r.right>innerWidth+1&&!e.closest('[hidden]')}).map(e=>e.tagName+'.'+e.className)})''')
    assert geometry['scroll']==geometry['width'],(state,geometry)
    assert not geometry['dashes'],(state,'forbidden dash')
    assert not geometry['small'],(state,geometry['small'])
    report['states'].append({'state':state,'width':geometry['width'],'geometry':geometry,'audit':findings})
    if shot:
        page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(SHOTS/f'{state}-{geometry["width"]}.png'),full_page=True)
    failures=[f for f in findings if f['severity']=='FAIL']
    if failures:print('AUDIT FAIL',state,json.dumps(failures),flush=True)
def nav(page,label):page.get_by_role('navigation',name='Main navigation').get_by_role('button',name=label,exact=True).click()
with sync_playwright() as pw:
    browser=pw.chromium.launch();context=browser.new_context(viewport={'width':390,'height':844},accept_downloads=True)
    page=context.new_page();page.on('pageerror',lambda e:report['console_errors'].append(str(e)));page.on('console',lambda msg:report['browser_console'].append({'type':msg.type,'text':msg.text}) if msg.type in ('error','warning') else None)
    assert urllib.request.urlopen(BASE).read()==(ROOT/'worker/public/index.html').read_bytes();passed('served index is the exact built artifact')
    page.goto(BASE);expect(page.get_by_role('heading',name='Sign in to Event Scout')).to_be_visible();measure(page,'auth')
    page.get_by_role('button',name='Create an account').click();email=f'ui-qa-{int(time.time())}@example.invalid';password='local-qa-only-123'
    page.get_by_label('Name',exact=True).fill('QA Field Scout');page.get_by_label('Email',exact=True).fill(email);page.get_by_label('Password',exact=True).fill(password)
    page.get_by_role('button',name='Create account',exact=True).click();expect(page.get_by_role('heading',name='Scan cards')).to_be_visible(timeout=15000);passed('register and enter default Scan tab')
    session=page.evaluate('JSON.parse(localStorage.eventScoutUser)');headers={'X-API-Key':session['session_token']};uid=session['user_id']
    def api(method,path,data=None):
        r=context.request.fetch(BASE+'/api'+path,method=method,headers=headers,data=data);assert r.ok,(path,r.status);return r.json()
    measure(page,'scan')
    nav(page,'Me');page.get_by_role('button',name='Sign out',exact=True).click();expect(page.get_by_role('heading',name='Sign in to Event Scout')).to_be_visible();passed('sign out clears legacy session storage')
    assert page.evaluate('localStorage.getItem("eventScoutUser")') is None
    page.get_by_label('Email',exact=True).fill(email);page.get_by_label('Password',exact=True).fill(password);page.get_by_role('button',name='Sign in',exact=True).click();expect(page.get_by_role('heading',name='Scan cards')).to_be_visible();passed('sign in with registered user')
    session=page.evaluate('JSON.parse(localStorage.eventScoutUser)');headers={'X-API-Key':session['session_token']}
    page.get_by_role('button',name='Enter a contact manually').click();page.get_by_label('Name',exact=True).fill('QA Fixture Contact');page.get_by_label('Company',exact=True).fill('Fixture Labs');page.get_by_label('Job title',exact=True).fill('Field sales');page.get_by_label('Email',exact=True).fill('fixture@example.invalid');measure(page,'manual')
    page.get_by_role('button',name='Save new contact').click();expect(page.get_by_role('heading',name='New contact')).not_to_be_visible();passed('manual add through real local Worker and D1')
    nav(page,'Contacts');expect(page.get_by_role('button',name='QA Fixture Contact',exact=False)).to_be_visible();measure(page,'contacts')
    contact=api('GET',f'/list_contacts/?user_id={uid}')['contacts'][0];cid=contact['id']
    page.get_by_role('button',name='QA Fixture Contact',exact=False).click();expect(page.get_by_role('dialog')).to_be_visible();page.get_by_label('Notes',exact=True).fill('QA follow-up from the local fixture booth.');measure(page,'detail')
    page.get_by_role('button',name='Add link',exact=True).click();page.get_by_label('Link 1 label').fill('Fixture website');page.get_by_label('Link 1 URL').fill('https://example.invalid/fixture');page.get_by_role('button',name='Save contact',exact=True).click();expect(page.get_by_role('dialog')).not_to_be_visible();assert api('GET',f'/contact/{cid}?user_id={uid}')['contact']['notes']=='QA follow-up from the local fixture booth.';passed('detail edit persists notes and a safe link')
    with page.expect_download() as dl:page.get_by_role('button',name='Export CSV',exact=True).click()
    csv=pathlib.Path(dl.value.path()).read_bytes();assert csv[:3]==bytes([239,187,191]);assert b'QA Fixture Contact' in csv;passed('CSV export downloads UTF-8 BOM EF BB BF')
    page.get_by_label('Search contacts').fill('not a matching fixture');expect(page.get_by_role('heading',name='No matching contacts')).to_be_visible();page.get_by_label('Search contacts').fill('');page.get_by_role('button',name='Today',exact=True).click();expect(page.get_by_role('button',name='QA Fixture Contact',exact=False)).to_be_visible();passed('client search and server date filter')
    nav(page,'Me');expect(page.get_by_label('Full name',exact=True)).to_be_visible();page.get_by_label('Full name',exact=True).fill('QA Field Scout');page.get_by_label('Company',exact=True).fill('Fixture Labs');page.get_by_role('button',name='Save profile',exact=True).click();expect(page.get_by_role('button',name='Save profile',exact=True)).to_be_enabled();passed('profile save')
    page.get_by_role('tab',name='Event',exact=True).click();page.get_by_label('Event name').fill('QA Local Expo');page.get_by_label('Event description').fill('Synthetic local UI verification');page.get_by_role('button',name='Save event',exact=True).click();expect(page.get_by_role('button',name='Save event',exact=True)).to_be_enabled();assert api('GET',f'/user/profile/?user_id={uid}')['profile']['current_event_name']=='QA Local Expo';measure(page,'event');passed('event save persists through real API')
    page.get_by_role('tab',name='My card',exact=True).click();page.get_by_label('Full name',exact=True).fill('QA Field Scout');page.get_by_label('Email',exact=True).fill('fixture@example.invalid');page.get_by_role('button',name='Save card',exact=True).click();expect(page.get_by_role('button',name='Save card',exact=True)).to_be_enabled();page.get_by_role('button',name='Generate card link',exact=True).click();expect(page.get_by_alt_text('QR code for your digital card')).to_be_visible();assert page.get_by_alt_text('QR code for your digital card').get_attribute('src').startswith('data:image/svg+xml;base64,');measure(page,'card');passed('save digital card, generate share URL and real QR')
    context.grant_permissions(['clipboard-read','clipboard-write'],origin=BASE);page.get_by_role('button',name='Copy link',exact=True).click();expect(page.get_by_text('Link copied',exact=True)).to_be_visible();assert page.evaluate('navigator.clipboard.readText()')==page.locator('.share-url').inner_text();passed('copy card link confirms and writes the exact URL')
    page.get_by_role('tab',name='Profile',exact=True).click();measure(page,'profile')
    nav(page,'Ask');page.get_by_label('Ask about your contacts').fill('Summarize my contacts')
    with page.expect_response(lambda r:'/api/converse/' in r.url,timeout=45000) as asked:page.get_by_role('button',name='Send question').click()
    answer=asked.value
    if answer.ok:
        expect(page.locator('.turn-assistant')).to_be_visible();assert page.locator('.turn-assistant p').first.inner_text().strip();measure(page,'ask');passed('Ask returns a real reply from local Worker AI path')
    else:
        report['ask_blocker']={'status':answer.status,'detail':answer.json().get('detail')};expect(page.get_by_role('alert')).to_be_visible();measure(page,'ask-unavailable');print('UNVERIFIED Ask success:',report['ask_blocker'],flush=True)

    # Never capture the benchmark card or its extracted identity in screenshots.
    nav(page,'Scan');card_path=pathlib.Path.home()/'repos/event-scout/bench/cards/c01_photo.jpg'
    if card_path.exists():
        with page.expect_response(lambda r:'/api/add_contact_from_image/' in r.url,timeout=45000) as scan:
            page.locator('input[aria-label="Choose card photos"]').set_input_files(str(card_path))
        response=scan.value;data=response.json();report['ocr']={'real_status':response.status,'real_result':data.get('status'),'source':data.get('ocr_source')}
        if response.ok and data.get('status') in ('success','duplicate'):
            expect(page.locator('.saved-ledger')).to_be_visible(timeout=5000)
            if data.get('contact_id'):api('DELETE',f'/contact/{data["contact_id"]}?user_id={uid}')
            passed('benchmark photo scanned through real local AI binding; no image or identity captured')
        else:
            print('OCR unavailable locally:',response.status,flush=True)
            # The brief permits mocking only OCR when the remote AI binding is unavailable.
            fake=api('POST','/add_contact/',{'user_id':uid,'contact':{'name':'QA OCR Fixture','email':'ocr@example.invalid','phone':'N/A','linkedin':'N/A','company_name':'Fixture Labs'}})
            page.route('**/api/add_contact_from_image/**',lambda route:route.fulfill(json={'status':'success','contact_id':fake['contact_id'],'extracted_fields':{'name':'QA OCR Fixture','company_name':'Fixture Labs'}}))
            page.get_by_role('button',name='Retry',exact=True).click();expect(page.locator('.saved-ledger')).to_be_visible();api('DELETE',f'/contact/{fake["contact_id"]}?user_id={uid}');page.unroute('**/api/add_contact_from_image/**');report['ocr']['mocked_response']=True;passed('LABELLED OCR-only response fixture exercises successful queue recovery')
        page.reload();expect(page.get_by_role('heading',name='Scan cards')).to_be_visible()
    else:raise AssertionError('Required benchmark image is missing')
    nav(page,'Contacts');page.get_by_role('button',name='QA Fixture Contact',exact=False).click();expect(page.get_by_label('Notes',exact=True)).to_have_value('QA follow-up from the local fixture booth.')
    with page.expect_download() as dl:page.get_by_role('button',name='Download vCard').click()
    assert b'BEGIN:VCARD' in pathlib.Path(dl.value.path()).read_bytes();passed('vCard downloads the saved contact')
    page.get_by_role('button',name='Delete contact',exact=True).click();page.get_by_role('button',name='Keep contact',exact=True).click();expect(page.get_by_role('button',name='Confirm delete',exact=True)).not_to_be_visible();page.get_by_role('button',name='Delete contact',exact=True).click();page.get_by_role('button',name='Confirm delete',exact=True).click();expect(page.get_by_role('dialog')).not_to_be_visible();expect(page.get_by_role('heading',name='Scan your first card')).to_be_visible();assert api('GET',f'/list_contacts/?user_id={uid}')['total_contacts']==0;passed('delete confirmation, cancellation and real deletion')
    nav(page,'Me');page.get_by_role('button',name='Sign out',exact=True).click();expect(page.get_by_role('heading',name='Sign in to Event Scout')).to_be_visible();passed('final logout')
    assert not report['console_errors'],report['console_errors'];passed('zero JavaScript page errors')
    context.close();browser.close()
(OUT/'playwright-real.json').write_text(json.dumps(report,indent=2));print('Report:',OUT/'playwright-real.json',flush=True)
