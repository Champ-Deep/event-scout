# Responsive QA on the real local Worker, using synthetic accounts and records only.
# Run: SCOUT_QA_URL=http://localhost:8792 <playwright-python> web/qa/responsive.py
import io, json, os, pathlib, time, urllib.parse
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright, expect
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'web/qa';BASE=os.environ.get('SCOUT_QA_URL','http://localhost:8792')
assert urllib.parse.urlparse(BASE).hostname in ('localhost','127.0.0.1')
from audit_probe import audit
results=[];errors=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch()
    for width in [360,375,390,699,768,1280,1440]:
        height=844 if width<700 else 900;context=browser.new_context(viewport={'width':width,'height':height},has_touch=width<700,reduced_motion='reduce' if width==699 else 'no-preference');page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        frames=[]
        def capture(state,shot=True):
            page.wait_for_timeout(250);page.evaluate('window.scrollTo(0,0)')
            findings=page.evaluate(audit.PROBE)
            geometry=page.evaluate('''() => ({w:innerWidth,scroll:document.documentElement.scrollWidth,dashes:(document.body.innerText.match(/[\\u2013\\u2014]/g)||[]).length,small:[...document.querySelectorAll('button,input,textarea,select,a,summary')].filter(e=>{const r=e.getBoundingClientRect();return r.width&&r.height&&!e.closest('[hidden]')&&(r.width<44||r.height<44)}).map(e=>e.outerHTML.slice(0,120)),overflow:[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return r.width&&r.right>innerWidth+1&&!e.closest('[hidden]')}).map(e=>e.tagName+'.'+e.className)})''')
            results.append({'width':width,'state':state,'geometry':geometry,'audit':findings})
            assert geometry['scroll']==width,(width,state,geometry)
            assert not geometry['dashes'],(width,state,'dashes')
            assert not geometry['small'],(width,state,geometry['small'])
            if shot and width!=699:frames.append((state,Image.open(io.BytesIO(page.screenshot())).convert('RGB')))
        def nav(label):page.get_by_role('navigation',name='Main navigation').get_by_role('button',name=label,exact=True).click()
        page.goto(BASE);capture('Sign in')
        email=f'responsive-{width}-{int(time.time())}@example.invalid';user=context.request.post(BASE+'/api/register/',data={'name':'QA Field Scout','email':email,'password':'local-qa-only-123'}).json();uid=user['user_id'];headers={'X-API-Key':user['session_token']}
        def api(method,path,data=None):
            r=context.request.fetch(BASE+'/api'+path,method=method,headers=headers,data=data);assert r.ok,(path,r.status);return r.json()
        cid=api('POST','/add_contact/',{'user_id':uid,'contact':{'name':'QA Fixture Contact','company_name':'Fixture Labs','title':'Field sales','email':'fixture@example.invalid','phone':'+1 555 0100','linkedin':'N/A'}})['contact_id']
        page.evaluate('(u)=>localStorage.setItem("eventScoutUser",JSON.stringify(u))',user);page.reload();expect(page.get_by_role('heading',name='Scan cards')).to_be_visible();capture('Scan')
        nav('Contacts');expect(page.get_by_role('button',name='QA Fixture Contact',exact=False)).to_be_visible();capture('Contacts')
        # Touch hits the same row as pointer and keyboard, without direct DOM click bypasses.
        row=page.get_by_role('button',name='QA Fixture Contact',exact=False)
        if width<700:row.tap()
        else:row.focus();page.keyboard.press('Enter')
        expect(page.get_by_role('dialog')).to_be_visible();expect(page.get_by_label('Name',exact=True)).to_have_value('QA Fixture Contact');capture('Contact detail')
        page.get_by_role('button',name='Close contact',exact=True).click();expect(page.get_by_role('dialog')).not_to_be_visible()
        nav('Ask');capture('Ask')
        # Labelled UI-only error/recovery fixture, not an AI success claim.
        long_reply='\n'.join(f'Line {i}: labelled layout fixture.' for i in range(1, 49))
        replies=[{'detail':'QA Ask unavailable. Please retry.'},{'response':long_reply,'retrieved_contacts':[]}]
        def answer(route):
            data=replies.pop(0);route.fulfill(status=500 if 'detail' in data else 200,json=data)
        page.route('**/api/converse/**',answer);page.get_by_label('Ask about your contacts').fill('QA recovery question');page.get_by_role('button',name='Send question').click();expect(page.get_by_role('button',name='Try again',exact=True)).to_be_visible();capture('Ask error fixture',False)
        assert page.evaluate('''() => {const b=document.querySelector('.notice button'),r=b.getBoundingClientRect(),c=document.querySelector('.composer').getBoundingClientRect();return r.bottom<=c.top&&document.elementFromPoint(r.left+r.width/2,r.top+r.height/2)?.closest('button')===b}'''),(width,'Ask Retry is covered')
        page.get_by_role('button',name='Try again',exact=True).click();expect(page.locator('.turn-assistant p')).to_have_text(long_reply);expect(page.get_by_role('alert')).not_to_be_visible();capture('Ask recovered fixture',False);page.unroute('**/api/converse/**')
        pane=page.locator('.conversation').bounding_box();reply=page.locator('.turn-assistant').last.bounding_box()
        assert reply['height'] > pane['height'] and pane['y'] <= reply['y'] < pane['y']+pane['height'], (width,pane,reply)
        if width in (360,390,768,1280): page.screenshot(path=OUT/'screenshots'/f'ask-long-{width}.png')
        print('Ask pane / answer top:',width,pane['y'],reply['y'],flush=True)
        nav('Me');expect(page.get_by_label('Full name',exact=True)).to_be_visible();capture('Profile')
        page.get_by_role('tab',name='Event',exact=True).click();capture('Event')
        page.get_by_role('tab',name='My card',exact=True).click();capture('My card')
        # Keyboard tabs stay within an opened detail sheet and Escape restores the invoking row.
        nav('Contacts');row=page.get_by_role('button',name='QA Fixture Contact',exact=False);row.click();expect(page.get_by_role('dialog')).to_be_visible();expect(page.get_by_label('Name',exact=True)).to_have_value('QA Fixture Contact');page.wait_for_timeout(100);page.keyboard.press('Tab');assert page.evaluate('!!document.activeElement.closest("[role=dialog]")'),page.evaluate('document.activeElement.outerHTML');page.keyboard.press('Escape');expect(page.get_by_role('dialog')).not_to_be_visible();capture('Keyboard return',False)
        if frames:
            label_h=28;sheet=Image.new('RGB',(width*2,(height+label_h)*4),'#e2e8f0');draw=ImageDraw.Draw(sheet)
            for i,(name,image) in enumerate(frames):
                x=(i%2)*width;y=(i//2)*(height+label_h);draw.text((x+12,y+6),f'{name} | {width}px | synthetic local QA',fill='#0f172a');sheet.paste(image,(x,y+label_h))
            sheet.save(OUT/'screenshots'/f'responsive-{width}.png')
        api('DELETE',f'/contact/{cid}?user_id={uid}');api('POST','/logout/');context.close();print('PASS viewport',width,'11 states, local data plus labelled Ask recovery fixture, pointer/touch, keyboard and geometry',flush=True)
    browser.close()
(OUT/'responsive.json').write_text(json.dumps({'results':results,'console_errors':errors},indent=2))
failures=[(r['width'],r['state'],f) for r in results for f in r['audit'] if f['severity']=='FAIL']
print('Audit FAILs:',json.dumps(failures));print('Console errors:',json.dumps(errors));assert not failures;assert not errors
