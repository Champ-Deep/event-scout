/* Stubbed run for the scan-to-score flow. Paste into the console of mobile-frontend.html (served over http),
   or evaluate it with a browser tool. It needs no backend, camera or login.
     await scanFlowScenario('permission_denied')   // leaves the page in that state so you can screenshot it
     await scanFlowRunAll()                        // walks every state and retry/manual path, returns [{name, ok, got}]
   RESULT below is the real output of scan_analysis.analyze() (local qwen2.5:1.5b via Ollama) for tests/test_scan_analysis.py HIT/PROFILE. */
(() => {
    const RESULT = { score: 94, temperature: 'hot', breakdown: { profile_fit: 20, role_relevance: 20, company_fit: 20, engagement: 14, timing: 20 },
        engine: 'ollama:qwen2.5:1.5b-instruct', signals: {},
        reasoning: "Asha Rao, the VP Marketing at PayCo fintech, has been identified as a high interest lead for Lake B2B's event lead data. With a score of 94/100, she is highly profiled, relevant to Lake B2B's target roles, and matches Lake B2B's industry focus on fintech. To proceed, Lake B2B should prioritize reaching out to Asha Rao via email and phone, verifying her contacts, and booking demos for her." };
    const FIELDS = { name: 'Asha Rao', company_name: 'PayCo fintech', email: 'asha@payco.io', phone: '+91 99999 11111', linkedin: 'N/A' };
    const origToBlob = HTMLCanvasElement.prototype.toBlob;
    const json = (status, body) => ({ ok: status < 400, status, json: async () => body });
    const never = () => new Promise(() => {});
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    const state = () => document.querySelector('#scanFlow [data-state]')?.dataset.state;
    const click = act => document.querySelector(`#scanFlow [data-act="${act}"]`).click();
    const until = async (want) => { for (let i = 0; i < 50 && state() !== want; i++) await sleep(20); return state(); };

    function fakeStream() {
        const c = Object.assign(document.createElement('canvas'), { width: 640, height: 400 }), g = c.getContext('2d');
        g.fillStyle = '#e8eef7'; g.fillRect(0, 0, 640, 400); g.fillStyle = '#1f2937'; g.font = 'bold 40px sans-serif';
        g.fillText('Asha Rao', 60, 150); g.font = '26px sans-serif'; g.fillText('VP Marketing, PayCo fintech', 60, 200);
        g.fillText('asha@payco.io  +91 99999 11111', 60, 250);
        return c.captureStream(5);
    }

    // routes: extract / analyze / save each get a handler
    function setup({ camera = 'ok', extract = () => json(200, { fields: FIELDS }), analyze = () => json(200, RESULT), save = () => json(200, { status: 'success' }) } = {}) {
        window.CONFIG = window.CONFIG || { API_BASE_URL: '', API_KEY: 'test' };
        AppState.currentUser = { user_id: '11111111-1111-1111-1111-111111111111' };
        navigator.mediaDevices.getUserMedia = camera === 'ok' ? async () => fakeStream()
            : async () => { throw Object.assign(new Error(camera), { name: camera }); };
        ScanFlow.api.http = async (url) => (url.includes('/scan/extract') ? extract() : url.includes('/scan/analyze') ? analyze() : save());
    }

    // each scenario forwards its stub options down the chain, so a later step never resets an earlier stub
    const scenarios = {
        async permission_denied() { setup({ camera: 'NotAllowedError' }); ScanFlow.open(); return until('permission_denied'); },
        async camera(o) { setup(o); ScanFlow.open(); await sleep(400); return state(); },
        async capturing() { setup(); HTMLCanvasElement.prototype.toBlob = function () {}; ScanFlow.open(); await sleep(500); click('shutter'); return until('capturing'); },
        async extracting() { await scenarios.camera({ extract: never }); await shutter(); return until('extracting'); },
        async extract_failed() { await scenarios.camera({ extract: () => json(422, { detail: 'No contact information found in the image.' }) }); await shutter(); return until('extract_failed'); },
        async data(o) { await scenarios.camera(o); await shutter(); return until('data'); },
        async analyzing() { await scenarios.data({ analyze: never }); click('analyze'); return until('analyzing'); },
        async analysis_failed() { await scenarios.data({ analyze: () => json(500, {}) }); click('analyze'); return until('analysis_failed'); },
        async result() { await scenarios.data(); click('analyze'); return until('result'); },
    };
    async function shutter() { // real canvas->blob path
        HTMLCanvasElement.prototype.toBlob = origToBlob; await sleep(300); click('shutter');
    }
    window.scanFlowScenario = async name => { ScanFlow.close(); const got = await scenarios[name](); return { name, got }; };

    window.scanFlowRunAll = async () => {
        const out = [], check = (name, got, want) => out.push({ name, ok: got === want, got });
        for (const n of Object.keys(scenarios)) { ScanFlow.close(); check(n, await scenarios[n](), n); }

        // every failure state has a way out
        ScanFlow.close(); await scenarios.permission_denied(); click('manual'); check('denied -> manual -> data', await until('data'), 'data');
        ScanFlow.close(); await scenarios.permission_denied(); setup({ camera: 'ok' }); click('retryCamera'); await sleep(300); check('denied -> retry camera', state(), 'camera');
        ScanFlow.close(); await scenarios.extract_failed(); setup(); click('retryExtract'); check('extract failed -> retry -> data', await until('data'), 'data');
        ScanFlow.close(); await scenarios.extract_failed(); click('manual'); check('extract failed -> manual -> data', await until('data'), 'data');
        ScanFlow.close(); await scenarios.analysis_failed(); setup(); click('analyze'); check('analysis failed -> retry -> result', await until('result'), 'result');
        ScanFlow.close(); await scenarios.analysis_failed(); click('edit'); check('analysis failed -> edit -> data', await until('data'), 'data');
        ScanFlow.close(); await scenarios.analysis_failed(); setup(); click('save'); await sleep(100);
        out.push({ name: 'analysis failed -> save without score closes flow', ok: !document.getElementById('scanFlow'), got: state() });

        // result content: score and reasoning rendered from the response, not template text
        ScanFlow.close(); await scenarios.result();
        const sc = document.querySelector('[data-score-value]')?.textContent, rs = document.querySelector('[data-score-reason]')?.textContent;
        out.push({ name: 'result shows score + reasoning naming user inputs', ok: sc === '94' && /Lake B2B/.test(rs) && /event lead data/.test(rs), got: `${sc} | ${rs?.slice(0, 60)}` });

        // data screen: validation and name-required guard
        ScanFlow.close(); await scenarios.data();
        const email = document.querySelector('[data-field="email"]'); email.value = 'not-an-email'; email.dispatchEvent(new Event('change', { bubbles: true }));
        const name = document.querySelector('[data-field="name"]'); name.value = ''; name.dispatchEvent(new Event('change', { bubbles: true }));
        click('analyze');
        out.push({ name: 'data screen flags bad email and blocks empty name', ok: state() === 'data' && document.querySelectorAll('#scanFlow [data-warn]').length === 2, got: document.querySelectorAll('#scanFlow [data-warn]').length });

        // save failure on result keeps the flow open with an error
        ScanFlow.close(); await scenarios.result(); ScanFlow.api.http = async url => (url.includes('/add_contact/') ? json(500, {}) : json(200, {})); click('save'); await sleep(100);
        out.push({ name: 'save failure shows error, stays on result', ok: state() === 'result' && !!document.querySelector('[data-save-error]'), got: state() });
        ScanFlow.close();
        return out;
    };
})();
