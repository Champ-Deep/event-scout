/* Hi-Fi scan-to-score flow: capture -> read card -> confirm fields -> local analysis -> score.
   Opens as a full-screen overlay, so it never touches the existing scan screen's state.
   Every failure state has a retry and a manual path. Needs CONFIG, AppState, resilientFetch from mobile-frontend.html.
   Tests replace ScanFlow.http and navigator.mediaDevices.getUserMedia (see tests/scan_flow_stubbed_run.js). */
window.ScanFlow = (() => {
    const COPY = {
        cameraTitle: 'Scan a business card',
        denied: ['Camera access is off', 'Allow the camera in your browser settings, or add the card another way.'],
        unavailable: ['No camera found', 'Upload a photo of the card, or type the details yourself.'],
        capturing: 'Capturing the card...',
        extracting: ['Reading the card', 'Finding name, company and contact details.'],
        extractFailed: ['We could not read that card', 'Try again, retake the photo, or type the details yourself.'],
        dataTitle: 'Check what we read',
        dataHint: 'Fix anything that looks wrong. Add a job title in notes: it drives the role match.',
        analyzing: ['Scoring this lead', 'Comparing the contact with what you sell and who you target. This runs on a local model, so it can take a few seconds.'],
        analysisFailed: ['Scoring did not finish', 'Your details are safe. Retry, edit them, or save the contact without a score.'],
        saveFailed: 'Could not save. Check your connection and try again.',
    };
    const FIELDS = [['name', 'Name', 'text'], ['company_name', 'Company', 'text'], ['email', 'Email', 'email'],
                    ['phone', 'Phone', 'tel'], ['linkedin', 'LinkedIn', 'url'], ['notes', 'Job title / notes', 'text']];
    const EMAIL = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
    const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const clean = v => (!v || String(v).trim().toUpperCase() === 'N/A') ? '' : String(v).trim();

    const api = { http: (url, opts) => resilientFetch(url, opts) };
    let S, root, stream, video;

    const uid = () => AppState.currentUser.user_id;
    const headers = () => ({ 'X-API-Key': CONFIG.API_KEY });
    const contact = () => Object.fromEntries(FIELDS.map(([k]) => [k, clean(S.fields[k]) || (k === 'notes' ? '' : 'N/A')]));

    async function detail(resp, fallback) {
        try { return (await resp.json()).detail || fallback; } catch { return fallback; }
    }

    function stopCamera() { stream?.getTracks().forEach(t => t.stop()); stream = null; }

    function go(state, extra = {}) {
        if (!root) return; // flow was closed while a request was in flight
        if (state !== 'camera') stopCamera();
        Object.assign(S, extra, { state });
        render();
    }

    // ---------- views ----------
    const wrap = (inner, state) => `<div data-state="${state}" class="h-full w-full flex flex-col">${inner}</div>`;
    const topBar = (title, back, label = 'Cancel') => `
        <div class="flex items-center justify-between px-4 py-3 bg-white border-b border-gray-200">
            <button data-act="close" class="text-gray-500 text-sm" aria-label="Close scan">${esc(label)}</button>
            <h2 class="font-semibold text-gray-900">${esc(title)}</h2>
            <span class="w-12">${back || ''}</span>
        </div>`;
    const btn = (act, label, kind = 'secondary') => `<button data-act="${act}" class="w-full py-3 rounded-xl font-semibold ${
        kind === 'primary' ? 'bg-blue-600 text-white active:bg-blue-700' : 'bg-white text-gray-800 border border-gray-300 active:bg-gray-100'}">${esc(label)}</button>`;
    const message = (icon, [title, body], actions, state, tone = 'text-gray-400', bar = COPY.cameraTitle) => wrap(`
        ${topBar(bar)}
        <div class="flex-1 flex flex-col items-center justify-center text-center px-8 bg-gray-50">
            <i class="fas ${icon} text-6xl ${tone} mb-4"></i>
            <h3 class="text-xl font-semibold text-gray-900 mb-2">${esc(title)}</h3>
            <p class="text-sm text-gray-600 mb-6 max-w-xs">${esc(body)}</p>
            <div class="w-full max-w-xs space-y-2">${actions}</div>
        </div>`, state);
    const cardThumb = (pulse) => S.imgUrl ? `<img src="${S.imgUrl}" alt="Scanned card" class="w-64 max-w-full rounded-xl shadow-lg ${pulse ? 'animate-pulse ring-4 ring-blue-400' : ''}">` : '';
    const progress = (steps, active) => `<ol class="mt-6 space-y-2 text-sm text-left">${steps.map((s, i) => `
        <li class="flex items-center gap-2 ${i < active ? 'text-green-600' : i === active ? 'text-gray-900 font-medium' : 'text-gray-400'}">
            <i class="fas ${i < active ? 'fa-check-circle' : i === active ? 'fa-spinner fa-spin' : 'fa-circle'} w-4"></i>${esc(s)}</li>`).join('')}</ol>`;

    const views = {
        camera: () => wrap(`
            <div class="flex-1 relative bg-black">
                <video id="scanFlowVideo" class="absolute inset-0 w-full h-full object-cover" autoplay playsinline muted></video>
                <button data-act="close" class="absolute top-4 left-4 text-white bg-black/50 px-3 py-1.5 rounded-full text-sm">Cancel</button>
                <div class="absolute inset-x-8 top-1/3 h-40 border-2 border-white rounded-xl pointer-events-none shadow-[0_0_0_9999px_rgba(0,0,0,0.4)]"></div>
                <p class="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black/60 text-white text-sm px-3 py-1 rounded-full whitespace-nowrap">Fit the card inside the frame</p>
            </div>
            <div class="bg-gray-900 px-4 py-5 flex justify-center items-center gap-8">
                <button data-act="gallery" class="w-11 h-11 bg-gray-700 rounded-full text-white" aria-label="Upload from gallery"><i class="fas fa-image"></i></button>
                <button data-act="shutter" class="w-16 h-16 bg-white rounded-full border-4 border-gray-300 active:scale-95" aria-label="Capture card"></button>
                <button data-act="manual" class="w-11 h-11 bg-gray-700 rounded-full text-white" aria-label="Type details instead"><i class="fas fa-keyboard"></i></button>
            </div>`, 'camera'),

        permission_denied: () => message('fa-video-slash', S.noCamera ? COPY.unavailable : COPY.denied,
            btn('retryCamera', 'Try camera again', 'primary') + btn('gallery', 'Upload from gallery') + btn('manual', 'Type details instead'), 'permission_denied'),

        capturing: () => wrap(`${topBar(COPY.cameraTitle)}
            <div class="flex-1 flex flex-col items-center justify-center bg-gray-900 text-white"><i class="fas fa-camera fa-beat text-5xl"></i><p class="mt-4 text-sm">${COPY.capturing}</p></div>`, 'capturing'),

        extracting: () => wrap(`${topBar(COPY.cameraTitle)}
            <div class="flex-1 flex flex-col items-center justify-center bg-gray-50 px-8 text-center">${cardThumb(true)}
                <h3 class="mt-5 text-lg font-semibold text-gray-900">${COPY.extracting[0]}</h3>
                <p class="text-sm text-gray-600">${COPY.extracting[1]}</p>
                ${progress(['Photo captured', 'Reading text', 'Matching fields'], 1)}</div>`, 'extracting'),

        extract_failed: () => message('fa-exclamation-triangle', [COPY.extractFailed[0], S.error || COPY.extractFailed[1]],
            btn('retryExtract', 'Try reading again', 'primary') + btn('retake', 'Retake photo') + btn('manual', 'Type details instead'), 'extract_failed', 'text-amber-500'),

        data: () => {
            const f = S.fields, badEmail = clean(f.email) && !EMAIL.test(f.email.trim()), noName = !clean(f.name);
            const inputs = FIELDS.map(([k, label, type]) => {
                const warn = (k === 'email' && badEmail) ? 'This does not look like an email address.' : (k === 'name' && noName) ? 'A name is needed to score the lead.' : '';
                return `<label class="block"><span class="text-xs font-medium text-gray-600">${label}</span>
                    <input data-field="${k}" type="${type}" value="${esc(clean(f[k]))}" placeholder="${S.fromScan && !clean(f[k]) ? 'Not found on the card' : ''}"
                        class="mt-1 w-full px-3 py-2 border ${warn ? 'border-amber-500' : 'border-gray-300'} rounded-lg text-sm focus:ring-2 focus:ring-blue-500 outline-none">
                    ${warn ? `<span data-warn class="text-xs text-amber-600">${warn}</span>` : ''}</label>`;
            }).join('');
            return wrap(`${topBar(COPY.dataTitle)}
                <div class="flex-1 overflow-y-auto bg-gray-50 p-4 space-y-3">
                    ${S.imgUrl ? `<div class="flex items-center gap-3 bg-white rounded-xl p-2"><img src="${S.imgUrl}" alt="Scanned card" class="w-24 rounded-lg"><p class="text-xs text-gray-500">What we read from this card is below. ${COPY.dataHint}</p></div>` : `<p class="text-xs text-gray-500">${COPY.dataHint}</p>`}
                    ${inputs}
                </div>
                <div class="p-4 bg-white border-t border-gray-200 space-y-2">${btn('analyze', 'Score this lead', 'primary')}${btn('retake', S.imgUrl ? 'Retake photo' : 'Use the camera')}</div>`, 'data');
        },

        analyzing: () => wrap(`${topBar('Lead score')}
            <div class="flex-1 flex flex-col items-center justify-center bg-gray-50 px-8 text-center">
                <div class="w-20 h-20 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-3xl"><i class="fas fa-brain fa-beat"></i></div>
                <h3 class="mt-5 text-lg font-semibold text-gray-900">${COPY.analyzing[0]}</h3>
                <p class="text-sm text-gray-600 max-w-xs">${COPY.analyzing[1]}</p>
                ${progress(['Details confirmed', 'Matching your profile', 'Writing the reasoning'], 1)}</div>`, 'analyzing'),

        analysis_failed: () => message('fa-exclamation-triangle', [COPY.analysisFailed[0], S.error || COPY.analysisFailed[1]],
            btn('analyze', 'Retry scoring', 'primary') + btn('edit', 'Edit details') + btn('save', 'Save without a score'), 'analysis_failed', 'text-amber-500', 'Lead score')
            + (S.saveError ? `<p data-save-error class="absolute bottom-2 inset-x-0 text-center text-xs text-red-600">${COPY.saveFailed}</p>` : ''),

        result: () => {
            const r = S.result, local = String(r.engine).startsWith('ollama');
            return wrap(`${topBar('Lead score', '', 'Close')}
                <div class="flex-1 overflow-y-auto bg-gray-50 p-4 space-y-3">
                    <div class="text-center"><p class="font-semibold text-gray-900">${esc(clean(S.fields.name))}</p><p class="text-sm text-gray-500">${esc(clean(S.fields.company_name))}</p></div>
                    ${ScoreCard.html({ score: r.score, temperature: r.temperature, reasoning: r.reasoning, breakdown: r.breakdown })}
                    <p data-engine class="text-xs text-gray-400 text-center">${local ? 'Written by a local model on this server.' : 'Local model offline: reasoning composed from your profile.'}</p>
                    ${S.saved ? '<p data-saved class="text-center text-sm text-green-600"><i class="fas fa-check-circle mr-1"></i>Saved to contacts</p>' : ''}
                    ${S.saveError ? `<p data-save-error class="text-center text-xs text-red-600">${COPY.saveFailed}</p>` : ''}
                </div>
                <div class="p-4 bg-white border-t border-gray-200 space-y-2">
                    ${S.saved ? btn('again', 'Scan another card', 'primary') + btn('close', 'Done') : btn('save', 'Save contact', 'primary') + btn('again', 'Scan another card') + btn('edit', 'Edit details')}
                </div>`, 'result');
        },
    };

    function render() {
        root.innerHTML = views[S.state]();
        if (S.state === 'camera') { video = root.querySelector('#scanFlowVideo'); if (stream) video.srcObject = stream; }
    }

    // ---------- actions ----------
    async function startCamera() {
        S.imgUrl && URL.revokeObjectURL(S.imgUrl);
        Object.assign(S, { blob: null, imgUrl: null, error: null, noCamera: false, result: null, saved: false, saveError: false });
        go('camera');
        try {
            if (!navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error('no api'), { name: 'NotFoundError' });
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            if (S.state === 'camera') video.srcObject = stream; else stopCamera();
        } catch (e) {
            go('permission_denied', { noCamera: e.name === 'NotFoundError' || e.name === 'NotReadableError' });
        }
    }

    function setImage(blob) {
        S.imgUrl && URL.revokeObjectURL(S.imgUrl);
        S.blob = blob; S.imgUrl = URL.createObjectURL(blob);
    }

    async function extract() {
        go('extracting', { error: null });
        try {
            const body = new FormData();
            body.append('file', S.blob, 'card.jpg');
            const resp = await api.http(`${CONFIG.API_BASE_URL}/scan/extract?user_id=${encodeURIComponent(uid())}`, { method: 'POST', headers: headers(), body });
            if (!resp.ok) return go('extract_failed', { error: await detail(resp, COPY.extractFailed[1]) });
            const { fields } = await resp.json();
            go('data', { fields: { ...Object.fromEntries(FIELDS.map(([k]) => [k, ''])), ...fields }, fromScan: true });
        } catch {
            go('extract_failed', { error: 'We could not reach the server. Check your connection and retry.' });
        }
    }

    async function analyze() {
        if (!clean(S.fields.name)) return render(); // data screen shows the "name needed" warning
        go('analyzing', { error: null });
        try {
            const resp = await api.http(`${CONFIG.API_BASE_URL}/scan/analyze`, {
                method: 'POST', headers: { ...headers(), 'Content-Type': 'application/json' },
                body: JSON.stringify({ contact: contact(), user_id: uid() }) });
            if (!resp.ok) return go('analysis_failed', { error: await detail(resp, COPY.analysisFailed[1]) });
            go('result', { result: await resp.json(), saveError: false });
        } catch {
            go('analysis_failed', { error: 'We could not reach the server. Your details are safe: retry when you are back online.' });
        }
    }

    async function save() {
        try {
            const resp = await api.http(`${CONFIG.API_BASE_URL}/add_contact/`, {
                method: 'POST', headers: { ...headers(), 'Content-Type': 'application/json' },
                body: JSON.stringify({ contact: contact(), user_id: uid() }) });
            if (!resp.ok) throw new Error('save');
            if (typeof loadContacts === 'function') loadContacts();
            if (S.state === 'analysis_failed') { close(); return typeof showToast === 'function' && showToast('Contact saved without a score'); }
            S.saved = true; S.saveError = false;
        } catch { S.saveError = true; }
        render();
    }

    const actions = {
        close, retryCamera: startCamera, retake: startCamera, retryExtract: extract, analyze, save,
        manual: () => go('data', { fields: Object.fromEntries(FIELDS.map(([k]) => [k, ''])), fromScan: false }),
        edit: () => go('data'),
        again: startCamera,
        gallery: () => S.fileInput.click(),
        shutter: () => {
            if (!video?.videoWidth) return;
            const c = Object.assign(document.createElement('canvas'), { width: video.videoWidth, height: video.videoHeight });
            c.getContext('2d').drawImage(video, 0, 0);
            go('capturing');
            c.toBlob(b => b ? (setImage(b), extract()) : go('extract_failed', { error: 'The photo could not be captured. Retake it or type the details yourself.' }), 'image/jpeg', 0.9);
        },
    };

    function open() {
        close();
        S = { state: 'camera', fields: {}, saved: false };
        root = Object.assign(document.createElement('div'), { id: 'scanFlow', className: 'fixed inset-0 z-[1000] bg-white' });
        root.setAttribute('role', 'dialog'); root.setAttribute('aria-label', COPY.cameraTitle);
        const file = Object.assign(document.createElement('input'), { type: 'file', accept: 'image/*', id: 'scanFlowFile', hidden: true });
        file.onchange = () => { const f = file.files[0]; file.value = ''; if (f) { setImage(f); extract(); } };
        root.addEventListener('click', e => { const a = e.target.closest('[data-act]')?.dataset.act; if (a) actions[a](); });
        root.addEventListener('change', e => { // keep the user's edits; re-render so the warnings update
            const k = e.target.dataset?.field; if (k) { S.fields[k] = e.target.value; render(); }
        });
        document.body.append(root, file);
        S.fileInput = file;
        startCamera();
    }

    function close() {
        stopCamera();
        S?.imgUrl && URL.revokeObjectURL(S.imgUrl);
        S?.fileInput?.remove();
        root?.remove(); root = null;
    }

    return { open, close, api, get state() { return S?.state; } };
})();
