/* ScoreCard: the one place score + reason is drawn. The 1B feed row and the scan result must both call this
   so the two surfaces agree. Colours match the existing contact badge (hot red, warm orange, cold gray).
   ScoreCard.html({score, temperature, reasoning, breakdown?}) -> string. */
window.ScoreCard = (() => {
    const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const TONE = { hot: ['bg-red-500', 'Hot lead'], warm: ['bg-orange-400', 'Warm lead'], cold: ['bg-gray-400', 'Cold lead'] };
    const LABEL = { profile_fit: 'Fits what you sell', role_relevance: 'Role match', company_fit: 'Industry match', engagement: 'Reachability', timing: 'Event timing' };

    function html({ score, temperature, reasoning, breakdown }) {
        const [bg, label] = TONE[temperature] || TONE.cold;
        const bars = Object.entries(breakdown || {}).map(([k, v]) => `
            <div class="flex items-center gap-2 text-xs text-gray-600">
                <span class="w-32 shrink-0">${esc(LABEL[k] || k)}</span>
                <div class="flex-1 h-1.5 bg-gray-200 rounded-full"><div class="h-1.5 rounded-full ${bg}" style="width:${Math.min(100, v * 5)}%"></div></div>
                <span class="w-8 text-right">${esc(v)}/20</span>
            </div>`).join('');
        return `
            <div data-score-card class="bg-white rounded-2xl shadow-sm p-5">
                <div class="flex items-center gap-4">
                    <div data-score-value class="w-20 h-20 rounded-full ${bg} text-white flex items-center justify-center text-3xl font-bold">${esc(score)}</div>
                    <div><p class="text-xs uppercase tracking-wide text-gray-500">Score out of 100</p>
                    <p class="text-lg font-semibold text-gray-900">${label}</p></div>
                </div>
                <p data-score-reason class="mt-4 text-sm text-gray-700 leading-relaxed">${esc(reasoning)}</p>
                ${bars ? `<div class="mt-4 space-y-1.5">${bars}</div>` : ''}
            </div>`;
    }
    return { html };
})();
