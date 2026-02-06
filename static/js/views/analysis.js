import { postFormData } from '../api.js';

export function init() {
    const form = document.getElementById('analysis-form');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = e.target.querySelector('button');
        const output = document.getElementById('analysis-output');
        const content = document.getElementById('analysis-content');
        
        if (btn) {
            btn.disabled = true;
            btn.textContent = "Analyzing (this may take a while)...";
        }
        
        if (output) output.style.display = 'none';
        if (content) content.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        
        try {
            const res = await postFormData('/analysis/', new FormData(e.target));
            if (output) output.style.display = 'block';
            if (content) {
                // Try to render as markdown if available, otherwise plain text
                if (window.marked && res.analysis) {
                    content.innerHTML = marked.parse(res.analysis);
                } else {
                    content.innerText = res.analysis || "No analysis returned.";
                }
            }
        } catch (err) {
            if (content) content.innerText = "Analysis failed: " + err.message;
            if (output) output.style.display = 'block';
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = "Analyze";
            }
        }
    });
}
