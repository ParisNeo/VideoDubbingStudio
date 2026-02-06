import { postFormData } from '../api.js';

export function init() {
    const form = document.getElementById('transcribe-form');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = e.target.querySelector('button');
        const output = document.getElementById('transcribe-text-out');
        const resultBox = document.getElementById('transcribe-result');
        
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        }
        
        try {
            const res = await postFormData('/transcribe/', new FormData(e.target));
            if (resultBox) resultBox.style.display = 'block';
            if (output) output.value = res.result?.translated_text || res.result?.original_text || 'No result';
        } catch (err) {
            alert("Transcription failed: " + err.message);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = 'Transcribe';
            }
        }
    });

    const copyBtn = document.getElementById('copy-transcribe-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const txt = document.getElementById('transcribe-text-out');
            if (txt) {
                txt.select();
                document.execCommand('copy');
            }
        });
    }
}
