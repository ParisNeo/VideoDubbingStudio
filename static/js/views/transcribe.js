import { postFormData, getJSON, formatTime } from '../api.js';
import { doSwitchView } from '../app.js';

// Make functions globally available
window.backToDashboard = backToDashboard;
window.startNewTranscription = startNewTranscription;
window.openTranscribeMonitor = openTranscribeMonitor;

export async function openTranscribeMonitor(taskId) {
    currentTaskId = taskId;
    showStep('transcribe-step-processing');
    
    try {
        const task = await getJSON(`/transcribe/status/${taskId}`);
        if (task.status === 'completed') {
            showResults(task);
        } else if (task.status === 'awaiting_validation') {
            // If it's waiting for speakers, we need the interim data
            // For simplicity, we trigger the WS connection which handles the state sync
            connectWebSocket(taskId);
        } else {
            connectWebSocket(taskId);
        }
    } catch (e) {
        console.error("Failed to monitor task", e);
    }
}

// State for current task
let currentTaskId = null;
let currentWs = null;
let transcriptionResult = null;
let interimTaskData = null; // Store speaker info before confirmation

let speakerMergeState = {
    selected: new Set(),
    groups: []
};

export function init() {
    // Bind upload form
    const uploadForm = document.getElementById('transcribe-upload-form');
    if (uploadForm) uploadForm.addEventListener('submit', handleUpload);

    // Bind file input change
    const fileInput = document.getElementById('transcribe-file-input');
    if (fileInput) fileInput.addEventListener('change', updateFileDisplay);

    // Bind diarization toggle
    const diarizationToggle = document.getElementById('diarization-toggle');
    if (diarizationToggle) {
        diarizationToggle.addEventListener('change', (e) => {
            const langSection = document.getElementById('transcribe-lang-section');
            if (langSection) {
                langSection.style.display = e.target.checked ? 'block' : 'none';
            }
        });
    }

    // Bind copy buttons
    const copyOrigBtn = document.getElementById('copy-original-btn');
    if (copyOrigBtn) {
        copyOrigBtn.addEventListener('click', () => copyToClipboard('transcribe-orig'));
    }

    const copyTransBtn = document.getElementById('copy-translated-btn');
    if (copyTransBtn) {
        copyTransBtn.addEventListener('click', () => copyToClipboard('transcribe-trans'));
    }

    // Bind export buttons
    const exportMdBtn = document.getElementById('export-md-btn');
    if (exportMdBtn) {
        exportMdBtn.addEventListener('click', exportToMarkdown);
    }

    const exportJsonBtn = document.getElementById('export-json-btn');
    if (exportJsonBtn) {
        exportJsonBtn.addEventListener('click', exportToJSON);
    }

    // Bind new project button
    const newProjectBtn = document.getElementById('transcribe-new-project-btn');
    if (newProjectBtn) {
        newProjectBtn.addEventListener('click', startNewTranscription);
    }
}

export function onShow() {
    // If we're coming from the dashboard/modal, we usually want to start fresh
    // unless currentTaskId was explicitly set via openMonitor
    const activeView = document.querySelector('.view-section.active');
    if (activeView && activeView.id === 'view-transcribe' && !currentTaskId) {
        resetToUpload();
    }
}

function backToDashboard() {
    // Don't cleanup WS here, let it run in background
    // but clear the local view state
    currentTaskId = null; 
    doSwitchView('dashboard');
}

function cleanup() {
    if (currentWs) {
        try { currentWs.close(); } catch(e) {}
        currentWs = null;
    }
    currentTaskId = null;
    transcriptionResult = null;
    interimTaskData = null;
}

function startNewTranscription() {
    cleanup();
    resetToUpload();
}

function resetToUpload() {
    // Hide all steps except upload
    document.querySelectorAll('.transcribe-step').forEach(el => {
        el.classList.add('hidden');
        el.classList.remove('active');
    });
    
    const uploadStep = document.getElementById('transcribe-step-upload');
    if (uploadStep) {
        uploadStep.classList.remove('hidden');
        uploadStep.classList.add('active');
    }

    // Reset form
    const form = document.getElementById('transcribe-upload-form');
    if (form) form.reset();

    // Reset file display
    const fileDisplay = document.getElementById('transcribe-file-display');
    if (fileDisplay) {
        fileDisplay.innerText = 'Click to browse or drag audio/video file here';
    }

    const dropZone = document.getElementById('transcribe-drop-zone');
    if (dropZone) dropZone.style.borderColor = 'var(--border)';

    // Hide language section by default
    const langSection = document.getElementById('transcribe-lang-section');
    if (langSection) langSection.style.display = 'none';

    // Clear results
    const resultContainer = document.getElementById('transcribe-result-container');
    if (resultContainer) resultContainer.classList.add('hidden');
}

function updateFileDisplay(e) {
    const file = e.target.files[0];
    if (file) {
        const fileDisplay = document.getElementById('transcribe-file-display');
        const dropZone = document.getElementById('transcribe-drop-zone');
        if (fileDisplay) fileDisplay.innerText = `Selected: ${file.name}`;
        if (dropZone) dropZone.style.borderColor = 'var(--success)';
        
        // Enable submit button
        const submitBtn = document.getElementById('transcribe-start-btn');
        if (submitBtn) submitBtn.disabled = false;
    }
}

async function handleUpload(e) {
    e.preventDefault();
    
    const btn = document.getElementById('transcribe-start-btn');
    const originalContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';


    try {
            const formData = new FormData(e.target);
            
            // Pack Diarization boolean explicitly
            const useDiarization = document.getElementById('diarization-toggle')?.checked || false;
            formData.set('use_diarization', useDiarization ? 'true' : 'false');
            
            // Ensure VAD threshold and model are included (FormData usually does this automatically, 
            // but explicit set handles dynamic UI changes better)
            const vadVal = e.target.querySelector('[name="vad_threshold"]').value;
            const modelVal = e.target.querySelector('[name="whisper_model"]').value;
            formData.set('vad_threshold', vadVal);
            formData.set('whisper_model', modelVal);

            const response = await postFormData('/transcribe/upload', formData);    

        
        if (response.task_id) {
            currentTaskId = response.task_id;
            
            if (useDiarization) {
                // Show processing step for diarization
                showStep('transcribe-step-processing');
                connectWebSocket(currentTaskId);
            } else {
                // Simple transcription - show processing then poll
                showStep('transcribe-step-processing');
                pollForResult(currentTaskId);
            }
        } else {
            throw new Error('No task ID received');
        }
    } catch (err) {
        await window.uiAlert('Error starting transcription: ' + err.message, 'Upload Failed');
        btn.disabled = false;
        btn.innerHTML = originalContent;
    }
}

function showStep(stepId) {
    document.querySelectorAll('.transcribe-step').forEach(el => {
        el.classList.add('hidden');
        el.classList.remove('active');
    });
    
    const target = document.getElementById(stepId);
    if (target) {
        target.classList.remove('hidden');
        target.classList.add('active');
    }
}

function connectWebSocket(taskId) {
    if (currentWs) {
        currentWs.close();
    }
    
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Match the router prefix /transcribe/ws/{taskId}
    const wsUrl = `${proto}//${window.location.host}/transcribe/ws/${taskId}`;
    
    currentWs = new WebSocket(wsUrl);
    
    currentWs.onopen = () => {
        console.log('Transcription WebSocket connected');
    };
    
    currentWs.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            
            if (msg.type === 'progress') {
                updateProgress(msg.data.percent, msg.data.message);
            } else if (msg.type === 'speakers_detected') {
                // Show speaker validation
                showSpeakerValidation(msg.data);
            } else if (msg.type === 'transcription_complete') {
                transcriptionResult = msg.data;
                showResults(msg.data);
            } else if (msg.type === 'error') {
                showError(msg.data.message);
            }
        } catch (err) {
            console.error('WS message error:', err);
        }
    };
    
    currentWs.onerror = (err) => {
        console.error('WebSocket error:', err);
    };
    
    currentWs.onclose = () => {
        console.log('WebSocket closed');
    };
}

async function pollForResult(taskId) {
    const maxAttempts = 60;
    let attempts = 0;
    
    const poll = async () => {
        try {
            const result = await getJSON(`/transcribe/status/${taskId}`);
            
            if (result.status === 'completed') {
                transcriptionResult = result;
                showResults(result);
                return;
            } else if (result.status === 'failed') {
                showError(result.error_message || 'Transcription failed');
                return;
            }
            
            // Update progress
            const progress = Math.min((attempts / maxAttempts) * 100, 95);
            updateProgress(progress, result.message || 'Transcribing...');
            
            attempts++;
            if (attempts < maxAttempts) {
                setTimeout(poll, 2000);
            } else {
                showError('Transcription timeout - please check status later');
            }
        } catch (err) {
            console.error('Poll error:', err);
            attempts++;
            if (attempts < maxAttempts) {
                setTimeout(poll, 2000);
            }
        }
    };
    
    poll();
}

function updateProgress(percent, message) {
    const bar = document.getElementById('transcribe-progress-bar');
    const text = document.getElementById('transcribe-progress-text');
    const status = document.getElementById('transcribe-status-text');
    
    if (bar) bar.style.width = `${percent}%`;
    if (text) text.innerText = `${Math.round(percent)}%`;
    if (status) status.innerText = message;
}

function showSpeakerValidation(data) {
    showStep('transcribe-step-speakers');
    interimTaskData = data;
    resetMergeUI();
    renderSpeakerCards();
    setupMergeUI(Object.keys(data.speakers));

    const confirmBtn = document.getElementById('confirm-speakers-btn');
    confirmBtn.onclick = () => {
        const config = {};
        const mergeMap = {};
        speakerMergeState.groups.forEach(g => g.slice(1).forEach(id => mergeMap[id] = g[0]));

        document.querySelectorAll('#transcribe-speaker-grid .speaker-card').forEach(card => {
            const id = card.dataset.id;
            config[id] = {
                name: card.querySelector('.spk-name').value,
                include: card.querySelector('.spk-include').checked,
                merged_into: mergeMap[id] || null,
                action: mergeMap[id] ? 'remove' : (card.querySelector('.spk-include').checked ? 'dub' : 'remove')
            };
        });

        if (currentWs?.readyState === WebSocket.OPEN) {
            currentWs.send(JSON.stringify({ type: 'confirm_speakers', speakers: config }));
            showStep('transcribe-step-processing');
        }
    };
}

function renderSpeakerCards() {
    const grid = document.getElementById('transcribe-speaker-grid');
    const speakers = interimTaskData.speakers;
    grid.innerHTML = Object.entries(speakers).map(([id, info]) => {
        const mergedGroup = speakerMergeState.groups.find(g => g.ids.includes(id));
        const isMaster = mergedGroup && mergedGroup.ids[0] === id;
        const isMerged = mergedGroup && !isMaster;
        const displayName = isMaster ? mergedGroup.name : info.name;

        return `
            <div class="speaker-card ${isMaster ? 'merged-master' : ''} ${isMerged ? 'merged-into' : ''}" data-id="${id}">
                <div class="spk-header">
                    <span class="avatar">${parseInt(id) + 1}</span>
                    <input type="text" class="spk-name" value="${displayName}" ${isMerged ? 'disabled' : ''}>
                </div>
                <audio controls src="${info.sample_path}" style="width:100%"></audio>
                <label class="checkbox-label" style="margin-top:10px">
                    <input type="checkbox" class="spk-include" ${isMerged ? 'disabled' : 'checked'}>
                    <span>Include in transcript</span>
                </label>
            </div>`;
    }).join('');
}

function setupMergeUI(ids) {
    const container = document.getElementById('trans-merge-checkboxes');
    container.innerHTML = ids.map(id => `
        <label class="merge-checkbox" data-id="${id}">
            <input type="checkbox" value="${id}"><span>Spk ${parseInt(id)+1}</span>
        </label>`).join('');

    container.querySelectorAll('.merge-checkbox').forEach(el => {
        el.onclick = (e) => {
            e.preventDefault();
            const id = el.dataset.id;
            if (speakerMergeState.selected.has(id)) speakerMergeState.selected.delete(id);
            else speakerMergeState.selected.add(id);
            el.classList.toggle('checked');
            document.getElementById('trans-apply-merge-btn').disabled = speakerMergeState.selected.size < 2;
        };
    });

    document.getElementById('trans-apply-merge-btn').onclick = () => {
        const name = document.getElementById('trans-merge-name').value.trim() || `Identity ${speakerMergeState.groups.length + 1}`;
        speakerMergeState.groups.push({
            ids: Array.from(speakerMergeState.selected),
            name: name
        });
        document.getElementById('trans-merge-name').value = '';
        speakerMergeState.selected.clear();
        renderMergeGroups();
        renderSpeakerCards();
        setupMergeUI(ids); 
    };
}

function renderMergeGroups() {
    const display = document.getElementById('trans-merge-groups-display');
    const preview = document.getElementById('trans-merge-preview');
    preview.style.display = speakerMergeState.groups.length ? 'block' : 'none';
    display.innerHTML = speakerMergeState.groups.map((g, idx) => `
        <div class="merge-group-tag">
            <i class="fas fa-object-group"></i> ${g.map(id => 'Spk '+(parseInt(id)+1)).join(' + ')}
            <i class="fas fa-times remove-merge" onclick="removeTransMerge(${idx})"></i>
        </div>`).join('');
}

window.removeTransMerge = (idx) => {
    speakerMergeState.groups.splice(idx, 1);
    renderMergeGroups();
    renderSpeakerCards();
    setupMergeUI(Object.keys(interimTaskData.speakers));
};

function resetMergeUI() {
    speakerMergeState = { selected: new Set(), groups: [] };
    document.getElementById('trans-merge-preview').style.display = 'none';
}

function showResults(data) {
    showStep('transcribe-step-result');
    transcriptionResult = data;

    const renderSegments = (segments, isTranslation = false) => {
        return segments.map((seg, idx) => {
            const speakerName = data.speakers?.[seg.speaker_id]?.name || `Speaker ${seg.speaker_id + 1}`;
            return `
                <div class="segment-review-card">
                    <div class="seg-meta">
                        <span class="seg-speaker">
                            <i class="fas fa-comment-dots"></i> ${speakerName}
                        </span>
                        <span class="seg-time">${formatTime(seg.start)} — ${formatTime(seg.end)}</span>
                    </div>
                    <audio controls class="studio-audio" src="/api/projects/${currentTaskId}/preview/${idx}"></audio>
                    <textarea class="edit-segment-text" 
                        data-idx="${idx}" 
                        data-type="${isTranslation ? 'trans' : 'orig'}" 
                        placeholder="Transcription text...">${isTranslation ? (seg.translated_text || '') : (seg.original_text || '')}</textarea>
                </div>
            `;
        }).join('');
    };

    const origContainer = document.getElementById('transcribe-orig');
    const transContainer = document.getElementById('transcribe-trans-container');
    const transList = document.getElementById('transcribe-trans');

    // Turn textareas into scrollable containers for cards
    if (origContainer) {
        origContainer.parentElement.innerHTML = `<div id="transcribe-orig-list" class="review-list" style="max-height:60vh; overflow-y:auto;">${renderSegments(data.segments)}</div>`;
    }

    if (data.translated_text && transContainer) {
        transContainer.style.display = 'block';
        transList.parentElement.innerHTML = `<div id="transcribe-trans-list" class="review-list" style="max-height:60vh; overflow-y:auto;">${renderSegments(data.segments, true)}</div>`;
    }
}

function showError(message) {
    showStep('transcribe-step-result');
    
    const origText = document.getElementById('transcribe-orig');
    if (origText) {
        origText.value = `Error: ${message}`;
    }
}

function copyToClipboard(elementId) {
    const el = document.getElementById(elementId);
    if (el) {
        el.select();
        document.execCommand('copy');
        
        // Visual feedback
        const btn = document.querySelector(`button[onclick*="${elementId}"]`) || 
                    document.activeElement;
        if (btn) {
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            setTimeout(() => {
                btn.innerHTML = originalText;
            }, 2000);
        }
    }
}

function exportToMarkdown() {
    if (!transcriptionResult) return;
    
    // Sync UI edits back to data object before export
    document.querySelectorAll('.edit-segment-text').forEach(el => {
        const idx = el.dataset.idx;
        if (el.dataset.type === 'orig') transcriptionResult.segments[idx].original_text = el.value;
        else transcriptionResult.segments[idx].translated_text = el.value;
    });

    const data = transcriptionResult;
    const useDiarization = data.speakers && Object.keys(data.speakers).length > 0;
    
    let md = `# Transcription\n\n`;
    md += `**Date:** ${new Date().toLocaleString()}\n`;
    md += `**File:** ${data.filename || 'Unknown'}\n`;
    md += `**Duration:** ${data.duration ? formatTime(data.duration) : 'Unknown'}\n\n`;
    
    if (useDiarization && data.segments) {
        md += `## Speakers\n\n`;
        Object.entries(data.speakers || {}).forEach(([id, info]) => {
            md += `- **${info.name || 'Speaker ' + (parseInt(id) + 1)}**\n`;
        });
        md += `\n`;
        
        md += `## Transcript with Timestamps\n\n`;
        data.segments.forEach(seg => {
            const timeStr = formatTime(seg.start);
            const speakerName = data.speakers?.[seg.speaker_id]?.name || `Speaker ${(seg.speaker_id || 0) + 1}`;
            // Corrected: Use original_text or translated_text
            const content = seg.original_text || seg.text || '';
            md += `**[${timeStr}] ${speakerName}:** ${content}\n\n`;
        });
    } else {
        md += `## Full Text\n\n`;
        md += `${data.original_text || data.text || ''}\n\n`;
    }
    
    if (data.translated_text) {
        md += `## Translation\n\n`;
        md += `${data.translated_text}\n\n`;
    }
    
    // Download
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcription_${new Date().toISOString().slice(0, 10)}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function exportToJSON() {
    if (!transcriptionResult) return;
    
    const blob = new Blob([JSON.stringify(transcriptionResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcription_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}