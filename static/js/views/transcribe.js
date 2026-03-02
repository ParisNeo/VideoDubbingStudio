import { postFormData, getJSON, postJSON, formatTime } from '../api.js';
import { doSwitchView } from '../app.js';

let currentTaskId = null;
let currentWs = null;
let transcriptionResult = null;
let interimTaskData = null;
let currentTaskState = {}; 

let speakerMergeState = {
    selected: new Set(),
    groups: []
};

export function init() {
    console.log("Transcribe: init() called");
    
    // Expose functions to window for inline HTML handlers safely after definition
    window.startNewTranscription = startNewProject;
    window.openTranscribeMonitor = openMonitor;
    window.jumpToPhaseReview = jumpToPhaseReview;

    // Step 1: Upload Only
    const uploadOnlyBtn = document.getElementById('transcribe-upload-only-btn');
    if (uploadOnlyBtn) uploadOnlyBtn.addEventListener('click', handleInitialUpload);

    // Step 2: Start Processing (Configuration)
    const configForm = document.getElementById('transcribe-config-form');
    if (configForm) configForm.addEventListener('submit', handleTranscribeStart);

    const fileInput = document.getElementById('transcribe-file-input');
    if (fileInput) fileInput.addEventListener('change', updateFileDisplay);

    const diarizationToggle = document.getElementById('diarization-toggle');
    if (diarizationToggle) {
        diarizationToggle.addEventListener('change', (e) => {
            const translateGroup = document.getElementById('trans-translate-group');
            if (translateGroup) translateGroup.classList.toggle('hidden', !e.target.checked);
            
            // Re-render the chain dynamically to add/remove the diarization steps
            if (!currentTaskId || currentTaskState.phase === 'init') {
                renderStaticChain({ phase: 'init', status: 'pending' });
            }
        });
    }

    // Bind footer actions
    document.getElementById('copy-original-btn')?.addEventListener('click', () => copyToClipboard('transcribe-orig'));
    document.getElementById('export-md-btn')?.addEventListener('click', exportToMarkdown);
    document.getElementById('transcribe-new-project-btn')?.addEventListener('click', startNewTranscription);
}

export function onShow() {
    const activeView = document.querySelector('.view-section.active');
    if (activeView && activeView.id === 'view-transcribe' && !currentTaskId) {
        resetToUpload();
    }
}

function updateFileDisplay(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('transcribe-file-display').innerText = `Selected: ${file.name}`;
        document.getElementById('transcribe-drop-zone').style.borderColor = 'var(--success)';
        document.getElementById('transcribe-upload-only-btn').disabled = false;
        
        // Show empty timeline structure from the very beginning
        const timeline = document.getElementById('transcribe-timeline-header');
        if (timeline) timeline.classList.remove('hidden');
        
        currentTaskState = { phase: 'init', status: 'pending' };
        renderStaticChain(currentTaskState);
    }
}

async function handleInitialUpload() {
    const btn = document.getElementById('transcribe-upload-only-btn');
    const uploadForm = document.getElementById('transcribe-upload-form');
    
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';

    try {
        const formData = new FormData(uploadForm);
        formData.set('auto_start', 'false'); 

        const response = await postFormData('/transcribe/upload', formData);    
        if (response.task_id) {
            currentTaskId = response.task_id;
            
            // Move to Config Step
            showStep('transcribe-step-config');
            
            // Initialize Timeline
            currentTaskState = { phase: 'init', status: 'pending', task_id: currentTaskId };
            renderStaticChain(currentTaskState);
            
            connectWebSocket(currentTaskId);
        }
    } catch (err) {
        await window.uiAlert("Upload failed: " + err.message);
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-upload"></i> Upload & Continue';
    }
}

async function handleTranscribeStart(e) {
    e.preventDefault();
    if (!currentTaskId) return;

    const btn = document.getElementById('transcribe-start-btn');
    const originalContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-sync fa-spin"></i> Starting Workflow...';

    try {
        const formData = new FormData(e.target);
        const useDiarization = document.getElementById('diarization-toggle')?.checked || false;
        
        // Determine first computation phase based on configuration
        const firstPhase = useDiarization ? 'identifying' : 'transcribing';

        await postJSON(`/api/projects/${currentTaskId}/restart`, {
            from_phase: firstPhase, // Jump to computation immediately
            whisper_model: formData.get('whisper_model'),
            src_lang: formData.get('src_lang'),
            tgt_lang: formData.get('tgt_lang'),
            vad_threshold: formData.get('vad_threshold'),
            use_diarization: useDiarization
        });

        showStep('transcribe-step-processing');
    } catch (err) {
        await window.uiAlert("Failed to start: " + err.message);
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

    const timeline = document.getElementById('transcribe-timeline-header');
    if (timeline) {
        const hasFile = document.getElementById('transcribe-file-input')?.files?.length > 0;
        const isLanding = !currentTaskId && !hasFile;
        timeline.classList.toggle('hidden', isLanding);
    }
}

function renderStaticChain(task) {
    const container = document.getElementById('transcribe-granular-task-list');
    if (!container) return;
    
    // Determine if we are using diarization to show/hide those steps dynamically
    let useDiarization = false;
    if (task.phase === 'init' || task.status === 'pending') {
        useDiarization = document.getElementById('diarization-toggle')?.checked || false;
    } else {
        useDiarization = task.phase === 'identifying' || 
                         task.phase === 'awaiting_speaker_validation' || 
                         (task.speaker_config && Object.keys(task.speaker_config).length > 0);
    }
    
    const phases = [
        { id: 'init', name: '1. Project Setup', type: 'interaction' },
        { id: 'identifying', name: '2. Speaker ID', type: 'computation', skip: !useDiarization },
        { id: 'awaiting_speaker_validation', name: '3. Speaker Confirmation', type: 'interaction', skip: !useDiarization },
        { id: 'transcribing', name: '4. Transcription', type: 'computation' },
        { id: 'awaiting_transcription_review', name: '5. Final Review', type: 'interaction' },
        { id: 'complete', name: '6. Completed', type: 'interaction' }
    ].filter(p => !p.skip);

    // Re-index step numbers sequentially based on active phases
    phases.forEach((p, i) => {
        p.name = `${i + 1}. ${p.name.split('. ')[1]}`;
    });

    let ph = task.phase || 'init';
    const phaseMap = {};
    phases.forEach((p, i) => phaseMap[p.id] = i);
    
    let currentPhaseIdx = phaseMap[ph] !== undefined ? phaseMap[ph] : 0;
    if (task.status === 'completed' || task.status === 'done') currentPhaseIdx = phases.length - 1;

    let html = '';
    phases.forEach((phaseObj, idx) => {
        let statusClass = 'pending';
        let iconClass = phaseObj.type === 'interaction' ? 'fa-user-pen' : 'fa-microchip';
        
        if (idx < currentPhaseIdx) {
            statusClass = 'completed';
            iconClass = 'fa-check';
        } else if (idx === currentPhaseIdx) {
            statusClass += ' active-step';
            if (task.status === 'failed' || task.status === 'error') {
                statusClass = 'failed';
                iconClass = 'fa-times';
            } else if (task.status === 'paused') {
                statusClass = 'skipped';
                iconClass = 'fa-pause';
            } else if (phaseObj.id.includes('awaiting')) { 
                if (task.status === 'awaiting_validation' || task.phase === 'awaiting_transcription_review') {
                    statusClass = 'skipped'; 
                    iconClass = 'fa-exclamation';
                } else {
                    statusClass = 'running';
                    iconClass = 'fa-spinner fa-spin';
                }
            } else if (task.status === 'completed' || task.phase === 'complete') {
                statusClass = 'completed';
                iconClass = 'fa-check';
            } else {
                statusClass = 'running';
                iconClass = 'fa-spinner fa-spin';
            }
        }
        
        html += `
            <div id="task-node-${phaseObj.id}" class="task-node ${statusClass}" onclick="jumpToPhaseReview('${phaseObj.id}')">
                <div class="node-icon-wrapper">
                    <i class="node-icon fas ${iconClass}"></i>
                </div>
                <div class="node-content">
                    <div>
                        <span class="node-title">${phaseObj.name}</span>
                        ${idx === currentPhaseIdx && task.message ? `<span class="node-status-text">- ${task.message}</span>` : ''}
                    </div>
                </div>
            </div>
        `;
    });
    container.innerHTML = html;
}

export function jumpToPhaseReview(phaseId) {
    if (phaseId === 'init') {
        showStep('transcribe-step-config');
        return;
    }
    
    if (phaseId === 'awaiting_speaker_validation') {
        showSpeakerValidation(interimTaskData || currentTaskState);
    } else if (phaseId === 'awaiting_transcription_review' || phaseId === 'complete') {
        showResults(transcriptionResult || currentTaskState);
    } else {
        showStep('transcribe-step-processing');
    }
}

function connectWebSocket(taskId) {
    if (currentWs) currentWs.close();
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    currentWs = new WebSocket(`${proto}//${window.location.host}/transcribe/ws/${taskId}`);
    
    currentWs.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        console.log("Transcribe WS Event:", msg.type, msg.data);

        if (msg.type === 'progress') {
            currentTaskState.progress = msg.data.percent;
            currentTaskState.phase = msg.data.phase;
            currentTaskState.message = msg.data.message;
            updateProgress(msg.data.percent, msg.data.message);

            // Force the timeline to show if it was hidden
            const timeline = document.getElementById('transcribe-timeline-header');
            if (timeline) timeline.classList.remove('hidden');
            renderStaticChain(currentTaskState);
        } else if (msg.type === 'speaker_validation_ready' || msg.type === 'speakers_detected') {
            currentTaskState.phase = 'awaiting_speaker_validation';
            // Handle both data formats
            const config = msg.data.speaker_config || msg.data.speakers;
            currentTaskState.speaker_config = config;
            interimTaskData = { speakers: config }; 
            renderStaticChain(currentTaskState);
            showSpeakerValidation(interimTaskData);
        } else if (msg.type === 'transcription_ready' || msg.type === 'transcription_complete') {
            // If it's a review stage
            if (msg.data.phase === 'awaiting_transcription_review' || msg.type === 'transcription_ready') {
                currentTaskState.phase = 'awaiting_transcription_review';
            } else {
                currentTaskState.phase = 'complete';
                currentTaskState.status = 'completed';
            }
            transcriptionResult = msg.data;
            Object.assign(currentTaskState, msg.data);
            renderStaticChain(currentTaskState);
            showResults(msg.data);
        }
    };
}

function updateProgress(percent, message) {
    document.getElementById('trans-main-progress-bar').style.width = `${percent}%`;
    document.getElementById('trans-processing-percent').innerText = `${Math.round(percent)}%`;
    document.getElementById('trans-processing-status-text').innerText = message;
}

function showSpeakerValidation(data) {
    showStep('transcribe-step-speakers');
    const grid = document.getElementById('transcribe-speaker-grid');
    const speakers = data.speakers || {};
    
    grid.innerHTML = Object.entries(speakers).map(([id, info]) => `
        <div class="speaker-card" data-id="${id}">
            <div class="spk-header">
                <span class="avatar">${parseInt(id) + 1}</span>
                <input type="text" class="spk-name studio-input" value="${info.name}" placeholder="Name speaker...">
            </div>
            <audio controls class="studio-audio" src="${info.sample_path}"></audio>
            <label class="checkbox-label" style="margin-top:10px">
                <input type="checkbox" class="spk-include" checked>
                <span>Include in transcript</span>
            </label>
        </div>`).join('');

    document.getElementById('confirm-speakers-btn').onclick = () => {
        const config = {};
        document.querySelectorAll('#transcribe-speaker-grid .speaker-card').forEach(card => {
            const id = card.dataset.id;
            config[id] = {
                name: card.querySelector('.spk-name').value,
                action: card.querySelector('.spk-include').checked ? 'dub' : 'remove',
                include: card.querySelector('.spk-include').checked
            };
        });
        currentWs.send(JSON.stringify({ type: 'confirm_speakers', speakers: config }));
        showStep('transcribe-step-processing');
    };
}

function showResults(data) {
    showStep('transcribe-step-result');
    transcriptionResult = data;

    const renderSegments = (segments, isTranslation = false) => {
        // Sort chronologically
        const sortedSegs = [...segments].sort((a, b) => a.start - b.start);
        
        return sortedSegs.map((seg) => {
            const speakerName = data.speakers?.[seg.speaker_id]?.name || `Speaker ${seg.speaker_id + 1}`;
            const lookupIdx = seg.idx !== undefined ? seg.idx : segments.indexOf(seg);
            const timeStr = `${formatTime(seg.start)} - ${formatTime(seg.end)}`;
            
            return `
                <div class="segment-review-card chronological-card" data-idx="${lookupIdx}">
                    <div class="seg-header">
                        <span class="avatar sm">${seg.speaker_id + 1}</span>
                        <span class="spk-name">${speakerName}</span>
                        <span class="seg-time"><i class="fas fa-clock"></i> ${timeStr}</span>
                    </div>
                    <audio controls preload="none" class="studio-audio" src="/api/projects/${currentTaskId}/preview/${lookupIdx}" style="width:100%; height:40px; margin:8px 0; border-radius:6px;"></audio>
                    <textarea class="edit-segment-text" 
                        data-idx="${lookupIdx}" 
                        data-type="${isTranslation ? 'trans' : 'orig'}" style="width:100%; min-height:80px;">${isTranslation ? (seg.translated_text || '') : (seg.original_text || '')}</textarea>
                </div>`;
        }).join('');
    };

    const origList = document.getElementById('transcribe-orig');
    if (origList) {
        origList.parentElement.innerHTML = `<div class="review-list chronological-list">${renderSegments(data.segments || [])}</div>`;
    }

    const transContainer = document.getElementById('transcribe-trans-container');
    if (data.translated_text && transContainer) {
        transContainer.style.display = 'block';
        const transList = document.getElementById('transcribe-trans');
        if (transList) transList.parentElement.innerHTML = `<div class="review-list chronological-list">${renderSegments(data.segments || [], true)}</div>`;
    }
}

function resetToUpload() {
    currentTaskId = null;
    if (currentWs) currentWs.close();
    showStep('transcribe-step-upload');
    document.getElementById('transcribe-upload-form').reset();
    document.getElementById('transcribe-file-display').innerText = 'Click to browse or drag audio/video (MP4, MP3, WAV, etc.)';
    document.getElementById('transcribe-drop-zone').style.borderColor = 'var(--border)';
    document.getElementById('transcribe-upload-only-btn').disabled = true;
    
    // Completely hide the timeline for a clean start
    const timeline = document.getElementById('transcribe-timeline-header');
    if (timeline) timeline.classList.add('hidden');
    
    const container = document.getElementById('transcribe-granular-task-list');
    if (container) container.innerHTML = '<div class="empty-state">Waiting for workflow...</div>';
}

function backToDashboard() {
    currentTaskId = null; 
    doSwitchView('dashboard');
}

export function startNewProject() {
    resetToUpload();
}

export async function openMonitor(taskId) {
    currentTaskId = taskId;
    try {
        const task = await getJSON(`/api/projects/${taskId}`);
        currentTaskState = task;
        showStep(task.status === 'completed' ? 'transcribe-step-result' : 'transcribe-step-processing');
        renderStaticChain(task);
        connectWebSocket(taskId);
        
        if (task.status === 'completed') showResults(task);
        else if (task.status === 'awaiting_validation') showSpeakerValidation(task);
    } catch (e) {
        console.error("Monitor error:", e);
    }
}

function copyToClipboard(id) {
    const text = transcriptionResult?.original_text || "";
    navigator.clipboard.writeText(text);
    const btn = document.getElementById('copy-original-btn');
    if (btn) btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
}

function exportToMarkdown() {
    if (!transcriptionResult || !transcriptionResult.segments) return;
    
    const segments = transcriptionResult.segments;
    const speakerConfig = transcriptionResult.speakers || {};
    
    let md = `# Transcription Export\n\n`;
    md += `*Generated on ${new Date().toLocaleString()}*\n\n---\n\n`;

    // Group contiguous segments by speaker
    let lastSpeakerId = null;
    let groupedText = "";

    segments.sort((a, b) => a.start - b.start).forEach((seg, i) => {
        const sid = seg.speaker_id;
        const spkName = speakerConfig[sid]?.name || `Speaker ${parseInt(sid) + 1}`;
        const text = seg.original_text.trim();

        if (sid !== lastSpeakerId) {
            if (groupedText) md += groupedText + "\n\n";
            md += `**${spkName}**:\n`;
            groupedText = text;
        } else {
            groupedText += " " + text;
        }
        
        lastSpeakerId = sid;
        
        // Final segment handling
        if (i === segments.length - 1) {
            md += groupedText + "\n";
        }
    });

    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `organized_transcript_${currentTaskId}.md`;
    a.click();
}

function exportToJSON() {
    if (!transcriptionResult || !transcriptionResult.segments) return;
    
    const segments = transcriptionResult.segments;
    const speakerConfig = transcriptionResult.speakers || {};
    const grouped = [];
    let current = null;

    segments.sort((a, b) => a.start - b.start).forEach(seg => {
        if (!current || current.speaker_id !== seg.speaker_id) {
            current = {
                speaker_id: seg.speaker_id,
                speaker_name: speakerConfig[seg.speaker_id]?.name || `Speaker ${seg.speaker_id + 1}`,
                text: seg.original_text.trim(),
                start: seg.start,
                end: seg.end
            };
            grouped.push(current);
        } else {
            current.text += " " + seg.original_text.trim();
            current.end = seg.end;
        }
    });

    const blob = new Blob([JSON.stringify({ 
        task_id: currentTaskId,
        export_date: new Date().toISOString(),
        grouped_segments: grouped,
        raw_segments: segments 
    }, null, 2)], { type: 'application/json' });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcript_${currentTaskId}.json`;
    a.click();
}