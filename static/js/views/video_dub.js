import { postFormData, getJSON, postJSON, formatTime } from '../api.js';
import { doSwitchView } from '../app.js';

window.openMonitor = openMonitor;
window.restartFromTask = restartFromTask;
window.startNewProject = startNewProject;
window.jumpToPhaseReview = jumpToPhaseReview;
window.removeMergeGroup = removeMergeGroup;

let currentWs = null;
let currentTaskId = null;
let downloadedVideoPath = null;
let downloadedVideoFilename = null;  // Store filename for download
let currentTaskState = {};           // Single source of truth for UI state
let activeUIPhase = 'init';          // Tracks which phase UI is currently displayed
let lastServerPhase = null;          // Tracks last reported server phase to detect progress

// Speaker merge state
let speakerMergeState = {
    selected: new Set(),
    groups: [] // Array of arrays: [[0, 2], [1]] means 0&2 merged, 1 separate
};

// Settings persistence key
const SETTINGS_KEY = 'voicedub_project_settings';

// Default settings
const DEFAULT_SETTINGS = {
    src_lang: 'auto',
    tgt_lang: 'en',
    tts_engine: 'f5',
    separate_audio: false
};

export function init() {
    // 1. Bind Config Form
    const configForm = document.getElementById('config-form');
    if (configForm) configForm.addEventListener('submit', handleDubStart);

    // 2. Bind upload-only button
    const uploadOnlyBtn = document.getElementById('upload-only-btn');
    if (uploadOnlyBtn) uploadOnlyBtn.addEventListener('click', handleInitialUpload);
    
    // 3. Bind YouTube Download
    const ytBtn = document.getElementById('download-youtube-btn');
    if (ytBtn) ytBtn.addEventListener('click', handleYouTubeDownload);
    
    // 4. Bind File Input Change (visual feedback)
    const fileInput = document.getElementById('video-input');
    if (fileInput) fileInput.addEventListener('change', updateFileDisplay);

    // 5. Bind "New Project" button on result page
    const newProjectBtn = document.getElementById('result-new-project-btn');
    if (newProjectBtn) {
        newProjectBtn.addEventListener('click', (e) => {
            e.preventDefault();
            startNewProject();
        });
    }

    // 6. Bind merge controls
    const applyMergeBtn = document.getElementById('apply-merge-btn');
    if (applyMergeBtn) {
        applyMergeBtn.addEventListener('click', applySpeakerMerge);
    }

    // 7. Bind settings change listeners for persistence
    bindSettingsPersistence();
}

function bindSettingsPersistence() {
    const srcLang = document.getElementById('src-lang');
    const tgtLang = document.getElementById('tgt-lang');
    const ttsEngine = document.getElementById('tts-engine');
    const separateAudio = document.getElementById('separate-audio');

    const saveSettings = () => {
        const settings = {
            src_lang: srcLang ? srcLang.value : 'auto',
            tgt_lang: tgtLang ? tgtLang.value : 'en',
            tts_engine: ttsEngine ? ttsEngine.value : 'f5',
            separate_audio: separateAudio ? separateAudio.checked : false
        };
        
        try {
            localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
        } catch (e) {
            console.warn('Could not save settings:', e);
        }
    };

    if (srcLang) srcLang.addEventListener('change', saveSettings);
    if (tgtLang) tgtLang.addEventListener('change', saveSettings);
    if (ttsEngine) ttsEngine.addEventListener('change', saveSettings);
    if (separateAudio) separateAudio.addEventListener('change', saveSettings);
}

function loadSavedSettings() {
    try {
        const saved = localStorage.getItem(SETTINGS_KEY);
        if (!saved) return DEFAULT_SETTINGS;
        return { ...DEFAULT_SETTINGS, ...JSON.parse(saved) };
    } catch (e) {
        return DEFAULT_SETTINGS;
    }
}

function applySettingsToForm(settings) {
    const srcLang = document.getElementById('src-lang');
    const tgtLang = document.getElementById('tgt-lang');
    const ttsEngine = document.getElementById('tts-engine');
    const separateAudio = document.getElementById('separate-audio');

    if (srcLang && settings.src_lang) srcLang.value = settings.src_lang;
    if (tgtLang && settings.tgt_lang) tgtLang.value = settings.tgt_lang;
    if (ttsEngine && settings.tts_engine) ttsEngine.value = settings.tts_engine;
    if (separateAudio && settings.separate_audio !== undefined) {
        separateAudio.checked = settings.separate_audio;
    }
}

export function onShow() {
    if (currentTaskId) {
        loadAndMonitor(currentTaskId);
    } else {
        const settings = loadSavedSettings();
        applySettingsToForm(settings);
    }
}

export function startNewProject() {
    currentTaskId = null;
    downloadedVideoPath = null;
    currentTaskState = {};
    speakerMergeState = { selected: new Set(), groups: [] };
    
    if (currentWs) {
        currentWs.close();
        currentWs = null;
    }
    
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) uploadForm.reset();
    const configForm = document.getElementById('config-form');
    if (configForm) configForm.reset();
    
    document.getElementById('media-selection-section')?.classList.remove('hidden');
    document.getElementById('dub-config-section')?.classList.add('hidden');
    
    const fileDisplay = document.getElementById('file-display-text');
    if (fileDisplay) fileDisplay.innerText = 'Click to browse or drag video file here';
    
    const dropZone = document.getElementById('drop-zone');
    if (dropZone) dropZone.style.borderColor = 'var(--border)';
    
    const youtubeStatus = document.getElementById('youtube-status');
    if (youtubeStatus) {
        youtubeStatus.style.display = 'none';
        youtubeStatus.innerHTML = '';
    }
    
    const startBtn = document.getElementById('start-btn');
    if (startBtn) startBtn.disabled = false;
    
    const term = document.getElementById('processing-term');
    if (term) term.innerHTML = '';
    
    renderStaticChain({ phase: 'init', status: 'pending' });
    
    const progressBar = document.getElementById('main-progress-bar');
    if (progressBar) progressBar.style.width = '0%';
    
    const progressText = document.getElementById('processing-percent');
    if (progressText) progressText.innerText = '0%';
    
    const statusText = document.getElementById('processing-status-text');
    if (statusText) statusText.innerText = 'Initializing...';
    
    const finalVideo = document.getElementById('final-video');
    if (finalVideo) {
        finalVideo.src = '';
        finalVideo.load();
    }
    
    resetMergeUI();
    
    const settings = loadSavedSettings();
    applySettingsToForm(settings);
    
    showWizardStep('state-upload');
    
    const viewEl = document.getElementById('view-video_dub');
    if (viewEl && !viewEl.classList.contains('active')) {
        doSwitchView('video_dub');
    }
}

export function openMonitor(taskId) {
    const currentView = document.querySelector('.view-section.active');
    if (!currentView || currentView.id !== 'view-video_dub') {
        doSwitchView('video_dub');
    }
    loadAndMonitor(taskId);
}

function showWizardStep(stepId) {
    document.querySelectorAll('#stage-content-area .wizard-step').forEach(el => {
        el.classList.add('hidden');
        el.classList.remove('active');
    });
    
    const target = document.getElementById(stepId);
    if (target) {
        target.classList.remove('hidden');
        target.classList.add('active');
    }

    const timeline = document.getElementById('task-timeline-header');
    if (timeline) {
        const hasFile = document.getElementById('video-input')?.files?.length > 0;
        const isLanding = !currentTaskId && !hasFile;
        timeline.classList.toggle('hidden', isLanding);
    }
}

async function downloadYouTubeVideo() {
    if (!downloadedVideoPath) {
        await window.uiAlert('No video available to download', 'Error');
        return;
    }
    
    try {
        const downloadUrl = `/api/youtube/download-file?file_path=${encodeURIComponent(downloadedVideoPath)}`;
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = downloadedVideoFilename || 'youtube_video.mp4';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } catch (err) {
        await window.uiAlert('Download failed: ' + err.message, 'Error');
    }
}

async function handleYouTubeDownload() {
    const urlInput = document.getElementById('youtube-url');
    const statusEl = document.getElementById('youtube-status');
    const btn = document.getElementById('download-youtube-btn');
    
    const url = urlInput.value.trim();
    if (!url) return await window.uiAlert("Please enter a YouTube URL", "Input Required");

    try {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Fetching...';
        statusEl.style.display = 'block';
        statusEl.innerText = "Downloading video from YouTube... please wait.";
        statusEl.style.color = 'var(--text-muted)';
        
        const res = await postJSON('/api/youtube/download', { url });
        
        if (res.success) {
            downloadedVideoPath = res.file_path;
            downloadedVideoFilename = res.filename || res.title || 'youtube_video.mp4';
            statusEl.innerHTML = `
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                    <span>✅ Downloaded: ${res.title}</span>
                    <button type="button" id="download-youtube-video-btn" class="btn-secondary btn-sm">
                        <i class="fas fa-download"></i> Download Video
                    </button>
                </div>
            `;
            statusEl.style.color = 'var(--success)';
            
            const downloadBtn = document.getElementById('download-youtube-video-btn');
            if (downloadBtn) {
                downloadBtn.addEventListener('click', downloadYouTubeVideo);
            }
            
            document.getElementById('media-selection-section').classList.add('hidden');
            document.getElementById('dub-config-section').classList.remove('hidden');
            document.getElementById('config-filename').innerText = res.title;
            
            const uploadRes = await postJSON('/api/upload-youtube', {
                file_path: downloadedVideoPath,
                filename: downloadedVideoFilename,
                tgt_lang: document.getElementById('tgt-lang').value,
                src_lang: document.getElementById('src-lang').value,
                separate_audio: document.getElementById('separate-audio').checked,
                tts_engine: document.getElementById('tts-engine').value,
                auto_start: false 
            });
            currentTaskId = uploadRes.task_id;
            
            currentTaskState = { phase: 'init', status: 'pending' };
            renderStaticChain(currentTaskState);
            const timeline = document.getElementById('task-timeline-header');
            if (timeline) timeline.classList.remove('hidden');
            
            connectWebSocket(currentTaskId);
            showWizardStep('state-upload');
        } else {
            throw new Error(res.error || "Unknown error");
        }
    } catch (e) {
        statusEl.innerText = `❌ Error: ${e.message}`;
        statusEl.style.color = 'var(--danger)';
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-cloud-download-alt"></i> Fetch';
    }
}

function updateFileDisplay(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('file-display-text').innerText = `Selected: ${file.name}`;
        document.getElementById('drop-zone').style.borderColor = 'var(--accent)';
        
        const uploadBtn = document.getElementById('upload-only-btn');
        if (uploadBtn) uploadBtn.disabled = false;
        
        downloadedVideoPath = null; 
        const youtubeStatus = document.getElementById('youtube-status');
        if (youtubeStatus) youtubeStatus.style.display = 'none';
        
        const timeline = document.getElementById('task-timeline-header');
        if (timeline) timeline.classList.remove('hidden');
        renderStaticChain({ phase: 'init', status: 'pending' });
    }
}

async function handleInitialUpload() {
    const btn = document.getElementById('upload-only-btn');
    const originalContent = btn.innerHTML;
    const uploadForm = document.getElementById('upload-form');
    
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';

    try {
        const formData = new FormData(uploadForm);
        formData.set('auto_start', 'false');

        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        if (!res.ok) throw new Error("Upload failed");
        
        const data = await res.json();
        currentTaskId = data.task_id;
        
        document.getElementById('media-selection-section').classList.add('hidden');
        document.getElementById('dub-config-section').classList.remove('hidden');
        
        const fileInput = uploadForm.querySelector('input[type="file"]');
        if (fileInput && fileInput.files[0]) {
            document.getElementById('config-filename').innerText = fileInput.files[0].name;
        }
        
        const timeline = document.getElementById('task-timeline-header');
        if (timeline) timeline.classList.remove('hidden');
        
        currentTaskState = { phase: 'init', status: 'pending' };
        renderStaticChain(currentTaskState);
        
        connectWebSocket(currentTaskId);

    } catch (err) {
        await window.uiAlert("Upload failed: " + err.message);
        btn.disabled = false;
        btn.innerHTML = originalContent;
    }
}

async function handleDubStart(e) {
    e.preventDefault();
    if (!currentTaskId) return;

    const btn = document.getElementById('start-btn');
    const originalContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-sync fa-spin"></i> Initializing...';

    try {
        const formData = new FormData(e.target);
        showWizardStep('state-processing');
        logToTerm("Starting pipeline initialization...", "step");

        await postJSON(`/api/projects/${currentTaskId}/restart`, {
            from_phase: 'identifying',
            src_lang: formData.get('src_lang'),
            tgt_lang: formData.get('tgt_lang'),
            tts_engine: formData.get('tts_engine'),
            whisper_model: formData.get('whisper_model'),
            separate_audio: formData.get('separate_audio') === 'on' || formData.get('separate_audio') === 'true',
            vad_threshold: parseFloat(formData.get('vad_threshold'))
        });
    } catch (err) {
        showWizardStep('state-upload');
        document.getElementById('media-selection-section').classList.add('hidden');
        document.getElementById('dub-config-section').classList.remove('hidden');
        
        await window.uiAlert("Failed to start: " + err.message);
        btn.disabled = false;
        btn.innerHTML = originalContent;
    }
}

async function loadAndMonitor(taskId, retryCount = 0) {
    currentTaskId = taskId;
    try {
        const task = await getJSON(`/api/projects/${taskId}`);
        Object.assign(currentTaskState, task);
        renderTaskState(task);
        connectWebSocket(taskId);
    } catch(err) {
        if (err.message.includes('404') && retryCount < 5) {
            const delay = 400 * (retryCount + 1);
            setTimeout(() => loadAndMonitor(taskId, retryCount + 1), delay);
            return;
        }
        logToTerm(`Error loading task: ${err.message}`, 'error');
    }
}

function connectWebSocket(taskId) {
    if (currentWs) {
        if (currentWs.readyState === WebSocket.OPEN || currentWs.readyState === WebSocket.CONNECTING) {
            return;
        }
        try { currentWs.close(); } catch (e) {}
        currentWs = null;
    }
    
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    currentWs = new WebSocket(`${proto}//${window.location.host}/ws/${taskId}`);
    let reconnectScheduled = false;
    
    currentWs.onopen = () => reconnectScheduled = false;
    
    currentWs.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            
            if (msg.type === 'state_sync' || msg.type === 'status_update') {
                renderTaskState(msg.data);
            } else if (msg.type === 'progress') {
                currentTaskState.phase = msg.data.phase;
                currentTaskState.progress = msg.data.percent;
                currentTaskState.message = msg.data.message;
                if (msg.data.phase === 'complete') currentTaskState.status = 'completed';
                renderTaskState(currentTaskState);
            } else if (msg.type === 'log') {
                logToTerm(msg.data.message, msg.data.style);
            } else if (msg.type === 'speaker_validation_ready') {
                Object.assign(currentTaskState, msg.data);
                showSpeakerValidation(currentTaskState);
            } else if (msg.type === 'transcription_ready') {
                Object.assign(currentTaskState, msg.data);
                showTranscriptionReview(currentTaskState);
            } else if (msg.type === 'translation_ready') {
                Object.assign(currentTaskState, msg.data);
                showTranslationReview(currentTaskState.data || currentTaskState);
            } else if (msg.type === 'audio_ready') {
                Object.assign(currentTaskState, msg.data);
                showAudioValidation(currentTaskState);
            } else if (msg.type === 'transcription_validated') {
                Object.assign(currentTaskState, msg.data);
                logToTerm('Transcription validated, starting translation...', 'success');
                renderTaskState(currentTaskState);
            } else if (msg.type === 'translation_validated') {
                Object.assign(currentTaskState, msg.data);
                logToTerm('Translation validated, starting voice synthesis...', 'success');
                renderTaskState(currentTaskState);
            } else if (msg.type === 'validation_accepted') {
                showWizardStep('state-processing');
            }
        } catch (err) {
            console.error('Error processing WS message:', err);
        }
    };
    
    currentWs.onclose = (event) => {
        const isCompleted = document.getElementById('state-result')?.classList.contains('active');
        const isNormalClose = event.code === 1000 || event.code === 1001;
        
        if (!currentTaskId || isCompleted || reconnectScheduled || isNormalClose) return;
        
        reconnectScheduled = true;
        logToTerm('Connection lost - reconnecting in 3s...', 'warning');
        
        setTimeout(() => {
            reconnectScheduled = false;
            if (currentTaskId && currentWs?.readyState !== WebSocket.OPEN) {
                connectWebSocket(currentTaskId);
            }
        }, 3000);
    };
}

async function renderTaskState(task) {
    const phaseChanged = task.phase !== lastServerPhase;
    lastServerPhase = task.phase;

    Object.assign(currentTaskState, task);
    renderStaticChain(currentTaskState);

    if (task.phase === 'init' || task.status === 'awaiting_input') {
        showWizardStep('state-upload');
        document.getElementById('media-selection-section').classList.add('hidden');
        document.getElementById('dub-config-section').classList.remove('hidden');
        document.getElementById('config-filename').innerText = task.filename || task.input_filename || "Project Ready";
        
        if (task.src_lang) document.getElementById('src-lang').value = task.src_lang;
        if (task.tgt_lang) document.getElementById('tgt-lang').value = task.tgt_lang;
        if (task.tts_engine) document.getElementById('tts-engine').value = task.tts_engine;
        if (task.separate_audio !== undefined) document.getElementById('separate-audio').checked = !!task.separate_audio;
        if (task.whisper_model) document.getElementById('whisper-model').value = task.whisper_model;
        if (task.vad_threshold) {
            const vadInp = document.querySelector('input[name="vad_threshold"]');
            if (vadInp) vadInp.value = task.vad_threshold;
            const vadVal = document.getElementById('vad-val');
            if (vadVal) vadVal.innerText = `Saved (${task.vad_threshold})`;
        }
        return; 
    }

    const isValidationPhase = task.status === 'awaiting_validation' || task.phase?.includes('awaiting');
    const needsData = isValidationPhase && !task.segments && !task.transcribed_segments && !task.speaker_config;
    
    if (needsData && task.task_id) {
        try {
            const fullTask = await getJSON(`/api/projects/${task.task_id}`);
            Object.assign(currentTaskState, fullTask);
        } catch (e) { }
    }

    const progressBar = document.getElementById('main-progress-bar');
    const progressText = document.getElementById('processing-percent');
    const statusText = document.getElementById('processing-status-text');

    let pct = currentTaskState.progress || 0;
    if (currentTaskState.status === 'completed' || currentTaskState.phase === 'complete') pct = 100;

    if (progressBar) progressBar.style.width = `${pct}%`;
    if (progressText) progressText.innerText = `${pct}%`;
    if (statusText && currentTaskState.message) statusText.innerText = currentTaskState.message;

    const timeline = document.getElementById('task-timeline-header');
    if (timeline && currentTaskState.phase !== 'init' && currentTaskState.status !== 'pending') {
        timeline.classList.remove('hidden');
    }

    const phase = currentTaskState.phase;
    const status = currentTaskState.status;
    const hasSegments = (currentTaskState.segments && currentTaskState.segments.length > 0) || 
                        (currentTaskState.transcribed_segments && currentTaskState.transcribed_segments.length > 0);
    
    const isVisibleStep1 = document.getElementById('state-upload').classList.contains('active');
    if (isVisibleStep1 && (status === 'pending' || status === 'awaiting_input')) {
        document.getElementById('media-selection-section').classList.add('hidden');
        document.getElementById('dub-config-section').classList.remove('hidden');
        return; 
    }

    const interactionPhases = [
        'awaiting_speaker_validation', 
        'awaiting_transcription_review', 
        'awaiting_translation_review', 
        'awaiting_audio_validation',
        'complete'
    ];

    let targetPhaseUI = phase;

    // CRITICAL: If status is awaiting_validation, force UI to the correct review phase regardless of 'phase' string
    if (status === 'awaiting_validation' || interactionPhases.includes(phase)) {
        if (phase === 'awaiting_audio_validation' || (currentTaskState.segments && currentTaskState.segments.some(s => s.audio_path))) {
            targetPhaseUI = 'awaiting_audio_validation';
        } else if (phase === 'awaiting_translation_review' || (currentTaskState.segments && currentTaskState.segments.some(s => s.translated_text))) {
            targetPhaseUI = 'awaiting_translation_review';
        } else if (phase === 'awaiting_transcription_review' || (currentTaskState.segments && currentTaskState.segments.some(s => s.original_text))) {
            targetPhaseUI = 'awaiting_transcription_review';
        } else {
            targetPhaseUI = 'awaiting_speaker_validation';
        }
    } else {
        // If the server is in a computing phase, the UI should stay in 'processing' (Log View)
        const computingPhases = ['identifying', 'transcribing', 'translating', 'synthesizing', 'recomposing'];
        if (computingPhases.includes(phase)) {
            targetPhaseUI = 'processing';
        }
    }

    if (currentTaskState.status === 'completed' || phase === 'complete') {
        targetPhaseUI = 'complete';
    } else if (currentTaskState.status === 'failed' || currentTaskState.status === 'error') {
        logToTerm(`FAILED: ${currentTaskState.error_message || 'Unknown error'}`, 'error');
        if (statusText) {
            statusText.innerHTML = `<span style="color:var(--danger)">FAILED: ${currentTaskState.error_message || 'Unknown error'}</span>`;
        }
        targetPhaseUI = 'processing';
    }

    // Only auto-jump UI if the server phase has actually changed/progressed
    // or if we are currently in the 'processing' catch-all view
    if (targetPhaseUI !== 'processing') {
        if (phaseChanged || activeUIPhase === 'processing') {
            jumpToPhaseReview(targetPhaseUI, false);
        }
    } else {
        showWizardStep('state-processing');
        activeUIPhase = 'processing';
    }

    if (targetPhaseUI === 'complete' || currentTaskState.status === 'completed') {
        const finalVideo = document.getElementById('final-video');
        const downloadBtn = document.getElementById('download-btn');
        
        if (finalVideo && currentTaskState.output_path) {
            let webPath = currentTaskState.output_path.replace(/\\/g, '/');
            if (!webPath.startsWith('/')) webPath = '/' + webPath;
            const cacheBuster = `?t=${new Date().getTime()}`;
            finalVideo.src = webPath + cacheBuster;
            finalVideo.load();
        }
        
        if (downloadBtn) {
            downloadBtn.href = `/api/projects/${currentTaskId}/download`;
        }
    }
}

function renderStaticChain(task) {
    const container = document.getElementById('granular-task-list');
    if (!container) return;
    
    // Ensure we have an active UI phase set
    if (!activeUIPhase) activeUIPhase = task.phase || 'init';

    const phases = [
        { id: 'init', name: '1. Configuration', type: 'interaction' },
        { id: 'identifying', name: '2. Diarization', type: 'computation' },
        { id: 'awaiting_speaker_validation', name: '3. Speaker Validation', type: 'interaction' },
        { id: 'transcribing', name: '4. Transcription', type: 'computation' },
        { id: 'awaiting_transcription_review', name: '5. Transcript Validation', type: 'interaction' },
        { id: 'translating', name: '6. Translation', type: 'computation' },
        { id: 'awaiting_translation_review', name: '7. Translation Validation', type: 'interaction' },
        { id: 'synthesizing', name: '8. Audio Generation', type: 'computation' },
        { id: 'awaiting_audio_validation', name: '9. Audio Validation', type: 'interaction' },
        { id: 'recomposing', name: '10. Fusion (Assembly)', type: 'computation' },
        { id: 'complete', name: '11. Preview & Download', type: 'interaction' }
    ];

    let ph = task.phase || 'init';
    
    const phaseMap = {
        'init': 0, 'identifying': 1, 'awaiting_speaker_validation': 2,
        'transcribing': 3, 'awaiting_transcription_review': 4,
        'translating': 5, 'awaiting_translation_review': 6,
        'synthesizing': 7, 'awaiting_audio_validation': 8,
        'recomposing': 9, 'complete': 10
    };
    
    let currentPhaseIdx = phaseMap[ph] !== undefined ? phaseMap[ph] : 0;
    
    // Force highlight to the next validation step if the computation step is technically "active" but finished
    if (task.status === 'awaiting_validation') {
        if (ph === 'translating') currentPhaseIdx = 6;
        else if (ph === 'transcribing') currentPhaseIdx = 4;
        else if (ph === 'identifying') currentPhaseIdx = 2;
        else if (ph === 'synthesizing') currentPhaseIdx = 8;
    }
    
    if (task.status === 'completed' || task.status === 'done') currentPhaseIdx = 10;

    let html = '';
    phases.forEach((phaseObj, idx) => {
        let statusClass = 'pending';
        let iconClass = phaseObj.type === 'interaction' ? 'fa-user-pen' : 'fa-microchip';
        const nodeTypeClass = phaseObj.type === 'interaction' ? 'interaction-node' : 'compute-node';

        if (idx < currentPhaseIdx) {
            statusClass = 'completed';
            iconClass = 'fa-check';
        } else if (idx === currentPhaseIdx) {
            if (task.status === 'failed' || task.status === 'error') {
                statusClass = 'failed';
                iconClass = 'fa-times';
            } else if (task.status === 'paused') {
                statusClass = 'skipped';
                iconClass = 'fa-pause';
            } else if (idx === 2 || idx === 4 || idx === 6 || idx === 8) { 
                if (task.status === 'awaiting_validation' || task.phase.includes('awaiting')) {
                    statusClass = 'attention'; // Changed from skipped
                    iconClass = 'fa-pen-nib'; // Changed from exclamation
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
        
        // Apply classes for AI progress vs User focus
        if (phaseObj.id === activeUIPhase) {
            statusClass += ' user-viewing';
        }
        
        const showRestart = (statusClass === 'completed' || statusClass === 'failed' || statusClass.includes('active-step')) && 
                            ['init', 'identifying', 'transcribing', 'translating', 'synthesizing', 'recomposing'].includes(phaseObj.id);
        
        html += `
            <div id="task-node-${phaseObj.id}" class="task-node ${statusClass} ${nodeTypeClass}" onclick="jumpToPhaseReview('${phaseObj.id}', true)">
                <div class="node-icon-wrapper">
                    <i class="node-icon fas ${iconClass}"></i>
                </div>
                <div class="node-content">
                    <div>
                        <span class="node-title">${phaseObj.name}</span>
                        ${idx === currentPhaseIdx && task.message ? `<span class="node-status-text">- ${task.message}</span>` : ''}
                    </div>
                    ${showRestart ? `
                        <button class="restart-btn" onclick="event.stopPropagation(); restartFromTask('${phaseObj.id}')" title="Restart from here">
                            <i class="fas fa-redo-alt"></i>
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    });
    container.innerHTML = html;
}

/**
 * Force switches the UI to a specific phase, fetching data if necessary.
 * @param {string} phaseId - The ID of the phase to show
 * @param {boolean} isManual - If true, triggers the "Stop Task" confirmation logic
 */
async function jumpToPhaseReview(phaseId, isManual = false) {
    if (!currentTaskId) return;

    console.log(`${isManual ? 'Manual' : 'System'} Jump Requested: ${phaseId}`);

    // 1. Force state refresh to ensure we have the most recent data (speaker configs, etc)
    const isValidationPhase = phaseId.includes('awaiting_') || phaseId === 'init';
    if (isValidationPhase) {
        try {
            const freshTask = await getJSON(`/api/projects/${currentTaskId}`);
            Object.assign(currentTaskState, freshTask);
        } catch (e) {
            console.warn("Could not refresh task state for jump:", e);
        }
    }

    // 2. Update tracked UI phase so re-renders keep the highlight
    activeUIPhase = phaseId;

    // Only show "Stop Task" confirmation if the user clicked the timeline manually
    if (isManual) {
        const isProcessing = ['processing', 'queued', 'running'].includes(currentTaskState.status);
        
        if (isProcessing && currentTaskState.phase !== phaseId && phaseId !== 'complete') {
            const confirmed = await window.uiConfirm(
                `An AI process is currently running.\n\nJumping to another step will STOP the current process.\n\nDo you want to proceed?`,
                "Stop Current Task?"
            );
            if (!confirmed) return;
            
            try {
                await postJSON(`/api/projects/${currentTaskId}/cancel`, {});
                logToTerm(`Task stopped by user. Routing to ${phaseId} UI...`, "warning");
                currentTaskState.status = 'paused';
            } catch (e) {
                console.error("Stop failed:", e);
            }
        }
    }

    document.querySelectorAll('.task-node').forEach(n => n.classList.remove('active-step'));
    const node = document.getElementById(`task-node-${phaseId}`);
    if (node) node.classList.add('active-step');

    const hasSegments = (currentTaskState.segments?.length > 0) || (currentTaskState.transcribed_segments?.length > 0);

    // Update the "Fun" Stage Header
    const stageHeader = document.getElementById('dynamic-stage-header');
    if (stageHeader) {
        const isWaiting = currentTaskState.status === 'awaiting_validation' || phaseId.includes('awaiting');
        stageHeader.innerHTML = isWaiting 
            ? `<div class="stage-badge waiting"><i class="fas fa-hand-paper"></i> Waiting for your input</div>`
            : `<div class="stage-badge thinking"><i class="fas fa-brain fa-spin"></i> AI is thinking...</div>`;
    }

    // Ensure the timeline updates its visual highlight immediately
    renderStaticChain(currentTaskState);

    switch(phaseId) {
        case 'init':
            showWizardStep('state-upload');
            document.getElementById('media-selection-section').classList.add('hidden');
            document.getElementById('dub-config-section').classList.remove('hidden');
            break;

        case 'identifying':
        case 'awaiting_speaker_validation': 
            console.log("Routing to Speaker Validation UI");
            showWizardStep('state-validation');
            showSpeakerValidation(currentTaskState);
            break;

        case 'transcribing':
        case 'awaiting_transcription_review':
            console.log("Routing to Transcription Review UI");
            showWizardStep('state-validation');
            showTranscriptionReview(currentTaskState);
            break;

        case 'translating':
        case 'awaiting_translation_review':
            console.log("Routing to Translation Review UI");
            showWizardStep('state-validation');
            showTranslationReview({
                segments: currentTaskState.segments || currentTaskState.transcribed_segments,
                target_language: currentTaskState.tgt_lang || 'en',
                source_language: currentTaskState.src_lang || 'auto'
            });
            break;

        case 'synthesizing':
        case 'awaiting_audio_validation':
            console.log("Routing to Audio Validation UI");
            showWizardStep('state-validation');
            showAudioValidation(currentTaskState);
            break;

        case 'recomposing':
            showWizardStep('state-processing');
            break;

        case 'complete':
            showWizardStep('state-result');
            break;
            
        case 'awaiting_transcription_review':
            showWizardStep('state-validation');
            if (hasSegments) {
                showTranscriptionReview(currentTaskState);
            } else {
                document.getElementById('state-validation').innerHTML = '<div class="card"><div class="empty-state">No transcription data available yet. Please run the previous steps.</div></div>';
            }
            break;
            
        case 'awaiting_translation_review':
            showWizardStep('state-validation');
            if (hasSegments) {
                showTranslationReview({
                    segments: currentTaskState.segments || currentTaskState.transcribed_segments,
                    target_language: currentTaskState.tgt_lang || 'en',
                    source_language: currentTaskState.src_lang || 'auto'
                });
            } else {
                document.getElementById('state-validation').innerHTML = '<div class="card"><div class="empty-state">No translation data available yet. Please run the previous steps.</div></div>';
            }
            break;
            
        case 'awaiting_audio_validation':
            showWizardStep('state-validation');
            if (hasSegments) {
                showAudioValidation(currentTaskState);
            } else {
                document.getElementById('state-validation').innerHTML = '<div class="card"><div class="empty-state">No audio data available yet. Please run the previous steps.</div></div>';
            }
            break;
            
        case 'complete':
            if (currentTaskState.output_path) {
                showWizardStep('state-result');
            } else {
                showWizardStep('state-processing');
            }
            break;
            
        default:
            showWizardStep('state-processing');
            break;
    }
}

async function restartFromTask(taskName) {
    const phaseMap = {
        'init': 'init',
        'identifying': 'identifying',
        'transcribing': 'transcribing',
        'translating': 'translating', 
        'synthesizing': 'synthesizing',
        'recomposing': 'recomposing'
    };
    
    const backendPhase = phaseMap[taskName] || 'init';

    const confirmed = await window.uiConfirm(`Restart from '${taskName}'? This will re-run this step and all following steps.`, 'Restart Workflow');
    if (!confirmed) return;
    
    const term = document.getElementById('processing-term');
    if (term) term.innerHTML = '';
    
    renderStaticChain({ phase: backendPhase, status: 'pending' });
    
    const progressBar = document.getElementById('main-progress-bar');
    if (progressBar) progressBar.style.width = '0%';
    
    const progressText = document.getElementById('processing-percent');
    if (progressText) progressText.innerText = '0%';
    
    const statusText = document.getElementById('processing-status-text');
    if (statusText) statusText.innerText = 'Restarting...';
    
    try {
        await postJSON(`/api/projects/${currentTaskId}/restart`, { from_phase: backendPhase });
        
        setTimeout(() => {
            if (currentTaskId) loadAndMonitor(currentTaskId);
        }, 500);
        
    } catch (e) {
        await window.uiAlert("Restart failed: " + e.message, "Error");
        logToTerm(`Restart failed: ${e.message}`, 'error');
    }
}

function logToTerm(msg, style='info') {
    const term = document.getElementById('processing-term');
    if (!term) return;
    
    const div = document.createElement('div');
    div.className = `log-line ${style}`;
    
    const time = new Date().toLocaleTimeString();
    div.innerText = `[${time}] ${msg}`;
    
    term.appendChild(div);
    term.scrollTop = term.scrollHeight;
    
    while (term.children.length > 500) {
        term.removeChild(term.firstChild);
    }
}

function resetMergeUI() {
    speakerMergeState = { selected: new Set(), groups: [] };
    const checkboxes = document.getElementById('merge-checkboxes');
    const preview = document.getElementById('merge-preview');
    const groupsDisplay = document.getElementById('merge-groups-display');
    const status = document.getElementById('merge-status');
    const applyBtn = document.getElementById('apply-merge-btn');
    
    if (checkboxes) checkboxes.innerHTML = '';
    if (preview) preview.style.display = 'none';
    if (groupsDisplay) groupsDisplay.innerHTML = '';
    if (status) status.innerText = '';
    if (applyBtn) {
        applyBtn.disabled = true;
        applyBtn.innerHTML = '<i class="fas fa-code-branch"></i> Merge Selected';
    }
}

function setupMergeUI(speakerIds) {
    resetMergeUI();
    
    const checkboxes = document.getElementById('merge-checkboxes');
    const status = document.getElementById('merge-status');
    if (!checkboxes) return;
    
    speakerIds.forEach(id => {
        const label = document.createElement('label');
        label.className = 'merge-checkbox';
        label.dataset.id = id;
        
        label.innerHTML = `
            <input type="checkbox" value="${id}">
            <span>Speaker ${parseInt(id) + 1}</span>
        `;
        
        label.addEventListener('click', (e) => {
            e.preventDefault();
            const cb = label.querySelector('input');
            cb.checked = !cb.checked;
            label.classList.toggle('checked', cb.checked);
            toggleSpeakerSelection(id);
        });
        
        checkboxes.appendChild(label);
    });
    
    if (status) {
        status.innerText = `${speakerIds.length} speaker${speakerIds.length !== 1 ? 's' : ''} detected`;
    }
}

function toggleSpeakerSelection(speakerId) {
    if (speakerMergeState.selected.has(speakerId)) {
        speakerMergeState.selected.delete(speakerId);
    } else {
        speakerMergeState.selected.add(speakerId);
    }
    
    const applyBtn = document.getElementById('apply-merge-btn');
    if (applyBtn) {
        applyBtn.disabled = speakerMergeState.selected.size < 2;
    }
    
    updateMergeStatus();
}

function updateMergeStatus() {
    const status = document.getElementById('merge-status');
    if (!status) return;
    
    const count = speakerMergeState.selected.size;
    if (count === 0) {
        status.innerText = 'Select speakers to merge';
    } else if (count === 1) {
        status.innerText = 'Select at least one more speaker';
    } else {
        status.innerText = `${count} speakers selected for merge`;
    }
}

function applySpeakerMerge() {
    const selected = Array.from(speakerMergeState.selected);
    const groupName = document.getElementById('merge-group-name').value.trim() || `Merged Group ${speakerMergeState.groups.length + 1}`;
    
    if (selected.length < 2) return;
    
    speakerMergeState.groups.push({
        ids: [...selected],
        name: groupName
    });
    
    document.getElementById('merge-group-name').value = '';
    speakerMergeState.selected.clear();
    
    document.querySelectorAll('.merge-checkbox').forEach(cb => {
        cb.classList.remove('checked');
        cb.querySelector('input').checked = false;
    });
    
    const applyBtn = document.getElementById('apply-merge-btn');
    if (applyBtn) applyBtn.disabled = true;
    
    updateMergeStatus();
    renderMergeGroups();
    updateSpeakerCardsForMerge();
}

function renderMergeGroups() {
    const preview = document.getElementById('merge-preview');
    const groupsDisplay = document.getElementById('merge-groups-display');
    if (!preview || !groupsDisplay) return;
    
    if (speakerMergeState.groups.length === 0) {
        preview.style.display = 'none';
        return;
    }
    
    preview.style.display = 'block';
    groupsDisplay.innerHTML = '';
    
    speakerMergeState.groups.forEach((group, idx) => {
        const tag = document.createElement('div');
        tag.className = 'merge-group-tag';
        
        const idChain = group.ids.map(id => `Spk ${parseInt(id) + 1}`).join(' + ');
        tag.innerHTML = `
            <i class="fas fa-object-group"></i>
            <b>${group.name}</b> <small style="opacity:0.8; margin-left:5px">(${idChain})</small>
            <span class="remove-merge" onclick="removeMergeGroup(${idx})" title="Unmerge" style="margin-left:8px; cursor:pointer;">
                <i class="fas fa-times"></i>
            </span>
        `;
        
        groupsDisplay.appendChild(tag);
    });
}

function removeMergeGroup(groupIdx) {
    speakerMergeState.groups.splice(groupIdx, 1);
    renderMergeGroups();
    updateSpeakerCardsForMerge();
}

function updateSpeakerCardsForMerge() {
    document.querySelectorAll('.speaker-card').forEach(card => {
        card.classList.remove('merged-master', 'merged-into');
        card.style.position = '';
    });
    
    speakerMergeState.groups.forEach(group => {
        const masterId = group.ids[0];
        const masterCard = document.querySelector(`.speaker-card[data-id="${masterId}"]`);
        if (masterCard) masterCard.classList.add('merged-master');
        
        group.ids.slice(1).forEach(id => {
            const card = document.querySelector(`.speaker-card[data-id="${id}"]`);
            if (card) {
                card.classList.add('merged-into');
                card.style.position = 'relative';
            }
        });
    });
}

// 1. Speaker Validation UI
function showSpeakerValidation(data) {
    showWizardStep('state-validation');
    const header = document.querySelector('#state-validation h3');
    if (header) header.innerHTML = '<i class="fas fa-user-check"></i> Step 1: Confirm Speakers';
    
    const mergeSection = document.querySelector('.merge-section');
    const subtitleText = document.querySelector('#state-validation > .subtitle');
    if (mergeSection) mergeSection.style.display = 'block';
    if (subtitleText) subtitleText.style.display = 'block';
    
    const grid = document.getElementById('speaker-grid');
    if(!grid) return;
    
    grid.className = 'speaker-grid';
    const config = data.speaker_config || data.data?.speaker_config || {};
    const entries = Object.entries(config);

    if (entries.length === 0) {
        grid.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-user-slash"></i>
                <p>No speakers detected in the audio.</p>
                <button class="btn-secondary" onclick="restartFromTask('init')">Try Again</button>
            </div>`;
        return;
    }

    grid.innerHTML = entries.map(([id, info]) => {
        const mergedGroup = speakerMergeState.groups.find(g => g.ids.includes(id));
        const isMaster = mergedGroup && mergedGroup.ids[0] === id;
        const isMerged = mergedGroup && !isMaster;
        
        return `
        <div class="speaker-card ${isMaster ? 'merged-master' : ''} ${isMerged ? 'merged-into' : ''}" 
             data-id="${id}" style="${isMerged ? 'position: relative;' : ''}">
            <div class="spk-header">
                <span class="avatar">${parseInt(id) + 1}</span>
                <label class="checkbox-label">
                    <input type="checkbox" class="spk-toggle" ${isMerged ? 'disabled' : 'checked'}>
                    <span class="checkmark"></span>
                    <span class="checkbox-text">
                        <span><i class="fas fa-volume-up"></i> Dub This Speaker</span>
                        <span class="help-text">${isMerged ? 'Merged with Spk ' + (parseInt(mergedGroup.ids[0]) + 1) : 'Voice will be cloned'}</span>
                    </span>
                </label>
            </div>
            <div class="spk-stats" style="display:flex; gap:12px; margin-bottom:8px; font-size:0.8rem; color:var(--text-muted);">
                <span><i class="fas fa-clock"></i> ${info.total_duration || 0}s total</span>
                <span><i class="fas fa-comment"></i> ${info.intervention_count || 0} segments</span>
            </div>
            <div class="spk-controls">
                <input type="text" class="spk-name" value="${info.name || 'Speaker '+ (parseInt(id)+1)}" placeholder="Speaker Name" ${isMerged ? 'disabled' : ''}>
            </div>
            <audio controls src="${info.sample_path}" style="width:100%; margin-top:10px;"></audio>
        </div>`;
    }).join('');

    setupMergeUI(Object.keys(config));
    updateSpeakerCardsForMerge();

    const actions = document.querySelector('#state-validation .header-actions');
    actions.innerHTML = `
        <button id="save-speakers-btn" class="btn-secondary">
            <i class="fas fa-save"></i> Save Progress
        </button>
        <button id="confirm-speakers-btn" class="btn-success">
            <i class="fas fa-check"></i> Confirm & Transcribe
        </button>
    `;
    document.getElementById('confirm-speakers-btn').onclick = submitSpeakerValidation;
    document.getElementById('save-speakers-btn').onclick = () => saveStepProgress('speakers');
}

async function submitSpeakerValidation() {
    // Optimistic UI update: Switch to processing view immediately
    // Set to 'processing' to prevent WS updates from jumping back to validation
    activeUIPhase = 'processing';
    showWizardStep('state-processing');
    renderStaticChain(currentTaskState);

    const speakers = {};
    const mergeMap = {};
    
    speakerMergeState.groups.forEach(group => {
        const master = group.ids[0];
        group.ids.slice(1).forEach(id => mergeMap[id] = master);
    });
    
    document.querySelectorAll('.speaker-card').forEach(card => {
        const id = card.dataset.id;
        const name = card.querySelector('.spk-name').value;
        const isDub = card.querySelector('.spk-toggle').checked;
        const mergedInto = mergeMap[id];
        
        const merged_speakers = [];
        speakerMergeState.groups.forEach(group => {
            if (group.ids[0] === id) merged_speakers.push(...group.ids.slice(1));
        });

        speakers[id] = {
            name: name,
            action: mergedInto ? 'remove' : (isDub ? 'dub' : 'remove'),
            merged_into: mergedInto,
            merged_speakers: merged_speakers
        };
    });
    
    try {
        await postJSON(`/api/projects/${currentTaskId}/validate-speakers`, { speakers });
    } catch (e) {
        // Fallback if network fails
        activeUIPhase = 'awaiting_speaker_validation';
        showWizardStep('state-validation');
        await window.uiAlert("Validation failed: " + e.message, "Error");
    }
}

/**
 * Common helper to extract all edited text from the current validation UI.
 */
function getSegmentsFromUI(selector = '.segment-edit-row', textAreaSelector = '.seg-original') {
    const segments = [];
    document.querySelectorAll(selector).forEach(row => {
        segments.push({
            idx: parseInt(row.dataset.idx),
            speaker_id: parseInt(row.dataset.speaker || row.dataset.speakerId),
            start: parseFloat(row.dataset.start),
            end: parseFloat(row.dataset.end),
            original_text: row.querySelector(textAreaSelector).value,
            translated_text: row.querySelector('.seg-translated')?.value || ""
        });
    });
    return segments;
}

/**
 * Generic save function that updates segments without triggering next phase
 */
async function saveTranscriptionProgress() {
    const segments = getSegmentsFromUI();
    const btn = document.getElementById('save-transcription-btn');
    const oldHtml = btn.innerHTML;
    
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';

    try {
        // Use the validation endpoint but with a flag to not proceed
        await postJSON(`/api/projects/${currentTaskId}/validate-transcription`, { 
            segments, 
            save_only: true 
        });
        
        // Update local state so it doesn't revert
        currentTaskState.segments = segments;
        
        btn.innerHTML = '<i class="fas fa-check"></i> Saved!';
        btn.classList.add('btn-success');
        
        setTimeout(() => {
            btn.innerHTML = oldHtml;
            btn.disabled = false;
            btn.classList.remove('btn-success');
        }, 2000);
    } catch (e) {
        btn.innerHTML = '<i class="fas fa-times"></i> Error';
        await window.uiAlert("Save failed: " + e.message);
        btn.disabled = false;
    }
}

// 2. Transcription Review UI (Chronological)
function showTranscriptionReview(data) {
    showWizardStep('state-validation');
    
    const mergeSection = document.querySelector('.merge-section');
    const step1Subtitle = document.querySelector('#state-validation > .subtitle');
    if (mergeSection) mergeSection.style.display = 'none';
    if (step1Subtitle) step1Subtitle.style.display = 'none';
    
    const header = document.querySelector('#state-validation h3');
    if (header) header.innerHTML = '<i class="fas fa-file-alt"></i> Step 2: Review Transcription';
    
    const grid = document.getElementById('speaker-grid');
    if (!grid) return;

    const actions = document.querySelector('#state-validation .header-actions');
    if (actions) actions.innerHTML = '';

    grid.className = 'review-list-container';
    grid.innerHTML = '<div class="empty-state"><i class="fas fa-spinner fa-spin"></i> Preparing transcript...</div>';
    
    let segments = data.transcribed_segments || data.segments || data.data?.transcribed_segments || data.data?.segments || [];
    let speakerConfig = data.speaker_config || data.data?.speaker_config || {};

    if (typeof segments === 'string') {
        try { segments = JSON.parse(segments); } catch(e) {}
    }
    if (typeof speakerConfig === 'string') {
        try { speakerConfig = JSON.parse(speakerConfig); } catch(e) {}
    }

    if (!Array.isArray(segments) || segments.length === 0) {
        grid.innerHTML = '<div class="empty-state">No transcription segments found.</div>';
        return;
    }
    
    segments.sort((a, b) => a.start - b.start);
    
    let html = '<div class="transcription-review chronological-list">';
    
    segments.forEach((seg, arrayIdx) => {
        const sid = seg.speaker_id;
        let spkName = `Speaker ${parseInt(sid) + 1}`;
        if (speakerConfig && (speakerConfig[sid] || speakerConfig[parseInt(sid)])) {
            const info = speakerConfig[sid] || speakerConfig[parseInt(sid)];
            if (info.name) spkName = info.name;
        }

        const lookupIdx = seg.idx !== undefined ? seg.idx : arrayIdx;
        const timeStr = formatTime(seg.start) + ' - ' + formatTime(seg.end);
        
        html += `
            <div class="segment-edit-row chronological-card" data-idx="${lookupIdx}" data-speaker="${sid}" data-start="${seg.start}" data-end="${seg.end}">
                <div class="seg-header">
                    <span class="avatar sm">${parseInt(sid) + 1}</span>
                    <span class="spk-name">${spkName}</span>
                    <span class="seg-time"><i class="fas fa-clock"></i> ${timeStr}</span>
                </div>
                <audio controls preload="none" src="/api/projects/${currentTaskId}/preview/${lookupIdx}" style="width:100%; height:40px; margin-bottom:6px; border-radius: 6px;"></audio>
                <textarea class="seg-original" style="width:100%; min-height:80px;">${seg.original_text || ''}</textarea>
            </div>
        `;
    });
    
    html += '</div>';
    grid.innerHTML = html;
    
    if (actions) {
        actions.innerHTML = `
            <button id="save-transcription-btn" class="btn-secondary">
                <i class="fas fa-save"></i> Save Progress
            </button>
            <button id="confirm-transcription-btn" class="btn-success">
                <i class="fas fa-language"></i> Confirm & Translate
            </button>
        `;
        document.getElementById('confirm-transcription-btn').onclick = submitTranscriptionReview;
        document.getElementById('save-transcription-btn').onclick = () => saveStepProgress('transcription');
    }
}

async function submitTranscriptionReview() {
    const segments = getSegmentsFromUI();

    // Optimistic UI update: Switch to processing view immediately
    activeUIPhase = 'processing';
    showWizardStep('state-processing');
    renderStaticChain(currentTaskState);
    
    try {
        await postJSON(`/api/projects/${currentTaskId}/validate-transcription`, { segments });
    } catch (e) {
        activeUIPhase = 'awaiting_transcription_review';
        showWizardStep('state-validation');
        await window.uiAlert("Failed to submit transcription: " + e.message);
    }
}

// 3. Translation Review UI (Chronological)
function showTranslationReview(data) {
    showWizardStep('state-validation');
    
    const mergeSection = document.querySelector('.merge-section');
    const subtitleText = document.querySelector('#state-validation > .subtitle');
    if (mergeSection) mergeSection.style.display = 'none';
    if (subtitleText) subtitleText.style.display = 'none';
    
    const header = document.querySelector('#state-validation h3');
    if (header) {
        header.innerHTML = '<i class="fas fa-language"></i> Step 3: Review Translations';
    }
    
    const grid = document.getElementById('speaker-grid');
    if (!grid) return;
    
    grid.className = 'review-list-container';
    
    // Fallback chain for segments data
    let segments = data.segments || data.translation_segments || data.transcribed_segments || [];
    const tgtLang = data.target_language || data.tgt_lang || 'en';
    const srcLang = data.source_language || 'auto';
    
    segments.sort((a, b) => a.start - b.start);
    const speakerConfig = currentTaskState.speaker_config || {};
    
    let html = '<div class="translation-review chronological-list">';
    html += `<p class="subtitle" style="margin-bottom:20px;">Review and edit translations side-by-side. ${srcLang !== 'auto' ? `Translating from ${srcLang} to ${tgtLang}` : `Auto-detect to ${tgtLang}`}</p>`;
    
    segments.forEach((seg) => {
        const sid = seg.speaker_id;
        let spkName = `Speaker ${parseInt(sid) + 1}`;
        if (speakerConfig && (speakerConfig[sid] || speakerConfig[parseInt(sid)])) {
            const info = speakerConfig[sid] || speakerConfig[parseInt(sid)];
            if (info.name) spkName = info.name;
        }

        const timeStr = formatTime(seg.start) + ' - ' + formatTime(seg.end);
        
        html += `
            <div class="translation-edit-row chronological-card" data-idx="${seg.idx}">
                <div class="seg-header">
                    <span class="avatar sm">${parseInt(sid) + 1}</span>
                    <span class="spk-name">${spkName}</span>
                    <span class="seg-time"><i class="fas fa-clock"></i> ${timeStr}</span>
                </div>
                <audio controls preload="none" src="/api/projects/${currentTaskId}/preview/${seg.idx}" style="width:100%; height:40px; margin-bottom:8px; border-radius: 6px;"></audio>
                
                <div class="side-by-side-container" style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                    <div class="original-column">
                        <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:4px;">
                            <i class="fas fa-file-alt"></i> Original:
                        </label>
                        <div class="original-text" style="padding:10px; background:var(--bg-panel); border-radius:4px; color:var(--text-main); font-style:italic; min-height:80px; line-height:1.5;">
                            ${seg.original_text || '[No text]'}
                        </div>
                    </div>
                    <div class="translation-column">
                        <label style="display:block; font-size:0.8rem; color:var(--accent); margin-bottom:4px;">
                            <i class="fas fa-language"></i> Translation (${tgtLang}):
                        </label>
                        <textarea class="seg-translated" data-idx="${seg.idx}" 
                            style="width:100%; background:var(--bg-panel); border:1px solid var(--border); color:var(--text-main); padding:10px; border-radius:4px; font-family:inherit; resize:vertical; min-height:80px; line-height:1.5;">${seg.translated_text || ''}</textarea>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    grid.innerHTML = html;
    
    const headerActions = document.querySelector('#state-validation .header-actions');
    if (headerActions) {
        headerActions.innerHTML = `
            <button onclick="document.getElementById('state-validation').classList.remove('active'); document.getElementById('state-processing').classList.add('active');" class="btn-icon" title="View Logs">
                <i class="fas fa-terminal"></i>
            </button>
            <button id="save-translation-btn" class="btn-secondary">
                <i class="fas fa-save"></i> Save Progress
            </button>
            <button id="retry-translation-btn" class="btn-secondary">
                <i class="fas fa-redo"></i> Retry Translation
            </button>
            <button id="start-dub-btn" class="btn-success">
                <i class="fas fa-microphone"></i> Generate Voices
            </button>
        `;
        document.getElementById('start-dub-btn').addEventListener('click', submitTranslationReview);
        document.getElementById('save-translation-btn').onclick = () => saveStepProgress('translation');
        document.getElementById('retry-translation-btn').addEventListener('click', () => {
            restartFromTask('translating');
        });
    }
}

/**
 * Generic save function for any validation step
 */
async function saveStepProgress(stepType) {
    let endpoint = '';
    let payload = {};
    let btnId = '';

    if (stepType === 'speakers') {
        endpoint = `/api/projects/${currentTaskId}/validate-speakers`;
        btnId = 'save-speakers-btn';
        const speakers = {};
        document.querySelectorAll('.speaker-card').forEach(card => {
            const id = card.dataset.id;
            speakers[id] = {
                name: card.querySelector('.spk-name').value,
                action: card.querySelector('.spk-toggle').checked ? 'dub' : 'remove'
            };
        });
        payload = { speakers, save_only: true };
    } else if (stepType === 'transcription') {
        endpoint = `/api/projects/${currentTaskId}/validate-transcription`;
        btnId = 'save-transcription-btn';
        payload = { segments: getSegmentsFromUI(), save_only: true };
    } else if (stepType === 'translation') {
        endpoint = `/api/projects/${currentTaskId}/validate-translation`;
        btnId = 'save-translation-btn';
        const edited = [];
        document.querySelectorAll('.translation-edit-row').forEach(row => {
            edited.push({
                idx: parseInt(row.dataset.idx),
                original_text: row.querySelector('.original-text').textContent.trim(),
                translated_text: row.querySelector('.seg-translated').value
            });
        });
        payload = { segments: edited, save_only: true };
    }

    const btn = document.getElementById(btnId);
    const oldHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';

    try {
        await postJSON(endpoint, payload);
        btn.innerHTML = '<i class="fas fa-check"></i> Saved';
        btn.classList.add('btn-success');
        setTimeout(() => {
            btn.innerHTML = oldHtml;
            btn.disabled = false;
            btn.classList.remove('btn-success');
        }, 2000);
    } catch (e) {
        btn.innerHTML = '<i class="fas fa-times"></i> Error';
        btn.disabled = false;
        await window.uiAlert("Save failed: " + e.message);
    }
}

async function submitTranslationReview() {
    // Optimistic UI update: Switch to processing view immediately
    activeUIPhase = 'processing';
    showWizardStep('state-processing');
    renderStaticChain(currentTaskState);

    const edited_segments = [];
    document.querySelectorAll('.translation-edit-row').forEach(row => {
        edited_segments.push({
            idx: parseInt(row.dataset.idx),
            original_text: row.querySelector('.original-text').textContent.trim(),
            translated_text: row.querySelector('.seg-translated').value
        });
    });
    
    const btn = document.getElementById('start-dub-btn');
    if (!btn) return;
    
    const originalContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting Voice Synthesis...';

    try {
        await postJSON(`/api/projects/${currentTaskId}/validate-translation`, { 
            segments: edited_segments,
            edited_segments: edited_segments,
            proceed_to_tts: true
        });
    } catch(err) {
        activeUIPhase = 'awaiting_translation_review';
        showWizardStep('state-validation');
        await window.uiAlert("Error: " + err.message, 'Validation Failed');
        btn.disabled = false;
        btn.innerHTML = originalContent;
    }
}

// 4. Audio Validation
function showAudioValidation(data) {
    showWizardStep('state-validation');
    document.querySelector('.merge-section').style.display = 'none';
    
    const header = document.querySelector('#state-validation h3');
    header.innerHTML = '<i class="fas fa-volume-up"></i> Step 4: Review Audio';
    
    const grid = document.getElementById('speaker-grid');
    grid.innerHTML = `
        <div style="text-align:center; padding:20px;">
            <p>Audio segments generated. You can listen to the preview below.</p>
            <div style="max-height:400px; overflow-y:auto; text-align:left;">
                ${(data.segments || []).map(s => `
                    <div style="padding:10px; border-bottom:1px solid var(--border);">
                        <div>Segment ${s.idx + 1} (${formatTime(s.start)}-${formatTime(s.end)})</div>
                        <audio controls src="/api/projects/${currentTaskId}/preview/${s.idx}" style="width:100%; margin-top:5px;"></audio>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    const actions = document.querySelector('#state-validation .header-actions');
    actions.innerHTML = `
        <button id="confirm-audio-btn" class="btn-success">
            <i class="fas fa-film"></i> Confirm & Build Video
        </button>
    `;
    document.getElementById('confirm-audio-btn').onclick = async () => {
        await postJSON(`/api/projects/${currentTaskId}/validate-audio`, {});
        currentTaskState.phase = 'recomposing';
        showWizardStep('state-processing');
    };
}