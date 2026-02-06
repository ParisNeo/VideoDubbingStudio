import { postFormData, getJSON, postJSON } from '../api.js';
import { doSwitchView } from '../app.js';

// Make openMonitor globally available
window.openMonitor = openMonitor;

// State for current project
let currentWs = null;
let currentTaskId = null;
let downloadedVideoPath = null;
let downloadedVideoName = null;
// Track if we've submitted validation to prevent flickering
let validationSubmitted = false;
// Track the expected next state to filter out stale WebSocket messages
let expectedState = null;
// Store transcription/translation segments for display
let transcriptionSegments = [];

export function init() {
    console.log('VideoDub: init() called');
    
    // YouTube download button
    const youtubeBtn = document.getElementById('download-youtube-btn');
    if (youtubeBtn) {
        youtubeBtn.addEventListener('click', handleYouTubeDownload);
    }
    
    // YouTube URL input - enable/disable submit based on input
    const youtubeInput = document.getElementById('youtube-url');
    if (youtubeInput) {
        youtubeInput.addEventListener('input', updateSubmitButtonState);
    }
    
    // Upload form
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleUpload);
        console.log('VideoDub: upload form bound');
    }
    
    // File input visual feedback
    const fileInput = document.getElementById('video-input');
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            const display = document.getElementById('file-display-text');
            if (file && display) {
                display.innerHTML = `<i class="fas fa-check"></i> ${file.name} (${formatFileSize(file.size)})`;
                display.classList.add('file-selected');
                // Clear any downloaded video when file is selected
                downloadedVideoPath = null;
                downloadedVideoName = null;
                hideDownloadOriginalButton();
                updateSubmitButtonState();
            }
        });
    }
    
    // Cancel/close project button
    const cancelBtn = document.getElementById('cancel-project-btn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', closeProject);
    }
    
    // Start dubbing button (from validation screen)
    const startBtn = document.getElementById('start-dub-btn');
    if (startBtn) {
        startBtn.addEventListener('click', startTranslationPhase);
    }
    
    // Download original YouTube video button
    const downloadOriginalBtn = document.getElementById('download-original-btn');
    if (downloadOriginalBtn) {
        downloadOriginalBtn.addEventListener('click', downloadOriginalVideo);
    }
}

export function onShow() {
    console.log('VideoDub: onShow() called');
    // If we have a current task, ensure we're connected
    if (currentTaskId && !currentWs) {
        connectWebSocket(currentTaskId);
    }
}

export function openMonitor(taskId) {
    console.log(`VideoDub: openMonitor(${taskId})`);
    // Switch to video_dub view using the imported function
    doSwitchView('video_dub');
    
    // Load task and connect
    loadAndMonitor(taskId);
}

function updateSubmitButtonState() {
    const fileInput = document.getElementById('video-input');
    const submitBtn = document.getElementById('upload-submit-btn');
    
    const hasFile = fileInput && fileInput.files && fileInput.files.length > 0;
    const hasDownloadedVideo = downloadedVideoPath !== null;
    
    // Enable submit if we have a file or a downloaded video
    const canSubmit = hasFile || hasDownloadedVideo;
    
    if (submitBtn) {
        submitBtn.disabled = !canSubmit;
    }
}

function showDownloadOriginalButton() {
    const btn = document.getElementById('download-original-btn');
    if (btn) {
        btn.classList.remove('hidden');
    }
}

function hideDownloadOriginalButton() {
    const btn = document.getElementById('download-original-btn');
    if (btn) {
        btn.classList.add('hidden');
    }
}

async function downloadOriginalVideo() {
    if (!downloadedVideoPath) {
        alert('No downloaded video available');
        return;
    }
    
    try {
        // Show loading state on button
        const btn = document.getElementById('download-original-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Downloading...';
        btn.disabled = true;
        
        // Construct download URL with file path as query parameter
        const params = new URLSearchParams({ file_path: downloadedVideoPath });
        const downloadUrl = `/api/youtube/download-file?${params.toString()}`;
        
        // Create a temporary anchor element to trigger download
        const a = document.createElement('a');
        a.href = downloadUrl;
        // Use the stored filename, fallback to extracting from path
        a.download = downloadedVideoName || downloadedVideoPath.split('/').pop() || 'youtube_video.mp4';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        // Restore button state
        btn.innerHTML = originalText;
        btn.disabled = false;
        
    } catch (err) {
        console.error('Failed to download original video:', err);
        alert('Failed to download video: ' + err.message);
        
        // Restore button state on error
        const btn = document.getElementById('download-original-btn');
        if (btn) {
            btn.innerHTML = '<i class="fas fa-download"></i> Download Original';
            btn.disabled = false;
        }
    }
}

async function handleYouTubeDownload() {
    const urlInput = document.getElementById('youtube-url');
    const progressDiv = document.getElementById('youtube-progress');
    const progressFill = document.getElementById('youtube-progress-fill');
    const statusText = document.getElementById('youtube-status');
    const errorDiv = document.getElementById('youtube-error');
    const downloadBtn = document.getElementById('download-youtube-btn');
    
    const url = urlInput.value.trim();
    if (!url) {
        showYouTubeError('Please enter a YouTube URL');
        return;
    }
    
    // Validate URL looks like YouTube
    if (!isValidYouTubeUrl(url)) {
        showYouTubeError('Please enter a valid YouTube URL');
        return;
    }
    
    // Reset UI
    errorDiv.classList.add('hidden');
    errorDiv.textContent = '';
    progressDiv.classList.remove('hidden');
    progressFill.style.width = '0%';
    statusText.textContent = 'Starting download...';
    downloadBtn.disabled = true;
    
    try {
        // Simulate progress updates
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            progressFill.style.width = `${progress}%`;
        }, 500);
        
        // Call API to download video
        const response = await postJSON('/api/youtube/download', {
            url: url,
            quality: 'best'
        });
        
        clearInterval(progressInterval);
        
        if (response.success && response.file_path) {
            progressFill.style.width = '100%';
            statusText.textContent = 'Download complete!';
            
            // Store the downloaded video info
            downloadedVideoPath = response.file_path;
            downloadedVideoName = response.filename || 'youtube_video.mp4';
            
            // Update the UI to show the downloaded video
            const fileDisplay = document.getElementById('file-display-text');
            if (fileDisplay) {
                fileDisplay.innerHTML = `<i class="fas fa-check-circle" style="color: var(--success);"></i> ${downloadedVideoName} (downloaded from YouTube)`;
                fileDisplay.classList.add('file-selected');
            }
            
            // Clear the file input since we're using the downloaded video
            const fileInput = document.getElementById('video-input');
            if (fileInput) fileInput.value = '';
            
            // Show the download original button
            showDownloadOriginalButton();
            
            // Enable the submit button
            updateSubmitButtonState();
            
            // Hide progress after a delay
            setTimeout(() => {
                progressDiv.classList.add('hidden');
                downloadBtn.disabled = false;
                downloadBtn.innerHTML = '<i class="fas fa-check"></i> Downloaded';
            }, 1500);
            
        } else {
            throw new Error(response.error || 'Download failed');
        }
        
    } catch (err) {
        console.error('YouTube download failed:', err);
        showYouTubeError('Download failed: ' + err.message);
        progressDiv.classList.add('hidden');
        downloadBtn.disabled = false;
        downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download';
        hideDownloadOriginalButton();
    }
}

function isValidYouTubeUrl(url) {
    const patterns = [
        /^https?:\/\/(www\.)?youtube\.com\/watch\?v=[\w-]+/,
        /^https?:\/\/(www\.)?youtube\.com\/shorts\/[\w-]+/,
        /^https?:\/\/(www\.)?youtu\.be\/[\w-]+/,
        /^https?:\/\/(www\.)?youtube\.com\/live\/[\w-]+/
    ];
    return patterns.some(pattern => pattern.test(url));
}

function showYouTubeError(message) {
    const errorDiv = document.getElementById('youtube-error');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
    }
}

function formatFileSize(bytes) {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

async function handleUpload(e) {
    e.preventDefault();
    const btn = e.target.querySelector('button[type="submit"]');
    const originalText = btn?.innerHTML;
    
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    }
    
    try {
        const formData = new FormData(e.target);
        
        // Add new configuration fields to form data
        // src_lang is already in the form
        // tgt_lang is already in the form
        // separate_audio checkbox - need to ensure it's included
        const separateAudioCheckbox = document.getElementById('separate-audio');
        if (separateAudioCheckbox && separateAudioCheckbox.checked) {
            formData.set('separate_audio', 'true');
        } else {
            formData.set('separate_audio', 'false');
        }
        // tts_engine is already in the form
        
        let response;
        
        // Check if we have a downloaded video to use
        if (downloadedVideoPath) {
            console.log('Using downloaded YouTube video:', downloadedVideoPath);
            
            // Gather configuration values
            const srcLang = document.getElementById('src-lang')?.value || 'auto';
            const tgtLang = document.getElementById('tgt-lang')?.value || 'en';
            const ttsEngine = document.getElementById('tts-engine')?.value || 'f5';
            const separateAudio = document.getElementById('separate-audio')?.checked || false;
            
            // Send the downloaded video path to the server with all config
            response = await postJSON('/api/upload-youtube', {
                file_path: downloadedVideoPath,
                filename: downloadedVideoName,
                tgt_lang: tgtLang,
                src_lang: srcLang,
                separate_audio: separateAudio,
                tts_engine: ttsEngine
            });
        } else {
            // Regular file upload - formData already contains all fields
            console.log('Uploading video file with config:', {
                src_lang: formData.get('src_lang'),
                tgt_lang: formData.get('tgt_lang'),
                separate_audio: formData.get('separate_audio'),
                tts_engine: formData.get('tts_engine')
            });
            response = await postFormData('/api/upload', formData);
        }
        
        if (response.task_id) {
            console.log(`Upload successful, task: ${response.task_id}`);
            // Reset state flags for new task
            validationSubmitted = false;
            expectedState = null;
            transcriptionSegments = [];
            // Switch to processing state and connect
            showWizardStep('state-processing');
            loadAndMonitor(response.task_id);
        } else {
            throw new Error('No task_id received from server');
        }
        
    } catch (err) {
        console.error('Upload failed:', err);
        alert('Upload failed: ' + err.message);
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    }
}

async function loadAndMonitor(taskId) {
    try {
        console.log(`Loading task: ${taskId}`);
        const task = await getJSON(`/api/projects/${taskId}`);
        currentTaskId = taskId;
        
        // Restore transcription segments if available
        if (task.transcribed_segments) {
            transcriptionSegments = task.transcribed_segments;
        }
        
        // Update UI
        const controls = document.getElementById('project-controls');
        const projectName = document.getElementById('current-project-name');
        if (controls) controls.style.display = 'flex';
        if (projectName) projectName.textContent = task.original_filename || 'Untitled';
        
        // Connect WebSocket for live updates
        connectWebSocket(taskId);
        
        // Render current state
        renderTaskState(task);
        
    } catch (err) {
        console.error('Failed to load task:', err);
        alert('Could not load project: ' + err.message);
    }
}

function connectWebSocket(taskId) {
    // Close existing
    if (currentWs) {
        currentWs.close();
    }
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${taskId}`;
    
    console.log(`Connecting WebSocket: ${wsUrl}`);
    currentWs = new WebSocket(wsUrl);
    
    currentWs.onopen = () => {
        console.log('WebSocket connected for task:', taskId);
    };
    
    currentWs.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        console.log('WS message:', msg.type, msg);
        handleWebSocketMessage(msg);
    };
    
    currentWs.onerror = (err) => {
        console.error('WebSocket error:', err);
    };
    
    currentWs.onclose = () => {
        console.log('WebSocket closed');
        currentWs = null;
        // Attempt reconnect after delay if still on this task
        setTimeout(() => {
            if (currentTaskId === taskId) {
                connectWebSocket(taskId);
            }
        }, 3000);
    };
}

function handleWebSocketMessage(msg) {
    switch (msg.type) {
        case 'state_sync':
        case 'status_update':
            handleStatusUpdate(msg.data);
            break;
            
        case 'progress':
            // Handle both msg.data structure and direct msg properties
            const progress = msg.data?.percent || msg.percent;
            const message = msg.data?.message || msg.message;
            updateProgress(progress, message);
            break;
            
        case 'log':
            addLogMessage(msg.data?.message, msg.data?.style);
            break;
            
        case 'transcription_update':
            // NEW: Received transcription with diarization immediately after extraction
            console.log('Received transcription_update with', msg.data?.segments?.length, 'segments');
            transcriptionSegments = msg.data.segments || [];
            renderTranscriptionTable();
            break;
            
        case 'translation_progress':
            // NEW: Received progressive translation updates
            console.log('Received translation_progress, updating UI');
            updateTranslationProgress(msg.data?.segments || []);
            break;
            
        case 'speaker_samples':
            // Received speaker samples for validation - show validation screen
            console.log('Received speaker_samples, rendering validation UI');
            // Clear flags since we're entering validation state
            validationSubmitted = false;
            expectedState = null;
            renderSpeakerSamples(msg.data);
            showWizardStep('state-validation');
            break;
            
        case 'validation_accepted':
            // Server confirmed validation was accepted - transition to processing
            console.log('Server confirmed validation_accepted');
            validationSubmitted = false;
            expectedState = 'translating';
            showWizardStep('state-processing');
            updateProgress(35, 'Starting translation pipeline...');
            break;
            
        case 'completed':
            validationSubmitted = false;
            expectedState = null;
            showWizardStep('state-result');
            if (msg.output_file) {
                const video = document.getElementById('final-video');
                const download = document.getElementById('download-btn');
                if (video) video.src = msg.output_file;
                if (download) download.href = msg.output_file;
            }
            break;
    }
}

function handleStatusUpdate(data) {
    if (!data) return;
    
    const status = data.status;
    const phase = data.phase;
    
    console.log('handleStatusUpdate:', { status, phase, validationSubmitted, expectedState });
    
    // If we submitted validation and are waiting for transition, filter out stale validation states
    if (validationSubmitted) {
        // We're expecting to move to processing/translating, so ignore any validation states
        if (status === 'awaiting_validation' || phase === 'awaiting_validation' || 
            status === 'waiting_validation' || phase === 'waiting_validation') {
            console.log('Filtering stale awaiting_validation state during transition');
            // Don't update UI - keep showing optimistic processing state
            return;
        }
        
        // If we see the expected next state, clear the flag
        if (status === 'processing' || phase === 'translating' || phase === 'recomposing' ||
            status === 'queued' && phase === 'translating') {
            console.log('Detected expected state, clearing validationSubmitted flag');
            validationSubmitted = false;
            expectedState = phase;
        }
    }
    
    // Apply the state update
    renderTaskState(data);
}

function renderTaskState(task) {
    console.log('renderTaskState:', task.status, task.phase);
    
    // Determine which wizard step to show based on task state
    const status = task.status;
    const phase = task.phase;
    
    // Check for validation needed states - BOTH status and phase
    if (status === 'awaiting_validation' || phase === 'awaiting_validation' || 
        status === 'waiting_validation' || phase === 'waiting_validation') {
        console.log('Task needs validation, showing validation screen');
        showWizardStep('state-validation');
        
        // Show transcription table if we have segments
        if (task.transcribed_segments && task.transcribed_segments.length > 0) {
            transcriptionSegments = task.transcribed_segments;
            renderTranscriptionTable();
        }
        
        if (task.speaker_config) {
            renderSpeakerConfig(task.speaker_config);
        }
        return;
    }
    
    // Queued or processing states
    if (!status || status === 'queued' || status === 'processing' || status === 'resuming') {
        if (phase === 'identifying' || phase === 'translating' || phase === 'recomposing') {
            showWizardStep('state-processing');
            updateProgress(task.progress || 0, task.message || 'Processing...');
            
            // Show transcription table during processing if we have segments
            if (transcriptionSegments.length > 0) {
                renderTranscriptionTable();
            }
        } else {
            showWizardStep('state-upload');
        }
    } else if (status === 'completed') {
        showWizardStep('state-result');
        if (task.output_path) {
            const video = document.getElementById('final-video');
            const download = document.getElementById('download-btn');
            if (video) video.src = task.output_path;
            if (download) download.href = task.output_path;
        }
    } else if (status === 'failed' || status === 'error') {
        // Clear flags on error
        validationSubmitted = false;
        expectedState = null;
        showWizardStep('state-processing');  // Stay on processing view to show error
        updateProgress(task.progress || 0, `Failed: ${task.error_message || 'Unknown error'}`);
        addLogMessage(`Error: ${task.error_message || 'Unknown error'}`, 'error');
        
        // Show resume/restart hint
        if (task.resumption?.can_resume) {
            addLogMessage('You can resume this task from the Dashboard', 'info');
        }
    } else if (status === 'paused') {
        // Clear flags on pause
        validationSubmitted = false;
        expectedState = null;
        showWizardStep('state-processing');
        updateProgress(task.progress || 0, 'Paused - resume from Dashboard');
        addLogMessage('Task paused - resume from Dashboard', 'info');
    }
    
    // Update progress if available
    if (task.progress !== undefined) {
        updateProgress(task.progress, task.message || task.status);
    }
}

function showWizardStep(stepId) {
    console.log(`Showing wizard step: ${stepId}`);
    
    // Hide all steps
    document.querySelectorAll('.wizard-step').forEach(step => {
        step.classList.add('hidden');
        step.classList.remove('active');
    });
    
    // Show target step
    const target = document.getElementById(stepId);
    if (target) {
        target.classList.remove('hidden');
        target.classList.add('active');
    } else {
        console.error(`Wizard step not found: ${stepId}`);
    }
}

function updateProgress(percent, message) {
    const bar = document.getElementById('main-progress-bar');
    const percentBadge = document.getElementById('processing-percent');
    const statusText = document.getElementById('processing-status-text');
    
    if (bar) bar.style.width = `${percent}%`;
    if (percentBadge) percentBadge.textContent = `${Math.round(percent)}%`;
    if (statusText && message) statusText.textContent = message;
}

function addLogMessage(message, style = 'info') {
    const term = document.getElementById('processing-term') || document.getElementById('term-content');
    if (!term || !message) return;
    
    const line = document.createElement('div');
    line.className = `log-line ${style}`;
    line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    term.appendChild(line);
    term.scrollTop = term.scrollHeight;
}

// NEW: Render transcription table with diarization
function renderTranscriptionTable() {
    let container = document.getElementById('transcription-container');
    
    // Create container if it doesn't exist
    if (!container) {
        const monitorPanel = document.querySelector('.monitor-panel');
        if (!monitorPanel) return;
        
        container = document.createElement('div');
        container.id = 'transcription-container';
        container.className = 'card';
        container.style.marginTop = '20px';
        
        // Insert after the terminal window
        const terminal = monitorPanel.querySelector('.terminal-window');
        if (terminal && terminal.nextSibling) {
            monitorPanel.insertBefore(container, terminal.nextSibling);
        } else if (terminal) {
            monitorPanel.appendChild(container);
        } else {
            monitorPanel.appendChild(container);
        }
    }
    
    // Build table HTML
    let html = `
        <h3 style="margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
            <i class="fas fa-closed-captioning" style="color: var(--accent);"></i>
            Speech Transcription & Translation
            <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: 400;">
                (updates as translation progresses)
            </span>
        </h3>
        <div style="max-height: 400px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                <thead style="position: sticky; top: 0; background: var(--bg-panel); z-index: 10;">
                    <tr>
                        <th style="padding: 12px; text-align: left; border-bottom: 1px solid var(--border); font-weight: 600; color: var(--accent); width: 80px;">Time</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 1px solid var(--border); font-weight: 600; color: var(--accent); width: 100px;">Speaker</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 1px solid var(--border); font-weight: 600; color: var(--accent);">Original</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 1px solid var(--border); font-weight: 600; color: var(--success);">Translated</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    transcriptionSegments.forEach((seg, index) => {
        const startTime = formatTime(seg.start || 0);
        const endTime = formatTime(seg.end || 0);
        const speakerNum = (seg.speaker_id !== undefined ? seg.speaker_id : 0) + 1;
        const originalText = escapeHtml(seg.original_text || '');
        const translatedText = seg.translated_text ? escapeHtml(seg.translated_text) : 
                              '<span style="color: var(--text-muted); font-style: italic;">(translating...)</span>';
        const rowClass = seg.translated_text ? 'translated' : 'pending';
        
        html += `
            <tr class="transcription-row ${rowClass}" data-segment-idx="${seg.idx !== undefined ? seg.idx : index}" 
                style="border-bottom: 1px solid var(--border); transition: background-color 0.3s;">
                <td style="padding: 10px 12px; color: var(--text-muted); font-family: var(--font-code); font-size: 0.8rem;">
                    ${startTime}<br><span style="font-size: 0.7rem;">${endTime}</span>
                </td>
                <td style="padding: 10px 12px;">
                    <span class="speaker-badge" style="
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        width: 28px;
                        height: 28px;
                        border-radius: 50%;
                        background: hsl(${(speakerNum * 60) % 360}, 70%, 50%);
                        color: white;
                        font-size: 0.75rem;
                        font-weight: 600;
                    ">S${speakerNum}</span>
                </td>
                <td style="padding: 10px 12px; color: var(--text-main); line-height: 1.5;">
                    ${originalText}
                </td>
                <td style="padding: 10px 12px; color: var(--text-main); line-height: 1.5;" class="translation-cell">
                    ${translatedText}
                </td>
            </tr>
        `;
    });
    
    html += `
                </tbody>
            </table>
        </div>
        <style>
            .transcription-row.translated {
                background-color: rgba(35, 134, 54, 0.05);
            }
            .transcription-row.translated .translation-cell {
                animation: flashSuccess 0.5s ease;
            }
            @keyframes flashSuccess {
                0% { background-color: rgba(35, 134, 54, 0.3); }
                100% { background-color: transparent; }
            }
            .transcription-row:hover {
                background-color: var(--bg-hover) !important;
            }
        </style>
    `;
    
    container.innerHTML = html;
    container.style.display = 'block';
}

// NEW: Update translation progress in existing table
function updateTranslationProgress(updatedSegments) {
    // Update our stored segments with new translations
    updatedSegments.forEach(updated => {
        const idx = updated.idx !== undefined ? updated.idx : updated.segment_idx;
        const existingIndex = transcriptionSegments.findIndex(s => 
            (s.idx !== undefined ? s.idx : s.segment_idx) === idx
        );
        if (existingIndex >= 0) {
            // Merge updated data
            transcriptionSegments[existingIndex] = {
                ...transcriptionSegments[existingIndex],
                ...updated,
                translated_text: updated.translated_text || transcriptionSegments[existingIndex].translated_text
            };
        } else {
            // Add new segment
            transcriptionSegments.push(updated);
        }
    });
    
    // Sort by index
    transcriptionSegments.sort((a, b) => {
        const idxA = a.idx !== undefined ? a.idx : a.segment_idx || 0;
        const idxB = b.idx !== undefined ? b.idx : b.segment_idx || 0;
        return idxA - idxB;
    });
    
    // Re-render the table
    renderTranscriptionTable();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 10);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}${ms > 0 ? '.' + ms : ''}`;
}

function renderSpeakerConfig(config) {
    const grid = document.getElementById('speaker-grid');
    if (!grid) return;
    
    console.log('Rendering speaker config:', config);
    
    grid.innerHTML = Object.entries(config).map(([id, info]) => `
        <div class="speaker-card" data-speaker-id="${id}">
            <div class="spk-header">
                <div class="avatar">${parseInt(id) + 1}</div>
                <input type="text" class="spk-name-input" value="${info.name || `Speaker ${parseInt(id) + 1}`}" 
                       data-speaker-id="${id}">
            </div>
            <audio controls src="${info.sample_path}" style="width: 100%; margin-top: 10px;"></audio>
            <select class="spk-action" data-speaker-id="${id}">
                <option value="dub" ${info.action === 'dub' || info.action === 'translate' ? 'selected' : ''}>Dub/Translate</option>
                <option value="remove" ${info.action === 'remove' ? 'selected' : ''}>Remove (Mute)</option>
            </select>
        </div>
    `).join('');
}

function renderSpeakerSamples(speakerData) {
    // Convert to same format as speaker_config
    const config = {};
    Object.entries(speakerData).forEach(([id, info]) => {
        config[id] = {
            name: info.default_name || `Speaker ${parseInt(id) + 1}`,
            action: 'dub',
            sample_path: `data:audio/wav;base64,${info.audio_base64}`
        };
    });
    renderSpeakerConfig(config);
}

async function startTranslationPhase() {
    if (!currentTaskId) {
        alert('No active project');
        return;
    }
    
    // Collect current configuration from inputs
    const config = {};
    document.querySelectorAll('.speaker-card').forEach(card => {
        const id = card.dataset.speakerId;
        const nameInput = card.querySelector('.spk-name-input');
        const actionSelect = card.querySelector('.spk-action');
        
        config[id] = {
            name: nameInput?.value || `Speaker ${parseInt(id) + 1}`,
            action: actionSelect?.value || 'dub'
        };
    });
    
    console.log('Submitting validation:', config);
    
    // CRITICAL: Set optimistic state BEFORE the API call to prevent flicker
    validationSubmitted = true;
    expectedState = 'translating';
    
    // Immediately show processing UI (optimistic update)
    showWizardStep('state-processing');
    updateProgress(35, 'Starting translation pipeline...');
    addLogMessage('Submitting speaker validation...', 'step');
    
    try {
        // Send validation to server
        const response = await postJSON(`/api/projects/${currentTaskId}/validate`, { speakers: config });
        console.log('Validation API response:', response);
        
        // Server accepted - wait for WebSocket confirmation
        // The optimistic state is already showing, so we just wait
        
        // Safety timeout: if server doesn't confirm within 15 seconds, re-check state
        setTimeout(() => {
            if (validationSubmitted && currentTaskId) {
                console.log('Safety check: validation still pending after 15s, refreshing state');
                validationSubmitted = false;
                expectedState = null;
                // Force a state refresh
                getJSON(`/api/projects/${currentTaskId}`).then(task => {
                    renderTaskState(task);
                }).catch(err => {
                    console.error('Safety check failed:', err);
                });
            }
        }, 15000);
        
    } catch (err) {
        console.error('Validation submission failed:', err);
        // Revert optimistic update on error
        validationSubmitted = false;
        expectedState = null;
        addLogMessage(`Failed to start dubbing: ${err.message}`, 'error');
        // Go back to validation screen so user can retry
        showWizardStep('state-validation');
        alert('Failed to start dubbing: ' + err.message);
    }
}

function closeProject() {
    if (currentWs) {
        currentWs.close();
        currentWs = null;
    }
    currentTaskId = null;
    
    // Reset state flags
    validationSubmitted = false;
    expectedState = null;
    transcriptionSegments = [];
    
    // Reset downloaded video state
    downloadedVideoPath = null;
    downloadedVideoName = null;
    
    // Remove transcription table if exists
    const transcriptionContainer = document.getElementById('transcription-container');
    if (transcriptionContainer) {
        transcriptionContainer.remove();
    }
    
    // Update UI
    const controls = document.getElementById('project-controls');
    if (controls) controls.style.display = 'none';
    
    // Reset YouTube section
    const youtubeInput = document.getElementById('youtube-url');
    const fileDisplay = document.getElementById('file-display-text');
    const downloadBtn = document.getElementById('download-youtube-btn');
    
    if (youtubeInput) youtubeInput.value = '';
    if (fileDisplay) {
        fileDisplay.innerHTML = 'Click to browse or drag video here';
        fileDisplay.classList.remove('file-selected');
    }
    if (downloadBtn) {
        downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download';
        downloadBtn.disabled = false;
    }
    
    // Reset project options to defaults
    const srcLang = document.getElementById('src-lang');
    const tgtLang = document.getElementById('tgt-lang');
    const separateAudio = document.getElementById('separate-audio');
    const ttsEngine = document.getElementById('tts-engine');
    
    if (srcLang) srcLang.value = 'auto';
    if (tgtLang) tgtLang.value = 'en';
    if (separateAudio) separateAudio.checked = false;
    if (ttsEngine) ttsEngine.value = 'f5';
    
    // Hide download original button
    hideDownloadOriginalButton();
    
    // Reset file input
    const fileInput = document.getElementById('video-input');
    if (fileInput) fileInput.value = '';
    
    updateSubmitButtonState();
    
    showWizardStep('state-upload');
    
    // Clear logs
    const term = document.getElementById('processing-term');
    if (term) term.innerHTML = '';
}
