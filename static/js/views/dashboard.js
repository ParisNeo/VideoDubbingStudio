import { getJSON, deleteResource, postJSON } from '../api.js';
import { doSwitchView } from '../app.js';

// Make functions globally available for inline onclick handlers
window.openTask = openTask;
window.deleteTask = deleteTask;
window.resumeTask = resumeTask;
window.restartTask = restartTask;
window.newProject = newProject;

export function init() {
    console.log('Dashboard: init() called');
    
    // Bind refresh button
    const refreshBtn = document.getElementById('refresh-dash-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            console.log('Dashboard: refresh clicked');
            loadTasks();
        });
    }
    
    // Bind new project button - opens the modal
    const newProjectBtn = document.getElementById('new-project-btn');
    if (newProjectBtn) {
        newProjectBtn.addEventListener('click', () => {
            openProjectModal();
        });
    }
    
    // Auto-refresh every 10 seconds when visible
    setInterval(() => {
        const dashView = document.getElementById('view-dashboard');
        const isVisible = dashView && (
            dashView.style.display === 'block' || 
            dashView.classList.contains('active')
        );
        if (isVisible) {
            console.log('Dashboard: auto-refresh triggered');
            loadTasks();
        }
    }, 10000);
}

export function onShow() {
    console.log('Dashboard: onShow() called - loading tasks');
    // Always load fresh data when dashboard becomes visible
    loadTasks();
}

async function loadTasks() {
    const grid = document.getElementById('task-grid');
    if (!grid) {
        console.error('Dashboard: task-grid element not found');
        return;
    }

    // Show loading state
    grid.innerHTML = '<div class="empty-state"><i class="fas fa-spinner fa-spin"></i> Loading projects...</div>';

    try {
        console.log('Dashboard: fetching /api/projects');
        const data = await getJSON('/api/projects');
        console.log('Dashboard: received data', data);
        renderTasks(data.tasks || [], grid);
    } catch (err) {
        console.error('Failed to load tasks:', err);
        grid.innerHTML = `<div class="error-state"><i class="fas fa-exclamation-circle"></i> Failed to load projects: ${err.message}</div>`;
    }
}

function renderTasks(tasks, container) {
    console.log(`Dashboard: rendering ${tasks.length} tasks`, tasks);
    
    if (!tasks.length) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-film" style="font-size: 3rem; margin-bottom: 15px; opacity: 0.5;"></i>
                <p>No projects yet. Click "New Project" to get started!</p>
                <button class="btn-primary" onclick="newProject()" style="margin-top: 15px;">
                    <i class="fas fa-plus"></i> New Project
                </button>
            </div>
        `;
        return;
    }

    container.innerHTML = tasks.map(task => {
        const statusClass = getStatusClass(task.status);
        const progressWidth = task.status === 'awaiting_validation' ? 30 : 
                             task.status === 'completed' ? 100 : 
                             (task.progress || 0);
        
        // Fix: Parse ISO string correctly
        const date = task.created_at ? 
            new Date(task.created_at).toLocaleDateString() : 'Unknown';
        
        // Determine project type icon
        const typeIcon = getProjectTypeIcon(task);
        
        // Simplified action logic
        const isAwaitingValidation = ['awaiting_validation', 'awaiting_translation_review'].includes(task.status) || ['awaiting_validation', 'awaiting_translation_review'].includes(task.phase);
        const canMonitor = ['processing', 'queued', 'running_translation'].includes(task.status) || isAwaitingValidation;
        const canResume = (['paused', 'failed', 'error'].includes(task.status) || task.was_running_at_shutdown) && !isAwaitingValidation;
        
        // Build action buttons HTML
        let actionButtons = '';
        
        if (task.status === 'completed') {
            actionButtons = `
                <button class="btn-secondary btn-sm" onclick="openTask('${task.task_id}')" title="View details and steps">
                    <i class="fas fa-eye"></i> View Steps
                </button>
                <a href="/api/projects/${task.task_id}/download" class="btn-primary btn-sm" title="Download result">
                    <i class="fas fa-download"></i> Download
                </a>
            `;
        } else if (canResume) {
            const resumeText = task.was_running_at_shutdown ? 'Resume' : 'Retry';
            actionButtons = `
                <button class="btn-primary btn-sm" onclick="resumeTask('${task.task_id}')" title="${resumeText} processing">
                    <i class="fas fa-play"></i> ${resumeText}
                </button>
            `;
        } else if (canMonitor) {
            actionButtons = `
                <button class="btn-primary btn-sm" onclick="openTask('${task.task_id}')" title="Monitor progress">
                    <i class="fas fa-eye"></i> Monitor
                </button>
            `;
        } else {
            actionButtons = `
                <button class="btn-secondary btn-sm" onclick="openTask('${task.task_id}')" title="View details">
                    <i class="fas fa-folder-open"></i> View
                </button>
            `;
        }
        
        const interruptedWarning = task.was_running_at_shutdown ? `
            <span class="status-badge attention" title="App was restarted while this task was running">
                <i class="fas fa-exclamation-triangle"></i> Interrupted
            </span>
        ` : '';
        
        return `
            <div class="task-card" data-task-id="${task.task_id}" data-status="${task.status}">
                <div class="task-header" onclick="openTask('${task.task_id}')" style="cursor: pointer;">
                    <div style="display: flex; gap: 15px; align-items: center; width: 100%;">
                        <div style="width: 60px; height: 45px; background: var(--bg-hover); border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; flex-shrink: 0; border: 1px solid var(--border);">
                            ${typeIcon}
                        </div>
                        <div class="task-title" style="flex: 1; min-width: 0;">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; width: 100%;">
                                <span class="filename" title="${task.filename || 'Untitled'}" style="font-size: 1.05rem; margin-bottom: 4px; color: var(--accent);">${truncate(task.filename || 'Untitled', 35)}</span>
                                <div class="task-badges">
                                    ${interruptedWarning}
                                    <span class="status-badge ${statusClass}">${formatStatus(task.status)}</span>
                                </div>
                            </div>
                            <span class="task-meta">
                                <span class="task-date"><i class="far fa-calendar"></i> ${date}</span>
                                ${task.src_lang && task.tgt_lang ? `<span class="task-langs"><i class="fas fa-language"></i> ${task.src_lang === 'auto' ? 'Auto' : task.src_lang} → ${task.tgt_lang}</span>` : ''}
                            </span>
                        </div>
                    </div>
                </div>
                
                <div class="progress-section">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progressWidth}%"></div>
                    </div>
                    <div class="progress-info">
                        <span class="progress-text">${progressWidth}%</span>
                        <span class="phase-label">${task.phase || 'pending'}</span>
                    </div>
                </div>
                
                ${task.error_message ? `<div class="error-banner"><i class="fas fa-exclamation-circle"></i> ${truncate(task.error_message, 60)}</div>` : ''}
                
                <div class="task-actions">
                    ${actionButtons}
                    <div class="task-actions-secondary">
                        <button class="btn-danger btn-sm" onclick="deleteTask('${task.task_id}', event)" title="Delete project">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function getProjectTypeIcon(task) {
    // Determine icon based on task characteristics
    if (task.source === 'youtube') {
        return '<i class="fab fa-youtube" style="color: #ff4444;"></i>';
    } else if (task.separate_audio) {
        return '<i class="fas fa-music" style="color: #a78bfa;"></i>';
    } else if (task.tts_engine === 'f5' || task.tts_engine === 'fishspeech') {
        return '<i class="fas fa-microphone-alt" style="color: #34d399;"></i>';
    }
    return '<i class="fas fa-file-video" style="color: var(--accent);"></i>';
}

function getStatusClass(status) {
    const map = {
        'completed': 'success',
        'processing': 'processing',
        'queued': 'pending',
        'awaiting_validation': 'attention',
        'awaiting_input': 'attention',
        'failed': 'error',
        'error': 'error',
        'paused': 'attention',
        'cancelled': 'error',
        'resuming': 'processing'
    };
    return map[status] || 'default';
}

function formatStatus(status) {
    const map = {
        'awaiting_input': 'Needs Input',
        'awaiting_validation': 'Review',
        'completed': 'Done',
        'processing': 'Running',
        'queued': 'Queued',
        'failed': 'Failed',
        'error': 'Error',
        'cancelled': 'Cancelled',
        'paused': 'Paused',
        'resuming': 'Resuming...'
    };
    return map[status] || status?.replace(/_/g, ' ') || 'Unknown';
}

function truncate(str, maxLen) {
    if (!str) return '';
    return str.length > maxLen ? str.slice(0, maxLen - 3) + '...' : str;
}

// New project - opens the modal (defined in dashboard.html)
function newProject() {
    // The modal is handled by the global function in dashboard.html
    if (window.openProjectModal) {
        window.openProjectModal();
    }
}

async function openTask(taskId) {
    console.log(`Opening task: ${taskId}`);
    try {
        const task = await getJSON(`/api/projects/${taskId}`);
        
        // Route to the correct view based on source
        if (task.source === 'transcribe') {
            doSwitchView('transcribe');
            // Assuming transcribe module has a similar monitor function
            import('./transcribe.js').then(mod => {
                if (mod.openMonitor) mod.openMonitor(taskId);
                else mod.onShow(); // Fallback if specialized monitor isn't ready
            });
        } else {
            // Default to video dubbing
            import('./video_dub.js').then(mod => {
                mod.openMonitor(taskId);
            });
        }
    } catch (err) {
        console.error('Failed to route task:', err);
    }
}
async function resumeTask(taskId) {
    console.log(`Resuming task: ${taskId}`);
    const confirmed = await window.uiConfirm('Resume this task from where it left off?', 'Resume Task');
    if (!confirmed) return;
    
    try {
        const response = await postJSON(`/api/projects/${taskId}/resume`, {});
        await window.uiAlert(`Task is resuming: ${response.message || 'Please wait...'}`, 'Success');
        openTask(taskId);
    } catch (err) {
        await window.uiAlert('Failed to resume task: ' + err.message, 'Error');
    }
}

async function restartTask(taskId, fromPhase = null) {
    const message = fromPhase 
        ? `Restart this task from the ${fromPhase} phase?`
        : `Restart this task from the beginning?`;
    
    const confirmed = await window.uiConfirm(message, 'Restart Task');
    if (!confirmed) return;
    
    const payload = fromPhase ? { from_phase: fromPhase } : {};
    
    try {
        const response = await postJSON(`/api/projects/${taskId}/restart`, payload);
        await window.uiAlert(`Task restarted: ${response.message || 'Please wait...'}`, 'Success');
        openTask(taskId);
    } catch (err) {
        await window.uiAlert('Failed to restart task: ' + err.message, 'Error');
    }
}

async function deleteTask(taskId, event) {
    if (event) event.stopPropagation();
    
    const confirmMsg = `This will permanently delete this project and all associated files.\n\nIf the task is currently running, it will be force-stopped.\n\nThis action cannot be undone.`;
    
    const confirmed = await window.uiConfirm(confirmMsg, '🗑️ Delete Project');
    if (!confirmed) return;
    
    console.log(`Force deleting task: ${taskId}`);
    performDelete(taskId);
}

async function performDelete(taskId) {
    // Visual feedback
    const card = document.querySelector(`.task-card[data-task-id="${taskId}"]`);
    if (card) {
        card.style.opacity = '0.5';
        card.style.pointerEvents = 'none';
    }
    
    try {
        await deleteResource(`/api/projects/${taskId}?force=true`);
        console.log(`Successfully deleted task: ${taskId}`);
        
        // Remove card with animation
        if (card) {
            card.style.transform = 'translateX(-20px)';
            card.style.opacity = '0';
            setTimeout(() => {
                card.remove();
                const remaining = document.querySelectorAll('.task-card');
                if (remaining.length === 0) {
                    loadTasks();
                }
            }, 200);
        } else {
            loadTasks();
        }
    } catch (err) {
        console.error('Delete failed:', err);
        if (card) {
            card.style.opacity = '1';
            card.style.pointerEvents = 'auto';
        }
        await window.uiAlert('Delete failed: ' + err.message, 'Error');
    }
}
