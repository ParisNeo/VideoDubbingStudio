import { getJSON, deleteResource, postJSON } from '../api.js';

// Import the switchView function from app.js
import { doSwitchView } from '../app.js';

// Make functions globally available for inline onclick handlers
window.openTask = openTask;
window.deleteTask = deleteTask;
window.resumeTask = resumeTask;
window.restartTask = restartTask;
window.newProject = newProject;

export function init() {
    console.log('Dashboard: init() called');
    
    // Bind refresh button if it exists
    const refreshBtn = document.getElementById('refresh-dash-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            console.log('Dashboard: refresh clicked');
            loadTasks();
        });
    }
    
    // Bind new project button
    const newProjectBtn = document.getElementById('new-project-btn');
    if (newProjectBtn) {
        newProjectBtn.addEventListener('click', () => {
            newProject();
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
    grid.innerHTML = '<div class="empty-state"><i class="fas fa-spinner fa-spin"></i> Loading tasks...</div>';

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
                <p>No projects yet. Start by creating a new project!</p>
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
        
        const date = task.created_at ? 
            new Date(task.created_at * 1000).toLocaleDateString() : 'Unknown';
        
        // Determine available actions based on status
        const isActuallyRunning = task.status === 'processing' && !task.was_running_at_shutdown;
        const canResume = ['paused', 'failed', 'error', 'resuming'].includes(task.status) ||
                         (task.status === 'processing' && task.was_running_at_shutdown);
        const canRestart = !['processing', 'queued'].includes(task.status) || task.was_running_at_shutdown;
        // Allow delete for any task except actively running ones
        const canDelete = !isActuallyRunning || task.was_running_at_shutdown || ['completed', 'failed', 'error', 'cancelled', 'paused'].includes(task.status);
        
        // Build action buttons HTML
        let actionButtons = '';
        
        if (task.status === 'completed') {
            actionButtons = `
                <button class="btn-primary btn-sm" onclick="openTask('${task.task_id}')">
                    <i class="fas fa-download"></i> Download
                </button>
                <button class="btn-secondary btn-sm" onclick="restartTask('${task.task_id}')" title="Start over with same video">
                    <i class="fas fa-redo"></i> Restart
                </button>
            `;
        } else if (canResume) {
            const resumeIcon = task.was_running_at_shutdown ? 'fa-play-circle' : 'fa-play';
            const resumeText = task.was_running_at_shutdown ? 'Resume' : 'Retry';
            actionButtons = `
                <button class="btn-primary btn-sm" onclick="resumeTask('${task.task_id}')">
                    <i class="fas ${resumeIcon}"></i> ${resumeText}
                </button>
                ${canRestart ? `
                <button class="btn-secondary btn-sm" onclick="restartTask('${task.task_id}')" title="Start over from beginning">
                    <i class="fas fa-redo"></i> Restart
                </button>
                ` : ''}
            `;
        } else if (['processing', 'queued'].includes(task.status)) {
            actionButtons = `
                <button class="btn-primary btn-sm" onclick="openTask('${task.task_id}')">
                    <i class="fas fa-eye"></i> Monitor
                </button>
            `;
        } else {
            actionButtons = `
                <button class="btn-primary btn-sm" onclick="openTask('${task.task_id}')">
                    <i class="fas fa-folder-open"></i> Open
                </button>
            `;
        }
        
        const deleteButton = canDelete ? `
            <button class="btn-danger btn-sm" onclick="deleteTask('${task.task_id}', event)" title="Delete project">
                <i class="fas fa-trash"></i>
            </button>
        ` : '';
        
        const interruptedWarning = task.was_running_at_shutdown ? `
            <span class="status-badge attention" title="App was restarted while this task was running">
                <i class="fas fa-exclamation-triangle"></i> Interrupted
            </span>
        ` : '';
        
        const resumeAttempts = task.resume_attempts > 0 ? `
            <span style="font-size: 0.75rem; color: var(--text-muted); margin-left: 8px;">
                (retried ${task.resume_attempts}x)
            </span>
        ` : '';
        
        return `
            <div class="task-card" data-task-id="${task.task_id}">
                <div class="task-header">
                    <div class="task-title">
                        <span class="filename" title="${task.filename || 'Untitled'}">${truncate(task.filename || 'Untitled', 30)}</span>
                        <span class="task-date">${date}${resumeAttempts}</span>
                    </div>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        ${interruptedWarning}
                        <span class="status-badge ${statusClass}">${formatStatus(task.status)}</span>
                    </div>
                </div>
                
                <div class="progress-section">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progressWidth}%"></div>
                    </div>
                    <span class="progress-text">${progressWidth}%</span>
                </div>
                
                <div class="phase-info">
                    <span class="phase-label">Phase: ${task.phase || 'unknown'}</span>
                    ${task.error_message ? `<span class="error-hint" title="${task.error_message}"><i class="fas fa-exclamation-triangle"></i> Error</span>` : ''}
                </div>
                
                <div class="task-actions">
                    ${actionButtons}
                    ${deleteButton}
                </div>
            </div>
        `;
    }).join('');
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
        'awaiting_validation': 'Review Speakers',
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

function newProject() {
    console.log('Dashboard: Creating new project');
    // Switch to video dub view and ensure fresh state
    import('./video_dub.js').then(mod => {
        mod.startNewProject();
    }).catch(err => {
        console.error('Failed to load video_dub module:', err);
        // Fallback: just switch view
        doSwitchView('video_dub');
    });
}

function openTask(taskId) {
    console.log(`Opening task: ${taskId}`);
    import('./video_dub.js').then(mod => {
        mod.openMonitor(taskId);
    }).catch(err => {
        console.error('Failed to load video_dub module:', err);
    });
}

async function resumeTask(taskId) {
    console.log(`Resuming task: ${taskId}`);
    if (!confirm('Resume this task from where it left off?')) return;
    
    try {
        const response = await postJSON(`/api/projects/${taskId}/resume`, {});
        alert(`Task is resuming: ${response.message || 'Please wait...'}`);
        openTask(taskId);
    } catch (err) {
        alert('Failed to resume task: ' + err.message);
    }
}

async function restartTask(taskId, fromPhase = null) {
    const phaseOptions = fromPhase ? '' : `
    
Options:
• From beginning - re-analyze speakers
• From translation - skip speaker ID (if already done)`;
    
    const message = fromPhase 
        ? `Restart this task from ${fromPhase}?`
        : `Restart this task from the beginning?${phaseOptions}`;
    
    if (!confirm(message)) return;
    
    const payload = fromPhase ? { from_phase: fromPhase } : {};
    
    try {
        const response = await postJSON(`/api/projects/${taskId}/restart`, payload);
        alert(`Task restarted: ${response.message || 'Please wait...'}`);
        openTask(taskId);
    } catch (err) {
        alert('Failed to restart task: ' + err.message);
    }
}

function deleteTask(taskId, event) {
    if (event) event.stopPropagation();
    
    if (!confirm('Delete this project permanently? This cannot be undone.\n\nNote: If the task appears stuck, deletion will force-stop it.')) return;
    
    console.log(`Attempting to delete task: ${taskId}`);
    
    // Try cancel first (best effort), then delete
    postJSON(`/api/projects/${taskId}/cancel`, {}).catch(err => {
        console.log('Cancel before delete failed (expected):', err);
    }).finally(() => {
        performDelete(taskId);
    });
}

async function performDelete(taskId) {
    try {
        await deleteResource(`/api/projects/${taskId}`);
        console.log(`Successfully deleted task: ${taskId}`);
        loadTasks();
    } catch (err) {
        console.error('Delete failed:', err);
        alert('Delete failed: ' + err.message + '\n\nTry refreshing the page and deleting again.');
    }
}
