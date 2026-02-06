// Main Application Entry Point
import * as Dashboard from './views/dashboard.js';
import * as VideoDub from './views/video_dub.js';
import * as Transcribe from './views/transcribe.js';
import * as Analysis from './views/analysis.js';
import * as Recorder from './views/recorder.js';

const modules = {
    'dashboard': Dashboard,
    'video_dub': VideoDub,
    'transcribe': Transcribe,
    'analysis': Analysis,
    'recorder': Recorder
};

// Global state
window.appState = {
    currentTaskId: null,
    currentView: null
};

document.addEventListener('DOMContentLoaded', () => {
    console.log("%c VoiceDub Pro Modular - Initializing ", 
                "background: #222; color: #bada55; font-size: 14px");

    // Setup navigation first
    setupNavigation();
    
    // Initialize all modules
    Object.entries(modules).forEach(([name, mod]) => {
        try {
            if (mod.init) {
                mod.init();
                console.log(`✅ ${name} initialized`);
            }
        } catch (e) {
            console.error(`❌ ${name} failed:`, e);
        }
    });

    // Load initial view - use a small delay to ensure DOM is fully ready
    setTimeout(() => {
        const savedView = localStorage.getItem('vdpu_active_view') || 'dashboard';
        console.log(`Loading initial view: ${savedView}`);
        doSwitchView(savedView);
    }, 100);
});

function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-links li[data-view]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const viewName = link.dataset.view;
            doSwitchView(viewName);
        });
    });
}

// Main switch view function - exported for ES modules AND bound to window for inline handlers
export function doSwitchView(viewName) {
    console.log(`Switching to view: ${viewName}`);
    
    // Validate view exists
    if (!modules[viewName]) {
        console.error(`Unknown view: ${viewName}`);
        return;
    }

    // Get current and target elements
    const currentId = window.appState.currentView;
    const target = document.getElementById(`view-${viewName}`);
    
    if (!target) {
        console.error(`View element not found: view-${viewName}`);
        return;
    }

    // Hide current view if exists
    if (currentId) {
        const currentEl = document.getElementById(`view-${currentId}`);
        if (currentEl) {
            currentEl.style.display = 'none';
            currentEl.classList.remove('active');
        }
    }

    // Show target view - CRITICAL: Must set display before calling onShow
    target.style.display = 'block';
    target.classList.add('active');

    // Update nav active state
    document.querySelectorAll('.nav-links li').forEach(li => {
        li.classList.toggle('active', li.dataset.view === viewName);
    });

    // Update global state BEFORE calling onShow
    window.appState.currentView = viewName;
    localStorage.setItem('vdpu_active_view', viewName);

    // Trigger module's onShow if exists
    const mod = modules[viewName];
    if (mod && mod.onShow) {
        console.log(`Calling onShow for ${viewName}`);
        try {
            mod.onShow();
        } catch (e) {
            console.error(`onShow failed for ${viewName}:`, e);
        }
    }

    console.log(`Switched to view: ${viewName}`);
}

// CRITICAL: Bind to window for inline onclick handlers in HTML
window.doSwitchView = doSwitchView;
// Also provide switchView alias for backward compatibility
window.switchView = doSwitchView;
