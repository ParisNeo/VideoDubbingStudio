// API Client Utilities

const API_BASE = '';

/**
 * Generic fetch wrapper with error handling
 */
export async function fetchAPI(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    
    const defaults = {
        headers: {
            'Accept': 'application/json',
        }
    };

    // Don't set Content-Type for FormData (browser sets it with boundary)
    if (!(options.body instanceof FormData)) {
        defaults.headers['Content-Type'] = 'application/json';
    }

    const config = { ...defaults, ...options };
    
    // Stringify JSON body
    if (config.body && typeof config.body === 'object' && !(config.body instanceof FormData)) {
        config.body = JSON.stringify(config.body);
    }

    try {
        const response = await fetch(url, config);
        
        if (!response.ok) {
            // Try to parse error as JSON first
            let errorText;
            const contentType = response.headers.get('content-type') || '';
            
            try {
                if (contentType.includes('application/json')) {
                    const errorJson = await response.json();
                    errorText = errorJson.detail || errorJson.message || JSON.stringify(errorJson);
                } else {
                    errorText = await response.text();
                }
            } catch (parseErr) {
                // If we can't read the body, use status text
                errorText = response.statusText || `HTTP ${response.status}`;
            }
            
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        // Check if response is JSON
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return response.json();
        }
        
        // Return text for non-JSON responses
        return response.text();
        
    } catch (err) {
        console.error('API Error:', err);
        throw err;
    }
}

export async function postFormData(endpoint, formData) {
    return fetchAPI(endpoint, {
        method: 'POST',
        body: formData
    });
}

export async function postJSON(endpoint, data) {
    return fetchAPI(endpoint, {
        method: 'POST',
        body: data
    });
}

export async function getJSON(endpoint) {
    return fetchAPI(endpoint, {
        method: 'GET'
    });
}

export async function deleteResource(endpoint) {
    return fetchAPI(endpoint, {
        method: 'DELETE'
    });
}

/**
 * Format seconds to MM:SS
 */
export function formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Format file size
 */
export function formatFileSize(bytes) {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/**
 * Debounce function
 */
export function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
