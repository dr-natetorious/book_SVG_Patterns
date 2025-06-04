/**
 * Filter Control - now much simpler with BootstrapUtils
 */
import { BootstrapWebComponent } from '../core/BootstrapWebComponent.js';
import { EventUtils } from '../utils/EventUtils.js';

class FilterControl extends BootstrapWebComponent {
    static get observedAttributes() {
        return ['data-schema', 'data-placeholder'];
    }

    onConnected() {
        this.render();
        this.setupComponents();
        this.setupEventListeners();
    }

    render() {
        // No need to inject Bootstrap CSS - it's already available!
        this.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <h6 class="card-title mb-0">
                        <i class="bi bi-funnel"></i> Advanced Filters
                    </h6>
                </div>
                <div class="card-body">
                    <form id="filterForm" novalidate>
                        <div class="mb-3">
                            <label for="searchInput" class="form-label">Search Query</label>
                            <div class="input-group">
                                <span class="input-group-text">
                                    <i class="bi bi-search"></i>
                                </span>
                                <input type="text" 
                                       class="form-control" 
                                       id="searchInput"
                                       placeholder="${this.getAttribute('data-placeholder') || 'Enter query...'}"
                                       required>
                                <button class="btn btn-outline-secondary dropdown-toggle" 
                                        type="button" 
                                        data-bs-toggle="dropdown">
                                    <i class="bi bi-lightbulb"></i>
                                </button>
                                <ul class="dropdown-menu dropdown-menu-end" id="suggestionsMenu">
                                    <!-- Populated dynamically -->
                                </ul>
                            </div>
                            <div class="invalid-feedback"></div>
                        </div>

                        <div class="d-grid gap-2 d-md-flex justify-content-md-end">
                            <button type="button" class="btn btn-outline-secondary" id="clearBtn">
                                <i class="bi bi-arrow-clockwise"></i> Clear
                            </button>
                            <button type="submit" class="btn btn-primary" id="searchBtn">
                                <i class="bi bi-search"></i> Search
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        `;
    }

    setupComponents() {
        // Bootstrap components work directly - no Shadow DOM barriers!
        const dropdown = this.querySelector('[data-bs-toggle="dropdown"]');
        this.dropdown = this.createDropdown(dropdown);
        
        // Populate suggestions
        this.populateSuggestions();
    }

    setupEventListeners() {
        const form = this.querySelector('#filterForm');
        const searchInput = this.querySelector('#searchInput');
        const clearBtn = this.querySelector('#clearBtn');

        // Form submission
        this.addCleanup(
            EventUtils.on(form, 'submit', (e) => {
                e.preventDefault();
                if (this.validateForm(form)) {
                    this.executeSearch();
                }
            })
        );

        // Real-time validation
        this.addCleanup(
            EventUtils.on(searchInput, 'input', EventUtils.debounce(() => {
                this.validateQuery(searchInput.value);
            }, 300))
        );

        // Clear button
        this.addCleanup(
            EventUtils.on(clearBtn, 'click', () => {
                this.clearFilter();
            })
        );

        // Suggestion clicks
        this.addCleanup(
            EventUtils.delegate(this, '[data-suggestion]', 'click', (e) => {
                e.preventDefault();
                const suggestion = e.target.dataset.suggestion;
                searchInput.value = suggestion;
                this.validateQuery(suggestion);
            })
        );
    }

    validateQuery(query) {
        const searchInput = this.querySelector('#searchInput');
        
        try {
            const isValid = this.isValidLuceneQuery(query);
            this.setValidationState(searchInput, isValid, 
                isValid ? '' : 'Invalid Lucene query syntax');
            
            if (isValid && query.trim()) {
                this.previewQuery(query);
            }
        } catch (error) {
            this.setValidationState(searchInput, false, error.message);
        }
    }

    executeSearch() {
        const searchInput = this.querySelector('#searchInput');
        const query = searchInput.value.trim();
        
        if (!query) return;

        // Show loading state
        const searchBtn = this.querySelector('#searchBtn');
        this.setLoadingState(searchBtn, true);

        // Emit search event
        this.dispatchEvent(new CustomEvent('search', {
            detail: { query },
            bubbles: true
        }));

        // Show success toast
        this.showToast('Executing search...', 'info');

        // Reset loading state after mock delay
        setTimeout(() => {
            this.setLoadingState(searchBtn, false);
            this.showToast('Search completed', 'success');
        }, 1000);
    }

    clearFilter() {
        const form = this.querySelector('#filterForm');
        BootstrapUtils.resetForm(form);
        
        this.dispatchEvent(new CustomEvent('clear', {
            bubbles: true
        }));

        this.showToast('Filters cleared', 'secondary');
    }

    populateSuggestions() {
        const menu = this.querySelector('#suggestionsMenu');
        const suggestions = [
            { text: 'Active Items', query: 'status:active' },
            { text: 'High Priority', query: 'priority:high' },
            { text: 'Last 7 Days', query: 'created:[now-7d TO *]' },
            { text: 'Not Archived', query: 'NOT archived:true' }
        ];

        menu.innerHTML = suggestions.map(item => `
            <li><a class="dropdown-item" href="#" data-suggestion="${item.query}">
                ${item.text}
                <small class="text-muted d-block">${item.query}</small>
            </a></li>
        `).join('');
    }

    isValidLuceneQuery(query) {
        // Simplified validation - you'd use your luqum library here
        if (!query.trim()) return true;
        
        // Basic bracket matching
        const openBrackets = (query.match(/\[/g) || []).length;
        const closeBrackets = (query.match(/\]/g) || []).length;
        const openParens = (query.match(/\(/g) || []).length;
        const closeParens = (query.match(/\)/g) || []).length;
        
        return openBrackets === closeBrackets && openParens === closeParens;
    }

    previewQuery(query) {
        // Could show query preview in an alert or tooltip
        console.log('Query preview:', query);
    }
}

customElements.define('filter-control', FilterControl);
export { FilterControl };