/**
 * Filter Control using Bootstrap form components
 */
import { BootstrapWebComponent } from '../core/BootstrapWebComponent.js';

class FilterControl extends BootstrapWebComponent {
    onConnected() {
        this.render();
        this.setupValidation();
    }

    render() {
        this.shadowRoot.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <h6 class="card-title mb-0">
                        <i class="bi bi-funnel"></i> Advanced Filters
                    </h6>
                </div>
                <div class="card-body">
                    <form id="filterForm" novalidate>
                        <!-- Search Input with Bootstrap styling -->
                        <div class="mb-3">
                            <label for="searchInput" class="form-label">Search Query</label>
                            <div class="input-group">
                                <span class="input-group-text">
                                    <i class="bi bi-search"></i>
                                </span>
                                <input type="text" 
                                       class="form-control" 
                                       id="searchInput"
                                       placeholder="field:value AND (range:[1 TO 10] OR status:active)"
                                       required>
                                <button class="btn btn-outline-secondary dropdown-toggle" 
                                        type="button" 
                                        data-bs-toggle="dropdown">
                                    Suggestions
                                </button>
                                <ul class="dropdown-menu dropdown-menu-end">
                                    <li><a class="dropdown-item" href="#" data-suggestion="status:active">Active Items</a></li>
                                    <li><a class="dropdown-item" href="#" data-suggestion="priority:high">High Priority</a></li>
                                    <li><hr class="dropdown-divider"></li>
                                    <li><a class="dropdown-item" href="#" data-suggestion="created:[now-7d TO *]">Last 7 Days</a></li>
                                </ul>
                            </div>
                            <div class="invalid-feedback"></div>
                            <div class="form-text">Use Lucene syntax for advanced queries</div>
                        </div>

                        <!-- Quick Filters with Bootstrap Button Group -->
                        <div class="mb-3">
                            <label class="form-label">Quick Filters</label>
                            <div class="btn-group-vertical d-grid gap-2" role="group">
                                <input type="checkbox" class="btn-check" id="filter-active" data-filter="status:active">
                                <label class="btn btn-outline-success" for="filter-active">
                                    <i class="bi bi-check-circle"></i> Active Items
                                </label>
                                
                                <input type="checkbox" class="btn-check" id="filter-priority" data-filter="priority:high">
                                <label class="btn btn-outline-warning" for="filter-priority">
                                    <i class="bi bi-exclamation-triangle"></i> High Priority
                                </label>
                                
                                <input type="checkbox" class="btn-check" id="filter-recent" data-filter="created:[now-7d TO *]">
                                <label class="btn btn-outline-info" for="filter-recent">
                                    <i class="bi bi-calendar"></i> Last 7 Days
                                </label>
                            </div>
                        </div>

                        <!-- Action Buttons -->
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

            <!-- Results Preview with Bootstrap Alert -->
            <div id="resultsPreview" class="alert alert-info mt-3 d-none" role="alert">
                <h6 class="alert-heading">
                    <i class="bi bi-info-circle"></i> Query Preview
                </h6>
                <div id="queryPreview"></div>
                <hr>
                <div class="mb-0">
                    <small class="text-muted">Press Enter to execute or click Search</small>
                </div>
            </div>
        `;
    }

    setupValidation() {
        const form = this.shadowRoot.getElementById('filterForm');
        const searchInput = this.shadowRoot.getElementById('searchInput');

        // Bootstrap form validation
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            e.stopPropagation();

            if (form.checkValidity()) {
                this.executeSearch();
            }

            form.classList.add('was-validated');
        });

        // Real-time validation
        searchInput.addEventListener('input', () => {
            this.validateQuery(searchInput.value);
        });
    }

    validateQuery(query) {
        const searchInput = this.shadowRoot.getElementById('searchInput');
        const feedback = searchInput.nextElementSibling;

        try {
            // Validate Lucene syntax (simplified)
            const isValid = this.isValidLuceneQuery(query);
            
            if (isValid) {
                searchInput.classList.remove('is-invalid');
                searchInput.classList.add('is-valid');
                this.showQueryPreview(query);
            } else {
                searchInput.classList.remove('is-valid');
                searchInput.classList.add('is-invalid');
                feedback.textContent = 'Invalid query syntax';
            }
        } catch (error) {
            searchInput.classList.add('is-invalid');
            feedback.textContent = error.message;
        }
    }

    showQueryPreview(query) {
        const preview = this.shadowRoot.getElementById('resultsPreview');
        const content = this.shadowRoot.getElementById('queryPreview');
        
        content.innerHTML = `<code>${this.highlightSyntax(query)}</code>`;
        preview.classList.remove('d-none');
    }
}

customElements.define('filter-control', FilterControl);