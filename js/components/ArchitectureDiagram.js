/**
 * Architecture Diagram with Bootstrap styling and components
 */
import { BootstrapWebComponent } from '../core/BootstrapWebComponent.js';
import { EventUtils } from '../utils/EventUtils.js';

class ArchitectureDiagram extends BootstrapWebComponent {
    static get observedAttributes() {
        return ['data-src', 'data-theme'];
    }

    onConnected() {
        this.render();
        this.setupEventListeners();
        this.setupBootstrapComponents();
    }

    render() {
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    width: 100%;
                    height: 100%;
                }
                
                /* Custom styles that work with Bootstrap */
                .diagram-svg {
                    background: var(--bs-body-bg, #ffffff);
                }
                
                .node {
                    cursor: pointer;
                    transition: all 0.3s ease;
                }
                
                .node:hover {
                    transform: scale(1.05);
                }
                
                .node:focus {
                    outline: 2px solid var(--bs-primary, #0d6efd);
                    outline-offset: 2px;
                }
            </style>
            
            <!-- Bootstrap Card for diagram container -->
            <div class="card h-100">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h5 class="card-title mb-0">Architecture Diagram</h5>
                    
                    <!-- Bootstrap Dropdown for controls -->
                    <div class="dropdown">
                        <button class="btn btn-outline-secondary btn-sm dropdown-toggle" 
                                type="button" 
                                id="diagramControls" 
                                data-bs-toggle="dropdown">
                            <i class="bi bi-gear"></i> Controls
                        </button>
                        <ul class="dropdown-menu">
                            <li><a class="dropdown-item" href="#" data-action="reset">Reset View</a></li>
                            <li><a class="dropdown-item" href="#" data-action="export">Export SVG</a></li>
                            <li><hr class="dropdown-divider"></li>
                            <li><a class="dropdown-item" href="#" data-action="help">Help</a></li>
                        </ul>
                    </div>
                </div>
                
                <div class="card-body p-0 position-relative">
                    <svg class="diagram-svg w-100 h-100" role="img" aria-labelledby="diagram-title">
                        <title id="diagram-title">System Architecture Diagram</title>
                        <g class="diagram-content"></g>
                    </svg>
                    
                    <!-- Bootstrap Offcanvas for property panel -->
                    <div class="offcanvas offcanvas-end" 
                         tabindex="-1" 
                         id="propertyOffcanvas">
                        <div class="offcanvas-header">
                            <h5 class="offcanvas-title">Component Details</h5>
                            <button type="button" 
                                    class="btn-close" 
                                    data-bs-dismiss="offcanvas"></button>
                        </div>
                        <div class="offcanvas-body">
                            <div id="propertyContent">
                                <div class="text-muted">Select a component to view details</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Bootstrap Toast for notifications -->
            <div class="toast-container position-fixed bottom-0 end-0 p-3">
                <div id="notificationToast" class="toast" role="alert">
                    <div class="toast-header">
                        <strong class="me-auto">Diagram</strong>
                        <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
                    </div>
                    <div class="toast-body"></div>
                </div>
            </div>
        `;
    }

    setupBootstrapComponents() {
        // Initialize Bootstrap dropdown
        const dropdown = this.shadowRoot.querySelector('#diagramControls');
        this.createDropdown(dropdown);

        // Initialize Bootstrap offcanvas
        const offcanvas = this.shadowRoot.querySelector('#propertyOffcanvas');
        this.propertyPanel = this.createOffcanvas(offcanvas);

        // Initialize Bootstrap toast
        const toast = this.shadowRoot.querySelector('#notificationToast');
        this.notificationToast = this.createToast(toast);
    }

    createOffcanvas(element, options = {}) {
        // Bootstrap 5.3 Offcanvas
        const { Offcanvas } = window.bootstrap;
        const offcanvas = new Offcanvas(element, options);
        this._bootstrapInstances.set(element, offcanvas);
        this.addCleanup(() => offcanvas.dispose());
        return offcanvas;
    }

    createToast(element, options = {}) {
        // Bootstrap 5.3 Toast
        const { Toast } = window.bootstrap;
        const toast = new Toast(element, options);
        this._bootstrapInstances.set(element, toast);
        this.addCleanup(() => toast.dispose());
        return toast;
    }

    selectNode(nodeElement) {
        const nodeId = nodeElement.dataset.nodeId;
        const nodeData = this.nodes.get(nodeId);

        if (nodeData) {
            this.updatePropertyPanel(nodeData);
            this.propertyPanel.show(); // Bootstrap offcanvas
            this.showNotification(`Selected: ${nodeData.name}`, 'success');
        }
    }

    updatePropertyPanel(nodeData) {
        const content = this.shadowRoot.getElementById('propertyContent');
        
        // Use Bootstrap components for property display
        content.innerHTML = `
            <div class="mb-3">
                <h6 class="text-primary">${nodeData.name}</h6>
                <span class="badge bg-secondary">${nodeData.type}</span>
            </div>
            
            <div class="mb-3">
                <h6>Technology</h6>
                <p class="text-muted mb-0">${nodeData.technology}</p>
            </div>
            
            <div class="mb-3">
                <h6>Status</h6>
                <span class="badge ${this.getStatusBadgeClass(nodeData.status)}">
                    ${nodeData.status}
                </span>
            </div>
            
            ${nodeData.properties ? `
                <div class="mb-3">
                    <h6>Properties</h6>
                    <div class="list-group list-group-flush">
                        ${Object.entries(nodeData.properties).map(([key, value]) => `
                            <div class="list-group-item px-0 py-2 border-0">
                                <div class="d-flex justify-content-between">
                                    <small class="text-muted">${key}:</small>
                                    <small class="fw-medium">${value}</small>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            
            <div class="d-grid gap-2">
                <button class="btn btn-outline-primary btn-sm" data-action="edit">
                    <i class="bi bi-pencil"></i> Edit
                </button>
                <button class="btn btn-outline-secondary btn-sm" data-action="connections">
                    <i class="bi bi-diagram-3"></i> View Connections
                </button>
            </div>
        `;
    }

    getStatusBadgeClass(status) {
        const statusMap = {
            'running': 'bg-success',
            'stopped': 'bg-danger',
            'pending': 'bg-warning',
            'unknown': 'bg-secondary'
        };
        return statusMap[status?.toLowerCase()] || 'bg-secondary';
    }

    showNotification(message, type = 'info') {
        const toast = this.shadowRoot.querySelector('#notificationToast');
        const body = toast.querySelector('.toast-body');
        
        body.textContent = message;
        
        // Add Bootstrap color classes
        toast.className = `toast text-bg-${type}`;
        
        this.notificationToast.show();
    }

    setupEventListeners() {
        // Bootstrap dropdown actions
        this.addCleanup(
            EventUtils.delegate(this.shadowRoot, '[data-action]', 'click', (e) => {
                e.preventDefault();
                const action = e.target.dataset.action;
                this.handleAction(action);
            })
        );

        // Node selection
        this.addCleanup(
            EventUtils.delegate(this.shadowRoot, '.node', 'click', (e) => {
                this.selectNode(e.currentTarget);
            })
        );
    }

    handleAction(action) {
        switch (action) {
            case 'reset':
                this.resetView();
                this.showNotification('View reset', 'info');
                break;
            case 'export':
                this.exportSVG();
                this.showNotification('Exporting diagram...', 'info');
                break;
            case 'help':
                this.showHelp();
                break;
        }
    }
}

customElements.define('architecture-diagram', ArchitectureDiagram);