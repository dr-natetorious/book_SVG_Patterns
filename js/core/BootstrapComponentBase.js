/**
 * Bootstrap-aware Web Component base class
 * Assumes Bootstrap 5.3 is already loaded in base template
 */
import { BootstrapUtils } from './BootstrapUtils.js';

class BootstrapWebComponent extends HTMLElement {
    constructor() {
        super();
        this.state = new Map();
        this._cleanup = new Set();
        this._bootstrapInstances = new Set(); // Track instances for cleanup
    }

    connectedCallback() {
        this.onConnected();
    }

    disconnectedCallback() {
        this.cleanup();
        this.onDisconnected();
    }

    attributeChangedCallback(name, oldValue, newValue) {
        this.onAttributeChanged?.(name, oldValue, newValue);
    }

    // Lifecycle hooks for subclasses
    onConnected() {}
    onDisconnected() {}

    // State management with Bootstrap events
    setState(key, value) {
        const oldValue = this.state.get(key);
        this.state.set(key, value);
        this.onStateChange?.(key, value, oldValue);
        
        // Emit Bootstrap-compatible event
        this.dispatchEvent(new CustomEvent('state.changed', {
            detail: { key, value, oldValue },
            bubbles: true
        }));
    }

    getState(key) {
        return this.state.get(key);
    }

    // Bootstrap component creation with cleanup tracking
    createModal(element, options = {}) {
        const modal = BootstrapUtils.createModal(element, options);
        this._bootstrapInstances.add(element);
        return modal;
    }

    createOffcanvas(element, options = {}) {
        const offcanvas = BootstrapUtils.createOffcanvas(element, options);
        this._bootstrapInstances.add(element);
        return offcanvas;
    }

    createDropdown(element, options = {}) {
        const dropdown = BootstrapUtils.createDropdown(element, options);
        this._bootstrapInstances.add(element);
        return dropdown;
    }

    createTooltip(element, options = {}) {
        const tooltip = BootstrapUtils.createTooltip(element, options);
        this._bootstrapInstances.add(element);
        return tooltip;
    }

    createToast(element, options = {}) {
        const toast = BootstrapUtils.createToast(element, options);
        this._bootstrapInstances.add(element);
        return toast;
    }

    // Utility methods
    showToast(message, type = 'primary', options = {}) {
        return BootstrapUtils.showToast(message, type, options);
    }

    showAlert(message, type = 'primary', dismissible = true) {
        const container = this.shadowRoot || this;
        const alertContainer = container.querySelector('.alert-container') || container;
        return BootstrapUtils.showAlert(alertContainer, message, type, dismissible);
    }

    setLoadingState(button, loading = true) {
        BootstrapUtils.setLoadingState(button, loading);
    }

    // Cleanup management
    addCleanup(fn) {
        this._cleanup.add(fn);
    }

    cleanup() {
        // Dispose Bootstrap instances
        BootstrapUtils.disposeAll(this._bootstrapInstances);
        this._bootstrapInstances.clear();
        
        // Run custom cleanup functions
        this._cleanup.forEach(fn => fn());
        this._cleanup.clear();
    }

    // CSS utilities (delegate to BootstrapUtils)
    addClass(element, ...classes) {
        return BootstrapUtils.addClass(element, ...classes);
    }

    removeClass(element, ...classes) {
        return BootstrapUtils.removeClass(element, ...classes);
    }

    toggleClass(element, className, force) {
        return BootstrapUtils.toggleClass(element, className, force);
    }

    // Form utilities
    validateForm(form) {
        return BootstrapUtils.validateForm(form);
    }

    setValidationState(element, isValid, message = '') {
        BootstrapUtils.setValidationState(element, isValid, message);
    }
}

export { BootstrapWebComponent };