/**
 * ROI Editor: interactive polygon drawing on the camera canvas overlay.
 */

const ROIEditor = {
    active: false,
    vertices: [],     // current polygon being drawn: [{x, y}, ...]
    canvasEl: null,
    ctx: null,
    imageEl: null,
    onPolygonClosed: null,  // callback(polygon) when polygon is completed

    init() {
        this.canvasEl = document.getElementById('cv-overlay');
        this.ctx = this.canvasEl.getContext('2d');
        this.imageEl = document.getElementById('camera-image');

        this.canvasEl.addEventListener('click', (e) => this._onClick(e));
        this.canvasEl.addEventListener('mousemove', (e) => this._onMouseMove(e));
    },

    startEditing(onPolygonClosed) {
        this.active = true;
        this.vertices = [];
        this.onPolygonClosed = onPolygonClosed;
        this.canvasEl.style.pointerEvents = 'auto';
        this.canvasEl.style.cursor = 'crosshair';
    },

    stopEditing() {
        this.active = false;
        this.vertices = [];
        this.onPolygonClosed = null;
        this.canvasEl.style.pointerEvents = 'none';
        this.canvasEl.style.cursor = 'default';
    },

    cancelCurrentPolygon() {
        this.vertices = [];
    },

    _getCanvasCoords(e) {
        const rect = this.canvasEl.getBoundingClientRect();
        const scaleX = this.canvasEl.width / rect.width;
        const scaleY = this.canvasEl.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    },

    _onClick(e) {
        if (!this.active) return;
        e.stopPropagation();

        const pt = this._getCanvasCoords(e);

        // Check if clicking near first vertex to close polygon
        if (this.vertices.length >= 3) {
            const first = this.vertices[0];
            const dist = Math.hypot(pt.x - first.x, pt.y - first.y);
            if (dist < 15) {
                // Close polygon
                const polygon = this.vertices.map(v => [v.x, v.y]);
                this.vertices = [];
                if (this.onPolygonClosed) {
                    this.onPolygonClosed(polygon);
                }
                return;
            }
        }

        this.vertices.push(pt);
    },

    _onMouseMove(e) {
        if (!this.active || this.vertices.length === 0) return;
        // Redraw preview will be handled by the draw loop
        this._drawPreview(this._getCanvasCoords(e));
    },

    _drawPreview(mousePos) {
        // This draws just the in-progress polygon preview
        // The main draw loop (drawROIs + drawBoxes) handles the rest
        const ctx = this.ctx;
        const verts = this.vertices;
        if (verts.length === 0) return;

        // Draw existing vertices and lines
        ctx.save();
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(verts[0].x, verts[0].y);
        for (let i = 1; i < verts.length; i++) {
            ctx.lineTo(verts[i].x, verts[i].y);
        }
        if (mousePos) {
            ctx.lineTo(mousePos.x, mousePos.y);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw vertices
        verts.forEach((v, i) => {
            ctx.beginPath();
            ctx.arc(v.x, v.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = i === 0 ? '#10b981' : '#fbbf24';  // first vertex green
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.stroke();
        });

        ctx.restore();
    },

    /**
     * Draw in-progress polygon on the canvas (call from animation/redraw loop).
     */
    drawInProgress() {
        if (!this.active || this.vertices.length === 0) return;
        this._drawPreview(null);
    },

    hasVertices() {
        return this.vertices.length > 0;
    },
};
