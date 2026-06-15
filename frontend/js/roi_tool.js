/**
 * ROI Tool: standalone page orchestrator for editing ROIs on I-94 cameras.
 * Self-contained — no dependency on app.js or camera_panel.js.
 */

(function () {
    // ─── Curated I-94 camera descriptions (west → east) ───
    // The full camera list is fetched dynamically from /api/cameras?road=I-94.
    // Any ID not in this map uses its raw ID as the label.
    const CURATED_DESCRIPTIONS = {
        'C805': 'I-94 EB E of Brockton Ln',
        'C809': 'I-94 EB (Elm Creek)',
        'C811': 'I-94 EB @ I-494 SB',
        'C813': 'I-94 EB @ U.S.169',
        'C814': 'I-94 EB @ Boone Ave',
        'C816': 'I-94 EB @ Zane Ave',
        'C818': 'I-94 WB @ Xerxes Ave',
        'C820': 'I-94 EB @ 57th Ave',
        'C822': 'I-94 EB @ 42nd Ave',
        'C824': 'I-94 NB @ Broadway Ave',
        'C832': 'I-94 EB @ Lyndale Ave',
        'C834': 'I-94 WB @ T.H.65',
        'C836': 'I-94 EB E of Cedar Ave',
        'C838': 'I-94 WB @ Huron Blvd',
        'C839': 'I-94 EB @ Franklin Ave',
        'C840': 'I-94 EB W of T.H.280',
        'C841': 'I-94 EB @ T.H.280',
        'C843': 'I-94 EB @ Prior Ave',
        'C844': 'I-94 EB @ Snelling Ave',
        'C845': 'I-94 EB @ Hamline Ave',
        'C848': 'I-94 EB @ Western Ave',
        'C850': 'I-94 EB @ Wabasha St',
        'C852': 'I-94 WB @ I-35E NB',
        'C855': 'I-94 EB @ Mounds Blvd',
        'C856': 'I-94 EB @ Johnson Pkwy',
        'C858': 'I-94 EB @ White Bear Ave',
        'C860': 'I-94 EB @ T.H.120',
        'C862': 'I-94 EB @ Co Rd 13',
        'C865': 'I-94 WB @ Manning Ave',
        'C866': 'I-94 WB @ Co Rd 71',
    };

    // Populated by loadCameras(); maintained in west→east (numeric ID) order.
    let CAMERAS = [];

    const ROI_COLORS = ['#a855f7', '#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#ec4899'];

    // ─── State ───
    let currentIndex = -1;
    let currentROIs = null;     // CameraROIs object
    let drawMode = false;
    let closedPolygon = null;   // pending polygon to save
    let roiStatusMap = {};      // camera_id -> boolean (has ROIs)

    // Vertex-edit mode state
    let editMode = false;
    let editRoiId = null;
    let editVertices = null;    // array of {x, y} in image coords
    let draggingIdx = -1;
    let hoverIdx = -1;
    const VERTEX_PICK_RADIUS = 10;   // px in canvas space
    const EDGE_PICK_RADIUS = 8;

    // ─── DOM refs ───
    const imageEl = document.getElementById('roi-image');
    const canvasEl = document.getElementById('roi-canvas');
    const ctx = canvasEl.getContext('2d');
    const imageContainer = document.getElementById('image-container');
    const noCameraMsg = document.getElementById('no-camera-msg');
    const imageStatus = document.getElementById('image-status');
    const cameraListEl = document.getElementById('camera-list');
    const ctrlCamName = document.getElementById('ctrl-cam-name');
    const ctrlStatus = document.getElementById('ctrl-status');
    const ctrlRoiList = document.getElementById('ctrl-roi-list');
    const drawEditor = document.getElementById('draw-editor');

    const btnAutoGen = document.getElementById('btn-auto-generate');
    const btnDrawMode = document.getElementById('btn-draw-mode');
    const btnClearAll = document.getElementById('btn-clear-all');
    const btnSavePoly = document.getElementById('btn-save-polygon');
    const btnCancelPoly = document.getElementById('btn-cancel-polygon');
    const vertexEditor = document.getElementById('vertex-editor');
    const vertexEditorLabel = document.getElementById('vertex-editor-label');
    const btnSaveEdits = document.getElementById('btn-save-edits');
    const btnCancelEdits = document.getElementById('btn-cancel-edits');
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');

    // ─── Init ───
    initROIEditor();
    bindEvents();
    loadCameras().then(() => {
        buildCameraList();
        checkAllROIStatus();
    });

    async function loadCameras() {
        // Fetch I-94 cameras from backend. Filter to the urban corridor
        // (C800-C870) to keep the list manageable; anything outside this
        // range can still appear if it already has an ROI file saved.
        const CORRIDOR_MIN = 800;
        const CORRIDOR_MAX = 870;
        const numeric = (id) => parseInt(id.replace(/\D/g, ''), 10) || 0;

        try {
            const [camResp, roiResp] = await Promise.all([
                fetch('/api/cameras?road=I-94'),
                fetch('/api/cameras/rois/list').catch(() => null),
            ]);
            const list = await camResp.json();

            // Also include any camera that already has an ROI file, even if
            // outside the corridor range.
            let roiIds = new Set();
            if (roiResp && roiResp.ok) {
                try {
                    const rois = await roiResp.json();
                    (rois.cameras || []).forEach(id => roiIds.add(id));
                } catch { /* endpoint may not exist */ }
            }

            const byId = new Map();
            list.forEach(c => {
                const n = numeric(c.id);
                if ((n >= CORRIDOR_MIN && n <= CORRIDOR_MAX) || roiIds.has(c.id)) {
                    byId.set(c.id, c);
                }
            });

            CAMERAS = Array.from(byId.values())
                .map(c => ({
                    id: c.id,
                    desc: CURATED_DESCRIPTIONS[c.id] || c.id,
                }))
                .sort((a, b) => numeric(a.id) - numeric(b.id));

            document.querySelector('#camera-list-header span').textContent =
                `I-94 Cameras (${CAMERAS.length})`;
        } catch (err) {
            console.error('Failed to load cameras:', err);
            // Fallback to curated IDs only
            CAMERAS = Object.keys(CURATED_DESCRIPTIONS)
                .map(id => ({ id, desc: CURATED_DESCRIPTIONS[id] }));
        }
    }

    // ─── Camera list ───
    function buildCameraList() {
        cameraListEl.innerHTML = '';
        CAMERAS.forEach((cam, i) => {
            const el = document.createElement('div');
            el.className = 'cam-item';
            el.dataset.index = i;
            el.innerHTML = `
                <span class="dot" id="dot-${cam.id}"></span>
                <span class="cam-id">${cam.id}</span>
                <span class="cam-desc" title="${cam.desc}">${cam.desc}</span>
            `;
            el.addEventListener('click', () => selectCamera(i));
            cameraListEl.appendChild(el);
        });
    }

    async function checkAllROIStatus() {
        // Fire all checks in parallel
        const promises = CAMERAS.map(async (cam) => {
            try {
                const resp = await fetch(`/api/camera/${cam.id}/rois`);
                const data = await resp.json();
                roiStatusMap[cam.id] = data.rois && data.rois.length > 0;
            } catch {
                roiStatusMap[cam.id] = false;
            }
        });
        await Promise.all(promises);
        updateDots();
    }

    function updateDots() {
        CAMERAS.forEach(cam => {
            const dot = document.getElementById(`dot-${cam.id}`);
            if (dot) {
                dot.classList.toggle('has-rois', !!roiStatusMap[cam.id]);
            }
        });
    }

    function updateCameraListHighlight() {
        cameraListEl.querySelectorAll('.cam-item').forEach((el, i) => {
            el.classList.toggle('active', i === currentIndex);
        });
        // Scroll active into view
        const active = cameraListEl.querySelector('.cam-item.active');
        if (active) active.scrollIntoView({ block: 'nearest' });
    }

    // ─── Select camera ───
    async function selectCamera(index) {
        if (index < 0 || index >= CAMERAS.length) return;

        // Exit any active modes
        if (drawMode) exitDrawMode();
        if (editMode) exitEditMode();

        currentIndex = index;
        const cam = CAMERAS[index];

        updateCameraListHighlight();
        ctrlCamName.textContent = `${cam.id} — ${cam.desc}`;

        // Enable buttons
        btnAutoGen.disabled = false;
        btnDrawMode.disabled = false;
        btnClearAll.disabled = false;

        // Show image container
        noCameraMsg.classList.add('hidden');
        imageContainer.classList.remove('hidden');
        imageStatus.classList.remove('hidden');
        imageStatus.textContent = 'Loading...';

        // Load image
        imageEl.src = `/api/camera/${cam.id}/image?t=${Date.now()}`;
        imageEl.onload = () => {
            canvasEl.width = imageEl.naturalWidth;
            canvasEl.height = imageEl.naturalHeight;
            imageStatus.textContent = `${imageEl.naturalWidth}x${imageEl.naturalHeight}`;
            drawAllROIs();
        };
        imageEl.onerror = () => {
            imageStatus.textContent = 'Failed to load image';
        };

        // Load ROIs
        currentROIs = null;
        ctrlStatus.textContent = '';
        try {
            const resp = await fetch(`/api/camera/${cam.id}/rois`);
            const data = await resp.json();
            if (data.rois && data.rois.length > 0) {
                currentROIs = data;
                ctrlStatus.textContent = `${data.rois.length} ROI(s) — ${data.source || 'unknown'}`;
            } else {
                ctrlStatus.textContent = 'No ROIs';
            }
        } catch {
            ctrlStatus.textContent = 'Failed to load ROIs';
        }

        updateROIList();
        drawAllROIs();
    }

    // ─── Drawing ───
    function clearCanvas() {
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    }

    function drawAllROIs() {
        clearCanvas();
        if (!currentROIs || !currentROIs.rois || currentROIs.rois.length === 0) return;

        const scaleX = canvasEl.width / (currentROIs.image_width || imageEl.naturalWidth);
        const scaleY = canvasEl.height / (currentROIs.image_height || imageEl.naturalHeight);

        currentROIs.rois.forEach(roi => {
            const isEditing = editMode && roi.roi_id === editRoiId;
            const poly = isEditing
                ? editVertices.map(v => [v.x, v.y])
                : roi.polygon;
            if (!poly || poly.length < 3) return;

            const color = roi.color || '#a855f7';

            // Fill
            ctx.save();
            ctx.globalAlpha = isEditing ? 0.25 : 0.15;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(poly[0][0] * scaleX, poly[0][1] * scaleY);
            for (let i = 1; i < poly.length; i++) {
                ctx.lineTo(poly[i][0] * scaleX, poly[i][1] * scaleY);
            }
            ctx.closePath();
            ctx.fill();
            ctx.restore();

            // Border
            ctx.strokeStyle = color;
            ctx.lineWidth = isEditing ? 3 : 2;
            ctx.beginPath();
            ctx.moveTo(poly[0][0] * scaleX, poly[0][1] * scaleY);
            for (let i = 1; i < poly.length; i++) {
                ctx.lineTo(poly[i][0] * scaleX, poly[i][1] * scaleY);
            }
            ctx.closePath();
            ctx.stroke();

            // Vertex handles when editing
            if (isEditing) {
                // Edge midpoint markers (small) for insert hint
                for (let i = 0; i < poly.length; i++) {
                    const j = (i + 1) % poly.length;
                    const mx = ((poly[i][0] + poly[j][0]) / 2) * scaleX;
                    const my = ((poly[i][1] + poly[j][1]) / 2) * scaleY;
                    ctx.beginPath();
                    ctx.arc(mx, my, 3, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(255,255,255,0.6)';
                    ctx.fill();
                }
                // Vertex handles (large, draggable)
                poly.forEach((p, i) => {
                    const px = p[0] * scaleX;
                    const py = p[1] * scaleY;
                    const isHover = i === hoverIdx;
                    const isDrag = i === draggingIdx;
                    ctx.beginPath();
                    ctx.arc(px, py, isHover || isDrag ? 7 : 5, 0, Math.PI * 2);
                    ctx.fillStyle = isDrag ? '#10b981' : (isHover ? '#fbbf24' : '#fff');
                    ctx.fill();
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                });
            }

            // Label at centroid
            let cx = 0, cy = 0;
            poly.forEach(p => { cx += p[0]; cy += p[1]; });
            cx = (cx / poly.length) * scaleX;
            cy = (cy / poly.length) * scaleY;

            const label = `${roi.road_name} ${roi.direction}`;
            ctx.font = 'bold 14px monospace';
            const textW = ctx.measureText(label).width;
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.8;
            ctx.fillRect(cx - textW / 2 - 4, cy - 10, textW + 8, 20);
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = '#fff';
            ctx.fillText(label, cx - textW / 2, cy + 5);
        });
    }

    // ─── Vertex edit helpers ───
    function _imageScale() {
        return {
            sx: canvasEl.width / (currentROIs?.image_width || imageEl.naturalWidth),
            sy: canvasEl.height / (currentROIs?.image_height || imageEl.naturalHeight),
        };
    }

    function _canvasFromEvent(e) {
        const rect = canvasEl.getBoundingClientRect();
        const scaleX = canvasEl.width / rect.width;
        const scaleY = canvasEl.height / rect.height;
        return {
            cx: (e.clientX - rect.left) * scaleX,
            cy: (e.clientY - rect.top) * scaleY,
        };
    }

    function _hitVertex(cx, cy) {
        if (!editVertices) return -1;
        const { sx, sy } = _imageScale();
        for (let i = 0; i < editVertices.length; i++) {
            const px = editVertices[i].x * sx;
            const py = editVertices[i].y * sy;
            if (Math.hypot(cx - px, cy - py) <= VERTEX_PICK_RADIUS) return i;
        }
        return -1;
    }

    function _hitEdgeMidpoint(cx, cy) {
        if (!editVertices) return -1;
        const { sx, sy } = _imageScale();
        const n = editVertices.length;
        for (let i = 0; i < n; i++) {
            const j = (i + 1) % n;
            const mx = ((editVertices[i].x + editVertices[j].x) / 2) * sx;
            const my = ((editVertices[i].y + editVertices[j].y) / 2) * sy;
            if (Math.hypot(cx - mx, cy - my) <= EDGE_PICK_RADIUS) return i;
        }
        return -1;
    }

    function enterEditMode(roiId) {
        const roi = currentROIs.rois.find(r => r.roi_id === roiId);
        if (!roi || !roi.polygon || roi.polygon.length < 3) return;

        // Exit draw mode if active
        if (drawMode) exitDrawMode();

        editMode = true;
        editRoiId = roiId;
        editVertices = roi.polygon.map(p => ({ x: p[0], y: p[1] }));
        draggingIdx = -1;
        hoverIdx = -1;

        vertexEditor.classList.remove('hidden');
        vertexEditorLabel.textContent = `${roi.road_name} ${roi.direction}`;
        canvasEl.style.pointerEvents = 'auto';
        canvasEl.style.cursor = 'default';
        ctrlStatus.textContent = 'Edit: drag vertices, dbl-click edge to insert, shift+click to delete';

        updateROIList();
        drawAllROIs();
    }

    function exitEditMode() {
        editMode = false;
        editRoiId = null;
        editVertices = null;
        draggingIdx = -1;
        hoverIdx = -1;

        vertexEditor.classList.add('hidden');
        canvasEl.style.pointerEvents = 'none';
        canvasEl.style.cursor = 'default';

        updateROIList();
        drawAllROIs();
    }

    async function saveEdits() {
        if (!editMode || !currentROIs) return;
        const cam = CAMERAS[currentIndex];

        // Build updated ROI list: replace the edited one, leave others intact
        const updatedRois = currentROIs.rois.map(r => {
            if (r.roi_id === editRoiId) {
                return { ...r, polygon: editVertices.map(v => [v.x, v.y]) };
            }
            return r;
        });

        const payload = {
            camera_id: cam.id,
            image_width: currentROIs.image_width || canvasEl.width,
            image_height: currentROIs.image_height || canvasEl.height,
            rois: updatedRois,
            generated_at: new Date().toISOString(),
            source: 'manual',
        };

        try {
            const resp = await fetch(`/api/camera/${cam.id}/rois`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            currentROIs = await resp.json();
            ctrlStatus.textContent = 'Edits saved';
            updateROIStatusForCurrent();
        } catch (err) {
            console.error('Failed to save edits:', err);
            ctrlStatus.textContent = 'Save failed';
            return;
        }
        exitEditMode();
    }

    function drawClosedPolygonPreview(polygon) {
        ctx.save();
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'rgba(251, 191, 36, 0.2)';
        ctx.beginPath();
        ctx.moveTo(polygon[0][0], polygon[0][1]);
        for (let i = 1; i < polygon.length; i++) {
            ctx.lineTo(polygon[i][0], polygon[i][1]);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.restore();
    }

    // ─── ROI list ───
    function updateROIList() {
        ctrlRoiList.innerHTML = '';

        if (!currentROIs || !currentROIs.rois || currentROIs.rois.length === 0) {
            ctrlRoiList.innerHTML = '<span class="no-rois-msg">No ROIs defined</span>';
            return;
        }

        currentROIs.rois.forEach(roi => {
            const isEditing = editMode && roi.roi_id === editRoiId;
            const item = document.createElement('div');
            item.className = 'roi-item';
            item.innerHTML = `
                <span class="roi-item-label">
                    <span class="roi-item-swatch" style="background:${roi.color || '#a855f7'}"></span>
                    ${roi.road_name} ${roi.direction}
                </span>
                <span style="display:flex;gap:0.1rem;">
                    <button class="roi-item-edit ${isEditing ? 'active' : ''}" data-roi-id="${roi.roi_id}" title="Edit vertices">edit</button>
                    <button class="roi-item-delete" data-roi-id="${roi.roi_id}" title="Delete">x</button>
                </span>
            `;
            ctrlRoiList.appendChild(item);
        });

        // Edit handlers
        ctrlRoiList.querySelectorAll('.roi-item-edit').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const roiId = e.target.dataset.roiId;
                if (editMode && roiId === editRoiId) {
                    exitEditMode();
                } else {
                    enterEditMode(roiId);
                }
            });
        });

        // Delete handlers
        ctrlRoiList.querySelectorAll('.roi-item-delete').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const roiId = e.target.dataset.roiId;
                if (editMode && roiId === editRoiId) exitEditMode();
                const cam = CAMERAS[currentIndex];
                try {
                    await fetch(`/api/camera/${cam.id}/rois/${roiId}`, { method: 'DELETE' });
                    // Reload
                    const resp = await fetch(`/api/camera/${cam.id}/rois`);
                    const data = await resp.json();
                    currentROIs = (data.rois && data.rois.length > 0) ? data : null;
                    updateROIList();
                    drawAllROIs();
                    updateROIStatusForCurrent();
                    ctrlStatus.textContent = currentROIs
                        ? `${currentROIs.rois.length} ROI(s)`
                        : 'No ROIs';
                } catch (err) {
                    console.error('Failed to delete ROI:', err);
                }
            });
        });
    }

    function updateROIStatusForCurrent() {
        if (currentIndex < 0) return;
        const cam = CAMERAS[currentIndex];
        roiStatusMap[cam.id] = currentROIs && currentROIs.rois && currentROIs.rois.length > 0;
        updateDots();
    }

    // ─── Auto-generate ───
    async function autoGenerate() {
        if (currentIndex < 0) return;
        const cam = CAMERAS[currentIndex];

        btnAutoGen.disabled = true;
        btnAutoGen.innerHTML = '<span class="loading-spinner"></span>Generating...';
        ctrlStatus.textContent = 'Sending image to VLM...';

        try {
            const resp = await fetch(`/api/camera/${cam.id}/rois/generate`, { method: 'POST' });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Generation failed');
            }
            const data = await resp.json();
            currentROIs = data;
            ctrlStatus.textContent = `Generated ${data.rois.length} ROI(s)`;
            updateROIList();
            drawAllROIs();
            updateROIStatusForCurrent();
        } catch (err) {
            console.error('ROI generation failed:', err);
            ctrlStatus.textContent = `Error: ${err.message}`;
        }

        btnAutoGen.disabled = false;
        btnAutoGen.textContent = 'Auto-Generate';
    }

    // ─── Clear all ───
    async function clearAllROIs() {
        if (currentIndex < 0 || !currentROIs) return;
        const cam = CAMERAS[currentIndex];

        // Save empty ROI set
        const payload = {
            camera_id: cam.id,
            image_width: currentROIs.image_width || canvasEl.width,
            image_height: currentROIs.image_height || canvasEl.height,
            rois: [],
            generated_at: new Date().toISOString(),
            source: 'manual',
        };

        try {
            await fetch(`/api/camera/${cam.id}/rois`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            currentROIs = null;
            updateROIList();
            drawAllROIs();
            updateROIStatusForCurrent();
            ctrlStatus.textContent = 'All ROIs cleared';
        } catch (err) {
            console.error('Failed to clear ROIs:', err);
            ctrlStatus.textContent = 'Clear failed';
        }
    }

    // ─── Draw mode ───
    function initROIEditor() {
        // Bind the ROIEditor to our canvas/image elements
        ROIEditor.canvasEl = canvasEl;
        ROIEditor.ctx = ctx;
        ROIEditor.imageEl = imageEl;

        // Click: draw-mode vertex placement OR edit-mode shift-delete
        canvasEl.addEventListener('click', (e) => {
            if (editMode) {
                // Shift-click → delete vertex
                if (e.shiftKey) {
                    const { cx, cy } = _canvasFromEvent(e);
                    const idx = _hitVertex(cx, cy);
                    if (idx >= 0 && editVertices.length > 3) {
                        editVertices.splice(idx, 1);
                        hoverIdx = -1;
                        drawAllROIs();
                    }
                }
                return;  // don't fall through to draw mode
            }
            ROIEditor._onClick(e);
        });

        // Mousemove: draw preview OR edit drag/hover
        canvasEl.addEventListener('mousemove', (e) => {
            if (editMode) {
                const { cx, cy } = _canvasFromEvent(e);
                if (draggingIdx >= 0) {
                    // Drag vertex in image coords
                    const { sx, sy } = _imageScale();
                    editVertices[draggingIdx].x = cx / sx;
                    editVertices[draggingIdx].y = cy / sy;
                    drawAllROIs();
                } else {
                    const newHover = _hitVertex(cx, cy);
                    if (newHover !== hoverIdx) {
                        hoverIdx = newHover;
                        canvasEl.style.cursor = hoverIdx >= 0 ? 'grab' : 'default';
                        drawAllROIs();
                    }
                }
                return;
            }
            if (ROIEditor.active && ROIEditor.vertices.length > 0) {
                drawAllROIs();
                if (closedPolygon) drawClosedPolygonPreview(closedPolygon);
                ROIEditor._drawPreview(ROIEditor._getCanvasCoords(e));
            }
        });

        // Mousedown: begin vertex drag
        canvasEl.addEventListener('mousedown', (e) => {
            if (!editMode || e.shiftKey) return;
            const { cx, cy } = _canvasFromEvent(e);
            const idx = _hitVertex(cx, cy);
            if (idx >= 0) {
                draggingIdx = idx;
                canvasEl.style.cursor = 'grabbing';
                e.preventDefault();
            }
        });

        // Mouseup: end drag (on window so drag survives leaving canvas)
        window.addEventListener('mouseup', () => {
            if (draggingIdx >= 0) {
                draggingIdx = -1;
                canvasEl.style.cursor = hoverIdx >= 0 ? 'grab' : 'default';
                drawAllROIs();
            }
        });

        // Double-click: insert vertex on nearest edge midpoint
        canvasEl.addEventListener('dblclick', (e) => {
            if (!editMode) return;
            const { cx, cy } = _canvasFromEvent(e);
            const edgeIdx = _hitEdgeMidpoint(cx, cy);
            if (edgeIdx >= 0) {
                const i = edgeIdx;
                const j = (i + 1) % editVertices.length;
                const mid = {
                    x: (editVertices[i].x + editVertices[j].x) / 2,
                    y: (editVertices[i].y + editVertices[j].y) / 2,
                };
                editVertices.splice(j, 0, mid);
                drawAllROIs();
            }
        });
    }

    function enterDrawMode() {
        if (editMode) exitEditMode();
        drawMode = true;
        closedPolygon = null;
        btnDrawMode.classList.add('active');
        btnDrawMode.textContent = 'Exit Draw Mode';
        drawEditor.classList.remove('hidden');
        btnSavePoly.disabled = true;
        document.getElementById('draw-road-name').value = '';
        document.getElementById('draw-direction').value = '';
        ctrlStatus.textContent = 'Draw mode: click to place vertices';

        ROIEditor.startEditing(onPolygonClosed);
    }

    function exitDrawMode() {
        drawMode = false;
        closedPolygon = null;
        btnDrawMode.classList.remove('active');
        btnDrawMode.textContent = 'Draw Mode';
        drawEditor.classList.add('hidden');
        btnSavePoly.disabled = true;

        ROIEditor.stopEditing();
        drawAllROIs();
    }

    function onPolygonClosed(polygon) {
        closedPolygon = polygon;
        btnSavePoly.disabled = false;
        ctrlStatus.textContent = 'Polygon closed — enter road info and save';

        // Redraw with preview
        drawAllROIs();
        drawClosedPolygonPreview(polygon);
    }

    async function savePolygon() {
        if (!closedPolygon || currentIndex < 0) return;

        const cam = CAMERAS[currentIndex];
        const roadName = document.getElementById('draw-road-name').value.trim();
        const direction = document.getElementById('draw-direction').value;

        if (!roadName) {
            ctrlStatus.textContent = 'Enter a road name';
            return;
        }

        const newROI = {
            roi_id: Math.random().toString(36).substr(2, 8),
            road_name: roadName,
            direction: direction,
            polygon: closedPolygon,
            color: ROI_COLORS[(currentROIs?.rois?.length || 0) % ROI_COLORS.length],
        };

        const imgW = canvasEl.width;
        const imgH = canvasEl.height;
        const payload = {
            camera_id: cam.id,
            image_width: currentROIs?.image_width || imgW,
            image_height: currentROIs?.image_height || imgH,
            rois: [...(currentROIs?.rois || []), newROI],
            generated_at: new Date().toISOString(),
            source: 'manual',
        };

        try {
            const resp = await fetch(`/api/camera/${cam.id}/rois`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            currentROIs = await resp.json();
            updateROIList();
            updateROIStatusForCurrent();
            ctrlStatus.textContent = `Saved: ${roadName} ${direction}`;
        } catch (err) {
            console.error('Failed to save ROI:', err);
            ctrlStatus.textContent = 'Save failed';
        }

        // Reset for next polygon
        closedPolygon = null;
        btnSavePoly.disabled = true;
        document.getElementById('draw-road-name').value = '';
        document.getElementById('draw-direction').value = '';

        drawAllROIs();
    }

    function cancelPolygon() {
        ROIEditor.cancelCurrentPolygon();
        closedPolygon = null;
        btnSavePoly.disabled = true;
        ctrlStatus.textContent = 'Polygon cancelled — click to start new one';
        drawAllROIs();
    }

    // ─── Events ───
    function bindEvents() {
        btnAutoGen.addEventListener('click', autoGenerate);
        btnDrawMode.addEventListener('click', () => {
            if (currentIndex < 0) return;
            drawMode ? exitDrawMode() : enterDrawMode();
        });
        btnClearAll.addEventListener('click', clearAllROIs);
        btnSavePoly.addEventListener('click', savePolygon);
        btnCancelPoly.addEventListener('click', cancelPolygon);
        btnSaveEdits.addEventListener('click', saveEdits);
        btnCancelEdits.addEventListener('click', exitEditMode);

        btnPrev.addEventListener('click', () => {
            if (currentIndex > 0) selectCamera(currentIndex - 1);
        });
        btnNext.addEventListener('click', () => {
            if (currentIndex < CAMERAS.length - 1) selectCamera(currentIndex + 1);
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Don't trigger when typing in inputs
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

            switch (e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    if (currentIndex > 0) selectCamera(currentIndex - 1);
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    if (currentIndex < CAMERAS.length - 1) selectCamera(currentIndex + 1);
                    break;
                case 'Escape':
                    if (editMode) {
                        exitEditMode();
                    } else if (drawMode) {
                        if (ROIEditor.hasVertices() || closedPolygon) {
                            cancelPolygon();
                        } else {
                            exitDrawMode();
                        }
                    }
                    break;
                case 'd':
                case 'D':
                    if (currentIndex >= 0) {
                        drawMode ? exitDrawMode() : enterDrawMode();
                    }
                    break;
                case 'g':
                case 'G':
                    if (currentIndex >= 0 && !btnAutoGen.disabled) {
                        autoGenerate();
                    }
                    break;
            }
        });
    }
})();
