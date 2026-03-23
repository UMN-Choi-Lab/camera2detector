/**
 * ROI Tool: standalone page orchestrator for editing ROIs on I-94 cameras.
 * Self-contained — no dependency on app.js or camera_panel.js.
 */

(function () {
    // ─── 30 I-94 cameras (west → east) ───
    const CAMERAS = [
        { id: 'C805', desc: 'I-94 EB E of Brockton Ln' },
        { id: 'C809', desc: 'I-94 EB (Elm Creek)' },
        { id: 'C811', desc: 'I-94 EB @ I-494 SB' },
        { id: 'C813', desc: 'I-94 EB @ U.S.169' },
        { id: 'C814', desc: 'I-94 EB @ Boone Ave' },
        { id: 'C816', desc: 'I-94 EB @ Zane Ave' },
        { id: 'C818', desc: 'I-94 WB @ Xerxes Ave' },
        { id: 'C820', desc: 'I-94 EB @ 57th Ave' },
        { id: 'C822', desc: 'I-94 EB @ 42nd Ave' },
        { id: 'C824', desc: 'I-94 NB @ Broadway Ave' },
        { id: 'C832', desc: 'I-94 EB @ Lyndale Ave' },
        { id: 'C834', desc: 'I-94 WB @ T.H.65' },
        { id: 'C836', desc: 'I-94 EB E of Cedar Ave' },
        { id: 'C838', desc: 'I-94 WB @ Huron Blvd' },
        { id: 'C839', desc: 'I-94 EB @ Franklin Ave' },
        { id: 'C840', desc: 'I-94 EB W of T.H.280' },
        { id: 'C841', desc: 'I-94 EB @ T.H.280' },
        { id: 'C843', desc: 'I-94 EB @ Prior Ave' },
        { id: 'C844', desc: 'I-94 EB @ Snelling Ave' },
        { id: 'C845', desc: 'I-94 EB @ Hamline Ave' },
        { id: 'C848', desc: 'I-94 EB @ Western Ave' },
        { id: 'C850', desc: 'I-94 EB @ Wabasha St' },
        { id: 'C852', desc: 'I-94 WB @ I-35E NB' },
        { id: 'C855', desc: 'I-94 EB @ Mounds Blvd' },
        { id: 'C856', desc: 'I-94 EB @ Johnson Pkwy' },
        { id: 'C858', desc: 'I-94 EB @ White Bear Ave' },
        { id: 'C860', desc: 'I-94 EB @ T.H.120' },
        { id: 'C862', desc: 'I-94 EB @ Co Rd 13' },
        { id: 'C865', desc: 'I-94 WB @ Manning Ave' },
        { id: 'C866', desc: 'I-94 WB @ Co Rd 71' },
    ];

    const ROI_COLORS = ['#a855f7', '#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#ec4899'];

    // ─── State ───
    let currentIndex = -1;
    let currentROIs = null;     // CameraROIs object
    let drawMode = false;
    let closedPolygon = null;   // pending polygon to save
    let roiStatusMap = {};      // camera_id -> boolean (has ROIs)

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
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');

    // ─── Init ───
    buildCameraList();
    initROIEditor();
    bindEvents();
    // Pre-check which cameras have ROIs
    checkAllROIStatus();

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

        // Exit draw mode if active
        if (drawMode) exitDrawMode();

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
            const poly = roi.polygon;
            if (!poly || poly.length < 3) return;

            const color = roi.color || '#a855f7';

            // Fill
            ctx.save();
            ctx.globalAlpha = 0.15;
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
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(poly[0][0] * scaleX, poly[0][1] * scaleY);
            for (let i = 1; i < poly.length; i++) {
                ctx.lineTo(poly[i][0] * scaleX, poly[i][1] * scaleY);
            }
            ctx.closePath();
            ctx.stroke();

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
            const item = document.createElement('div');
            item.className = 'roi-item';
            item.innerHTML = `
                <span class="roi-item-label">
                    <span class="roi-item-swatch" style="background:${roi.color || '#a855f7'}"></span>
                    ${roi.road_name} ${roi.direction}
                </span>
                <button class="roi-item-delete" data-roi-id="${roi.roi_id}" title="Delete">x</button>
            `;
            ctrlRoiList.appendChild(item);
        });

        // Delete handlers
        ctrlRoiList.querySelectorAll('.roi-item-delete').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const roiId = e.target.dataset.roiId;
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

        // Attach events (same as ROIEditor.init but for our elements)
        canvasEl.addEventListener('click', (e) => ROIEditor._onClick(e));
        canvasEl.addEventListener('mousemove', (e) => {
            if (ROIEditor.active && ROIEditor.vertices.length > 0) {
                // Redraw everything, then draw in-progress polygon
                drawAllROIs();
                if (closedPolygon) drawClosedPolygonPreview(closedPolygon);
                ROIEditor._drawPreview(ROIEditor._getCanvasCoords(e));
            }
        });
    }

    function enterDrawMode() {
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
                    if (drawMode) {
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
