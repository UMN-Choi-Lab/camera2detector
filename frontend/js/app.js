/**
 * Main application: wire everything together.
 * Includes road data, direction detection, and historical ClickHouse overlay.
 */

(async function () {
    // Initialize components
    CameraMap.init();
    CameraPanel.init();
    Charts.init();

    ROIEditor.init();

    // State
    let currentCameraId = null;
    let currentDetectors = [];
    let currentRoads = [];
    let roadColorMap = {};
    let currentROIs = null;

    // Load cameras from backend
    let cameras = [];
    try {
        const resp = await fetch('/api/cameras?road=I-94');
        cameras = await resp.json();
    } catch (err) {
        console.error('Failed to load cameras:', err);
        return;
    }

    // Add cameras to map
    CameraMap.loadCameras(cameras, onCameraClick);

    // Set up history button
    document.getElementById('history-load-btn').addEventListener('click', loadHistorical);

    // ROI button handlers
    document.getElementById('roi-generate-btn').addEventListener('click', generateROIs);
    document.getElementById('roi-edit-btn').addEventListener('click', toggleEditMode);
    document.getElementById('roi-save-polygon-btn').addEventListener('click', saveCurrentPolygon);
    document.getElementById('roi-cancel-polygon-btn').addEventListener('click', cancelCurrentPolygon);
    document.getElementById('roi-done-btn').addEventListener('click', doneEditing);

    async function onCameraClick(cam) {
        // Stop previous stream if switching cameras
        if (currentCameraId && currentCameraId !== cam.id) {
            CameraPanel.stopMjpegStream();
        }
        currentCameraId = cam.id;

        // Fetch camera details with matched detectors
        currentDetectors = [];
        try {
            const resp = await fetch(`/api/camera/${cam.id}/detectors`);
            const data = await resp.json();
            currentDetectors = data.detectors || [];
        } catch (err) {
            console.error('Failed to load detectors:', err);
        }

        // Fetch nearby roads
        currentRoads = [];
        roadColorMap = {};
        try {
            const resp = await fetch(`/api/camera/${cam.id}/roads`);
            const data = await resp.json();
            currentRoads = data.roads || [];
        } catch (err) {
            console.error('Failed to load roads:', err);
        }

        // Fetch ROIs
        currentROIs = null;
        try {
            const roiResp = await fetch(`/api/camera/${cam.id}/rois`);
            const roiData = await roiResp.json();
            if (roiData.rois && roiData.rois.length > 0) {
                currentROIs = roiData;
            }
        } catch (err) {
            console.error('Failed to load ROIs:', err);
        }

        // Set ROIs on panel
        CameraPanel.setROIs(currentROIs);

        // Show ROI section
        const roiSection = document.getElementById('roi-section');
        roiSection.classList.remove('hidden');
        updateROIList();

        // Update map
        CameraMap.selectCamera(cam.id, currentDetectors);

        // Draw road centerlines on map
        if (currentRoads.length > 0) {
            roadColorMap = CameraMap.drawRoads(currentRoads);
        } else {
            CameraMap.clearRoads();
        }

        // Set color maps on panel and charts
        CameraPanel.setRoadColorMap(roadColorMap);
        CameraPanel.updateRoadLegend(roadColorMap);
        Charts.setRoadColorMap(roadColorMap);

        // Show camera panel
        CameraPanel.show(cam.id, cam.name, currentDetectors);

        // Show history section if detectors available
        const histSection = document.getElementById('history-section');
        if (currentDetectors.length > 0) {
            histSection.classList.remove('hidden');
            // Default to yesterday
            const yesterday = new Date();
            yesterday.setDate(yesterday.getDate() - 1);
            document.getElementById('history-date').value = yesterday.toISOString().split('T')[0];
        } else {
            histSection.classList.add('hidden');
        }

        // Reset charts
        Charts.reset();

        // Start MJPEG live stream (annotations rendered server-side, perfectly synced)
        CameraPanel.startMjpegStream(cam.id);

        // Fetch ClearGuide speed for comparison
        fetchClearGuideSpeed(cam.id);

        // Connect SSE
        SSEClient.connect(cam.id, onSSEUpdate);
    }

    async function fetchClearGuideSpeed(cameraId) {
        try {
            const resp = await fetch(`/api/camera/${cameraId}/clearguide/speed?hours=1`);
            if (!resp.ok) return;
            const data = await resp.json();
            if (data.data && data.data.length > 0) {
                Charts.setClearGuideOverlay(data.data);
                if (data.link_title) {
                    console.log(`ClearGuide link: ${data.link_title} (${data.link_id})`);
                }
            }
        } catch (err) {
            console.debug('ClearGuide speed not available:', err.message);
        }
    }

    function onSSEUpdate(data) {
        // Video + tracking overlay handled by HLS stream + tracking SSE
        // This SSE handler only processes 30s interval data (stats, charts)

        // Use station-level aggregates for detector comparison
        const stations = data.stations || [];

        // Update stat cards (pass interval + stations)
        CameraPanel.updateStats(data.cv, stations, data.interval);

        // Update stream info bar
        CameraPanel.updateStreamInfo(data.interval);

        // Update road breakdown — combine CV and detector per-direction
        if (data.interval && data.interval.detectors && data.interval.detectors.length > 0) {
            CameraPanel.updateRoadBreakdown(data.interval.detectors, stations);
        } else if (data.cv.road_counts) {
            CameraPanel.updateRoadBreakdown(data.cv.road_counts, stations);
        }

        // Station totals: sum volumes, average occ/speed across stations
        const stationVols = stations.map(s => s.volume).filter(v => v != null);
        const detVol = stationVols.length > 0 ? stationVols.reduce((a, b) => a + b, 0) : null;
        const detOcc = Utils.avg(stations.map(s => s.occupancy));
        const detSpeed = Utils.avg(stations.map(s => s.speed));

        Charts.addDataPoint(
            data.timestamp,
            data.cv.vehicle_count,
            data.cv.occupancy,
            detVol,
            detOcc,
        );

        // Speed chart: MnDOT detector speed + CV speed from interval
        let cvSpeed = null;
        if (data.interval && data.interval.detectors) {
            const speeds = data.interval.detectors.map(d => d.speed).filter(s => s != null);
            if (speeds.length > 0) {
                cvSpeed = speeds.reduce((a, b) => a + b, 0) / speeds.length;
            }
        }
        Charts.addSpeedDataPoint(data.timestamp, detSpeed, cvSpeed);

        // Add per-road data to charts
        if (data.interval && data.interval.detectors && data.interval.detectors.length > 0) {
            const roadCounts = data.interval.detectors.map(d => ({
                road_name: d.road_name,
                direction: d.direction,
                vehicle_count: d.volume,
            }));
            Charts.addRoadDataPoint(roadCounts);
        } else if (data.cv.road_counts) {
            Charts.addRoadDataPoint(data.cv.road_counts);
        }
    }

    async function loadHistorical() {
        const dateStr = document.getElementById('history-date').value;
        if (!dateStr || currentDetectors.length === 0) return;

        const btn = document.getElementById('history-load-btn');
        btn.disabled = true;
        btn.textContent = 'Loading...';

        // Use first detector for historical comparison
        const detId = currentDetectors[0].id;
        const nextDay = new Date(dateStr);
        nextDay.setDate(nextDay.getDate() + 1);
        const endStr = nextDay.toISOString().split('T')[0];

        try {
            // Fetch historical volume
            const volResp = await fetch(
                `/api/history/${detId}?type=v30&start=${dateStr}&end=${endStr}&interval=5`
            );
            const volData = await volResp.json();

            if (volData.data && volData.data.length > 0) {
                Charts.addHistoricalOverlay(
                    `Hist ${dateStr} vol`,
                    volData.data,
                    'count',
                );
            }

            // Fetch historical occupancy
            const occResp = await fetch(
                `/api/history/${detId}?type=o30&start=${dateStr}&end=${endStr}&interval=5`
            );
            const occData = await occResp.json();

            if (occData.data && occData.data.length > 0) {
                Charts.addHistoricalOverlay(
                    `Hist ${dateStr} occ`,
                    occData.data,
                    'occupancy',
                );
            }
        } catch (err) {
            console.error('Failed to load historical data:', err);
        }

        btn.disabled = false;
        btn.textContent = 'Load';
    }

    // --- ROI Functions ---

    async function generateROIs() {
        if (!currentCameraId) return;

        const btn = document.getElementById('roi-generate-btn');
        const status = document.getElementById('roi-status');
        btn.disabled = true;
        btn.textContent = 'Generating...';
        status.textContent = 'Sending image to GPT-4o...';

        try {
            const resp = await fetch(`/api/camera/${currentCameraId}/rois/generate`, {
                method: 'POST',
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Generation failed');
            }
            const data = await resp.json();
            currentROIs = data;
            CameraPanel.setROIs(currentROIs);
            updateROIList();
            status.textContent = `Generated ${data.rois.length} ROI(s)`;

            // Redraw overlay with ROIs
            CameraPanel.clearOverlay();
            if (currentROIs && currentROIs.rois) {
                CameraPanel.drawROIs(currentROIs.rois, currentROIs.image_width, currentROIs.image_height);
            }
        } catch (err) {
            console.error('ROI generation failed:', err);
            status.textContent = `Error: ${err.message}`;
        }

        btn.disabled = false;
        btn.textContent = 'Auto-Generate';
    }

    function updateROIList() {
        const container = document.getElementById('roi-list');
        container.innerHTML = '';

        if (!currentROIs || !currentROIs.rois || currentROIs.rois.length === 0) {
            container.innerHTML = '<span style="font-size:0.7rem;color:#666">No ROIs defined</span>';
            return;
        }

        currentROIs.rois.forEach(roi => {
            const item = document.createElement('div');
            item.className = 'roi-list-item';
            item.innerHTML = `
                <span class="roi-list-label">
                    <span class="roi-color-swatch" style="background:${roi.color || '#a855f7'}"></span>
                    ${roi.road_name} ${roi.direction}
                </span>
                <button class="roi-delete-btn" data-roi-id="${roi.roi_id}">x</button>
            `;
            container.appendChild(item);
        });

        // Delete handlers
        container.querySelectorAll('.roi-delete-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const roiId = e.target.dataset.roiId;
                try {
                    await fetch(`/api/camera/${currentCameraId}/rois/${roiId}`, { method: 'DELETE' });
                    // Reload ROIs
                    const resp = await fetch(`/api/camera/${currentCameraId}/rois`);
                    currentROIs = await resp.json();
                    if (!currentROIs.rois || currentROIs.rois.length === 0) currentROIs = null;
                    CameraPanel.setROIs(currentROIs);
                    updateROIList();
                    // Redraw
                    CameraPanel.clearOverlay();
                    if (currentROIs && currentROIs.rois) {
                        CameraPanel.drawROIs(currentROIs.rois, currentROIs.image_width, currentROIs.image_height);
                    }
                } catch (err) {
                    console.error('Failed to delete ROI:', err);
                }
            });
        });
    }

    function toggleEditMode() {
        const editor = document.getElementById('roi-editor');
        const btn = document.getElementById('roi-edit-btn');

        if (editor.classList.contains('hidden')) {
            editor.classList.remove('hidden');
            btn.textContent = 'Cancel Edit';
            ROIEditor.startEditing(onPolygonClosed);
        } else {
            doneEditing();
        }
    }

    function onPolygonClosed(polygon) {
        // Polygon closed — enable save button
        document.getElementById('roi-save-polygon-btn').disabled = false;
        // Store temporarily
        ROIEditor._closedPolygon = polygon;

        // Redraw with preview
        CameraPanel.clearOverlay();
        if (currentROIs && currentROIs.rois) {
            CameraPanel.drawROIs(currentROIs.rois, currentROIs.image_width, currentROIs.image_height);
        }
        // Draw closed polygon preview
        const ctx = CameraPanel.ctx;
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

    async function saveCurrentPolygon() {
        const polygon = ROIEditor._closedPolygon;
        if (!polygon || !currentCameraId) return;

        const roadName = document.getElementById('roi-road-name').value.trim();
        const direction = document.getElementById('roi-direction').value;

        if (!roadName) {
            document.getElementById('roi-status').textContent = 'Enter a road name';
            return;
        }

        // Build new ROI
        const newROI = {
            roi_id: Math.random().toString(36).substr(2, 8),
            road_name: roadName,
            direction: direction,
            polygon: polygon,
            color: ['#a855f7', '#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#ec4899'][
                (currentROIs?.rois?.length || 0) % 6
            ],
        };

        // Build full payload
        const imgW = CameraPanel.canvasEl.width;
        const imgH = CameraPanel.canvasEl.height;
        const payload = {
            camera_id: currentCameraId,
            image_width: currentROIs?.image_width || imgW,
            image_height: currentROIs?.image_height || imgH,
            rois: [...(currentROIs?.rois || []), newROI],
            generated_at: new Date().toISOString(),
            source: 'manual',
        };

        try {
            const resp = await fetch(`/api/camera/${currentCameraId}/rois`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            currentROIs = await resp.json();
            CameraPanel.setROIs(currentROIs);
            updateROIList();
            document.getElementById('roi-status').textContent = `Saved: ${roadName} ${direction}`;
        } catch (err) {
            console.error('Failed to save ROI:', err);
            document.getElementById('roi-status').textContent = 'Save failed';
        }

        // Reset editor state for next polygon
        ROIEditor._closedPolygon = null;
        document.getElementById('roi-save-polygon-btn').disabled = true;
        document.getElementById('roi-road-name').value = '';
        document.getElementById('roi-direction').value = '';

        // Redraw
        CameraPanel.clearOverlay();
        if (currentROIs && currentROIs.rois) {
            CameraPanel.drawROIs(currentROIs.rois, currentROIs.image_width, currentROIs.image_height);
        }
    }

    function cancelCurrentPolygon() {
        ROIEditor.cancelCurrentPolygon();
        ROIEditor._closedPolygon = null;
        document.getElementById('roi-save-polygon-btn').disabled = true;

        // Redraw without preview
        CameraPanel.clearOverlay();
        if (currentROIs && currentROIs.rois) {
            CameraPanel.drawROIs(currentROIs.rois, currentROIs.image_width, currentROIs.image_height);
        }
    }

    function doneEditing() {
        ROIEditor.stopEditing();
        document.getElementById('roi-editor').classList.add('hidden');
        document.getElementById('roi-edit-btn').textContent = 'Edit ROIs';

        // Redraw clean
        CameraPanel.clearOverlay();
        if (currentROIs && currentROIs.rois) {
            CameraPanel.drawROIs(currentROIs.rois, currentROIs.image_width, currentROIs.image_height);
        }
    }
})();
