/**
 * Main application: wire everything together.
 * Includes road data, direction detection, and historical ClickHouse overlay.
 */

(async function () {
    // Initialize components
    CameraMap.init();
    CameraPanel.init();
    Charts.init();

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

    // Calibration button
    document.getElementById('calibrate-btn').addEventListener('click', calibrateCamera);

    // ROI button handlers

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

        // Load calibration overlay on camera view
        loadCalibrationOverlay(cam.id);

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

        // Update camera movement / SSIM display
        if (data.interval) {
            const movedBadge = document.getElementById('stream-moved-badge');
            const ssimEl = document.getElementById('stream-ssim');
            if (data.interval.camera_moved) {
                movedBadge.classList.remove('hidden');
            } else {
                movedBadge.classList.add('hidden');
            }
            if (data.interval.ssim_score != null) {
                ssimEl.textContent = `SSIM: ${data.interval.ssim_score.toFixed(3)}`;
                ssimEl.classList.remove('hidden');
            }
            // Update overlay with live SSIM + movement
            _updateCalOverlayLive(data.interval);
        }

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

    // --- Calibration Functions ---

    let _currentCal = null;  // cached calibration for overlay

    async function loadCalibrationOverlay(cameraId) {
        const overlay = document.getElementById('cal-overlay');
        const content = document.getElementById('cal-overlay-content');
        try {
            const resp = await fetch(`/api/camera/${cameraId}/calibration`);
            if (resp.ok) {
                _currentCal = await resp.json();
                _renderCalOverlay(_currentCal, null);
                overlay.classList.remove('hidden');
            } else {
                _currentCal = null;
                content.innerHTML = '<div class="cal-title">Not Calibrated</div><div class="cal-row">Stream ~5 min, then Calibrate</div>';
                overlay.classList.remove('hidden');
            }
        } catch {
            overlay.classList.add('hidden');
        }
    }

    function _renderCalOverlay(cal, live) {
        const content = document.getElementById('cal-overlay-content');
        const ssimLine = (live && live.ssim_score != null)
            ? `<div class="cal-row"><span>SSIM:</span> <span class="cal-value">${live.ssim_score.toFixed(3)}</span></div>` : '';
        const movedLine = (live && live.camera_moved)
            ? `<div class="cal-row"><span class="cal-moved">CAMERA MOVED</span></div>` : '';

        if (cal) {
            content.innerHTML = `
                <div class="cal-title">Calibration</div>
                ${movedLine}
                <div class="cal-row"><span>Azimuth:</span> <span class="cal-value">${cal.azimuth_offset_deg.toFixed(1)}&deg;</span></div>
                <div class="cal-row"><span>Confidence:</span> <span class="cal-value">${(cal.confidence * 100).toFixed(0)}%</span></div>
                <div class="cal-row"><span>Vehicles:</span> <span class="cal-value">${cal.n_vehicles}</span></div>
                <div class="cal-row"><span>Road:</span> <span class="cal-value">${cal.primary_road}</span></div>
                <div class="cal-row"><span>Flow axis:</span> <span class="cal-value">${cal.pixel_flow_axis_deg.toFixed(1)}&deg;</span></div>
                <div class="cal-row"><span>Road bearing:</span> <span class="cal-value">${cal.road_bearing_deg.toFixed(1)}&deg;</span></div>
                ${ssimLine}
            `;
        } else {
            content.innerHTML = `
                <div class="cal-title">Not Calibrated</div>
                ${movedLine}${ssimLine}
            `;
        }
    }

    function _updateCalOverlayLive(interval) {
        if (!interval) return;
        const overlay = document.getElementById('cal-overlay');
        overlay.classList.remove('hidden');
        _renderCalOverlay(_currentCal, interval);
    }

    async function calibrateCamera() {
        if (!currentCameraId) return;
        const btn = document.getElementById('calibrate-btn');
        const status = document.getElementById('roi-status');
        btn.disabled = true;
        btn.textContent = 'Calibrating...';
        status.textContent = 'Projecting road geometry...';

        try {
            const resp = await fetch(`/api/camera/${currentCameraId}/calibrate`, {
                method: 'POST',
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Calibration failed');
            }
            const data = await resp.json();
            currentROIs = data;
            CameraPanel.setROIs(currentROIs);
            updateROIList();
            status.textContent = `Projected ${data.rois.length} road ROI(s)`;
            // Reload overlay with new calibration
            await loadCalibrationOverlay(currentCameraId);
        } catch (err) {
            console.error('Calibration failed:', err);
            status.textContent = `Error: ${err.message}`;
        }
        btn.disabled = false;
        btn.textContent = 'Calibrate';
    }

    function updateROIList() {
        const container = document.getElementById('roi-list');
        container.innerHTML = '';

        if (!currentROIs || !currentROIs.rois || currentROIs.rois.length === 0) {
            container.innerHTML = '<span style="font-size:0.7rem;color:#666">No ROIs — click Calibrate after streaming</span>';
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
            `;
            container.appendChild(item);
        });
    }
})();
