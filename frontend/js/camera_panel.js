/**
 * Camera panel: image display + CV bounding box overlay + road-colored boxes.
 */

const CameraPanel = {
    imageEl: null,
    videoEl: null,
    canvasEl: null,
    ctx: null,
    roadColorMap: {},  // route_label -> color
    currentROIs: null, // CameraROIs object
    _hls: null,        // HLS.js instance
    _trackingSSE: null, // EventSource for tracking data
    _latestTrackingData: null,
    _prevTrackingData: null,
    _trackingTimestamp: 0,    // performance.now() when latest data arrived
    _trackVelocities: {},     // track_id -> {vx, vy} pixels per second
    _rafId: null,      // requestAnimationFrame ID

    init() {
        this.imageEl = document.getElementById('camera-image');
        this.videoEl = document.getElementById('camera-video');
        this.canvasEl = document.getElementById('cv-overlay');
        this.ctx = this.canvasEl.getContext('2d');
    },

    setROIs(rois) {
        this.currentROIs = rois;
    },

    show(cameraId, cameraName, detectors) {
        document.getElementById('detail-placeholder').classList.add('hidden');
        document.getElementById('detail-content').classList.remove('hidden');

        document.getElementById('camera-name').textContent = cameraName;
        document.getElementById('camera-id-badge').textContent = cameraId;

        // Video will be started by startHlsStream() from app.js

        // Populate detectors table
        this._populateDetectors(detectors);

        // Reset stat cards
        document.getElementById('cv-count').textContent = '--';
        document.getElementById('cv-occupancy').textContent = '--';
        document.getElementById('cv-speed').textContent = '--';
        document.getElementById('det-count').textContent = '--';
        document.getElementById('det-occupancy').textContent = '--';
        document.getElementById('det-speed').textContent = '--';

        // Hide road breakdown and stream info until data arrives
        document.getElementById('road-breakdown').classList.add('hidden');
        document.getElementById('stream-info').classList.add('hidden');
    },

    setRoadColorMap(colorMap) {
        this.roadColorMap = colorMap || {};
    },

    /**
     * Start hybrid HLS stream: native <video> + canvas overlay from tracking SSE.
     */
    startHlsStream(cameraId) {
        this.stopHlsStream();

        // Show video, hide img
        const cameraView = document.getElementById('camera-view');
        cameraView.classList.add('hls-mode');
        this.videoEl.style.display = 'block';
        this.imageEl.style.display = 'none';

        // Load HLS via hls.js
        const hlsUrl = `https://video.dot.state.mn.us/public/${cameraId}.stream/playlist.m3u8`;

        if (Hls.isSupported()) {
            const hls = new Hls({
                liveSyncDurationCount: 1,
                liveMaxLatencyDurationCount: 3,
                lowLatencyMode: true,
                maxBufferLength: 4,
                maxMaxBufferLength: 8,
            });
            hls.loadSource(hlsUrl);
            hls.attachMedia(this.videoEl);
            hls.on(Hls.Events.MANIFEST_PARSED, () => {
                this.videoEl.play().catch(() => {});
            });
            hls.on(Hls.Events.ERROR, (_, data) => {
                if (data.fatal) {
                    console.warn('HLS fatal error, attempting recovery:', data.type);
                    if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
                        hls.startLoad();
                    } else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) {
                        hls.recoverMediaError();
                    }
                }
            });
            this._hls = hls;
        } else if (this.videoEl.canPlayType('application/vnd.apple.mpegurl')) {
            // Native HLS (Safari)
            this.videoEl.src = hlsUrl;
            this.videoEl.play().catch(() => {});
        }

        // Connect tracking SSE for annotation data
        this._trackingSSE = new EventSource(`/api/tracking/${cameraId}`);
        this._trackingSSE.addEventListener('tracking', (e) => {
            try {
                const newData = JSON.parse(e.data);
                const now = performance.now();
                const dt = (now - this._trackingTimestamp) / 1000; // seconds since last update

                // Compute per-track velocity from consecutive frames
                if (this._latestTrackingData && dt > 0.01 && dt < 1.0) {
                    const prevMap = {};
                    (this._latestTrackingData.detections || []).forEach(d => {
                        if (d.track_id != null) prevMap[d.track_id] = d;
                    });
                    const vels = {};
                    (newData.detections || []).forEach(d => {
                        if (d.track_id != null && prevMap[d.track_id]) {
                            const prev = prevMap[d.track_id];
                            vels[d.track_id] = {
                                vx: (d.cx - prev.cx) / dt,
                                vy: (d.cy - prev.cy) / dt,
                            };
                        }
                    });
                    this._trackVelocities = vels;
                }

                this._latestTrackingData = newData;
                this._trackingTimestamp = now;
            } catch (err) {
                console.debug('Failed to parse tracking data:', err);
            }
        });

        // Render loop: interpolate box positions between SSE updates
        const renderLoop = () => {
            if (this._latestTrackingData) {
                const elapsed = (performance.now() - this._trackingTimestamp) / 1000;
                // Clamp interpolation to avoid overshoot when SSE is delayed
                const t = Math.min(elapsed, 0.2);
                this._drawTrackingOverlayInterp(this._latestTrackingData, t);
            }
            this._rafId = requestAnimationFrame(renderLoop);
        };
        this._rafId = requestAnimationFrame(renderLoop);
    },

    /**
     * Stop HLS stream and tracking overlay.
     */
    stopHlsStream() {
        if (this._rafId) {
            cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
        if (this._trackingSSE) {
            this._trackingSSE.close();
            this._trackingSSE = null;
        }
        this._latestTrackingData = null;
        if (this._hls) {
            this._hls.destroy();
            this._hls = null;
        }
        this.videoEl.src = '';
        this.videoEl.style.display = 'none';
        this.imageEl.style.display = 'block';
        const cameraView = document.getElementById('camera-view');
        cameraView.classList.remove('hls-mode');
        this.clearOverlay();
    },

    /**
     * Start MJPEG live stream (annotations are baked into the JPEG server-side).
     * No canvas overlay needed — avoids duplicate bounding boxes.
     */
    startMjpegStream(cameraId) {
        this.stopMjpegStream();
        this._mjpegCameraId = cameraId;
        this.imageEl.src = `/api/stream/${cameraId}`;
    },

    /**
     * Stop MJPEG stream.
     */
    stopMjpegStream() {
        this._mjpegCameraId = null;
        this.imageEl.src = '';
        this.clearOverlay();
    },

    /**
     * Draw tracking data with velocity interpolation.
     * Extrapolates box/trail positions by velocity * elapsed seconds since last SSE update.
     */
    _drawTrackingOverlayInterp(data, elapsed) {
        const vels = this._trackVelocities;

        // Build interpolated detections: shift cx/cy/x1/y1/x2/y2 by velocity * elapsed
        const interpDetections = (data.detections || []).map(d => {
            const vel = (d.track_id != null) ? vels[d.track_id] : null;
            if (!vel) return d;
            const dx = vel.vx * elapsed;
            const dy = vel.vy * elapsed;
            return {
                ...d,
                x1: d.x1 + dx, y1: d.y1 + dy,
                x2: d.x2 + dx, y2: d.y2 + dy,
                cx: d.cx + dx, cy: d.cy + dy,
            };
        });

        // Build interpolated trails: shift last point of each trail
        const interpTrails = {};
        const trails = data.trails || {};
        for (const [tid, points] of Object.entries(trails)) {
            if (points.length === 0) { interpTrails[tid] = points; continue; }
            const vel = vels[tid];
            if (!vel) { interpTrails[tid] = points; continue; }
            const dx = vel.vx * elapsed;
            const dy = vel.vy * elapsed;
            // Shift last point only (the head)
            const shifted = [...points];
            const last = shifted[shifted.length - 1];
            shifted[shifted.length - 1] = [last[0] + dx, last[1] + dy];
            interpTrails[tid] = shifted;
        }

        this._drawTrackingOverlay({
            ...data,
            detections: interpDetections,
            trails: interpTrails,
        });
    },

    /**
     * Draw tracking data (boxes + trails + ROIs) on canvas overlay.
     */
    _drawTrackingOverlay(data) {
        // Use video or image intrinsic size, fall back to 720x480
        let vw = 720, vh = 480;
        if (this.videoEl && this.videoEl.videoWidth > 0) {
            vw = this.videoEl.videoWidth;
            vh = this.videoEl.videoHeight;
        } else if (this.imageEl && this.imageEl.naturalWidth > 0) {
            vw = this.imageEl.naturalWidth;
            vh = this.imageEl.naturalHeight;
        }
        if (this.canvasEl.width !== vw) this.canvasEl.width = vw;
        if (this.canvasEl.height !== vh) this.canvasEl.height = vh;

        const ctx = this.ctx;
        ctx.clearRect(0, 0, vw, vh);

        // Draw ROI polygons
        (data.rois || []).forEach(roi => {
            const poly = roi.polygon;
            if (!poly || poly.length < 3) return;
            const color = roi.color || '#a855f7';

            // Fill
            ctx.save();
            ctx.globalAlpha = 0.12;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(poly[0][0], poly[0][1]);
            for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
            ctx.closePath();
            ctx.fill();
            ctx.restore();

            // Border
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(poly[0][0], poly[0][1]);
            for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
            ctx.closePath();
            ctx.stroke();

            // Label
            let cx = 0, cy = 0;
            poly.forEach(p => { cx += p[0]; cy += p[1]; });
            cx /= poly.length; cy /= poly.length;
            const label = `${roi.road_name || ''} ${roi.direction || ''}`;
            ctx.font = 'bold 12px monospace';
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.8;
            ctx.fillRect(cx - tw/2 - 3, cy - 8, tw + 6, 16);
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = '#fff';
            ctx.fillText(label, cx - tw/2, cy + 4);
        });

        // Draw trajectory trails with fading
        const trails = data.trails || {};
        const detMap = {};
        (data.detections || []).forEach(d => {
            if (d.track_id != null) detMap[d.track_id] = d;
        });

        for (const [tid, points] of Object.entries(trails)) {
            if (points.length < 2) continue;
            const det = detMap[tid];
            const color = (det && det.color) || '#4ecca3';
            const n = points.length;

            for (let i = 1; i < n; i++) {
                const fade = 0.15 + 0.85 * (i / (n - 1));
                ctx.globalAlpha = fade;
                ctx.strokeStyle = color;
                ctx.lineWidth = Math.max(1, Math.floor(1 + 2 * (i / (n - 1))));
                ctx.beginPath();
                ctx.moveTo(points[i-1][0], points[i-1][1]);
                ctx.lineTo(points[i][0], points[i][1]);
                ctx.stroke();
            }
            // Head dot
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(points[n-1][0], points[n-1][1], 4, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.globalAlpha = 1.0;

        // Draw bounding boxes
        (data.detections || []).forEach(det => {
            const color = det.color || '#4ecca3';
            const x = det.x1, y = det.y1;
            const w = det.x2 - det.x1, h = det.y2 - det.y1;

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);

            // Label
            let label = det.label;
            if (det.track_id != null) label += ` #${det.track_id}`;
            ctx.font = '11px monospace';
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = color;
            ctx.fillRect(x, y - 15, tw + 6, 15);
            ctx.fillStyle = '#000';
            ctx.fillText(label, x + 3, y - 3);
        });
    },

    clearOverlay() {
        this.ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);
    },

    drawROIs(rois, imageWidth, imageHeight) {
        if (!rois || rois.length === 0) return;

        const scaleX = this.canvasEl.width / (imageWidth || this.imageEl.naturalWidth);
        const scaleY = this.canvasEl.height / (imageHeight || this.imageEl.naturalHeight);

        rois.forEach(roi => {
            const poly = roi.polygon;
            if (!poly || poly.length < 3) return;

            const color = roi.color || '#a855f7';

            // Semi-transparent fill
            this.ctx.save();
            this.ctx.globalAlpha = 0.15;
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.moveTo(poly[0][0] * scaleX, poly[0][1] * scaleY);
            for (let i = 1; i < poly.length; i++) {
                this.ctx.lineTo(poly[i][0] * scaleX, poly[i][1] * scaleY);
            }
            this.ctx.closePath();
            this.ctx.fill();
            this.ctx.restore();

            // Colored border
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(poly[0][0] * scaleX, poly[0][1] * scaleY);
            for (let i = 1; i < poly.length; i++) {
                this.ctx.lineTo(poly[i][0] * scaleX, poly[i][1] * scaleY);
            }
            this.ctx.closePath();
            this.ctx.stroke();

            // Label at centroid
            let cx = 0, cy = 0;
            poly.forEach(p => { cx += p[0]; cy += p[1]; });
            cx = (cx / poly.length) * scaleX;
            cy = (cy / poly.length) * scaleY;

            const label = `${roi.road_name} ${roi.direction}`;
            this.ctx.font = 'bold 12px monospace';
            const textW = this.ctx.measureText(label).width;
            this.ctx.fillStyle = color;
            this.ctx.globalAlpha = 0.8;
            this.ctx.fillRect(cx - textW / 2 - 3, cy - 8, textW + 6, 16);
            this.ctx.globalAlpha = 1.0;
            this.ctx.fillStyle = '#fff';
            this.ctx.fillText(label, cx - textW / 2, cy + 4);
        });
    },

    drawBoxes(boxes) {
        this.clearOverlay();

        // Draw ROIs first (underneath bounding boxes)
        if (this.currentROIs && this.currentROIs.rois) {
            this.drawROIs(this.currentROIs.rois, this.currentROIs.image_width, this.currentROIs.image_height);
        }

        const scaleX = this.canvasEl.width / this.imageEl.naturalWidth;
        const scaleY = this.canvasEl.height / this.imageEl.naturalHeight;

        boxes.forEach(box => {
            const x = box.x1 * scaleX;
            const y = box.y1 * scaleY;
            const w = (box.x2 - box.x1) * scaleX;
            const h = (box.y2 - box.y1) * scaleY;

            // Color by road assignment, fallback to default green
            const roadColor = box.road_name && this.roadColorMap[box.road_name]
                ? this.roadColorMap[box.road_name]
                : '#4ecca3';

            this.ctx.strokeStyle = roadColor;
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(x, y, w, h);

            // Direction arrow for tracked vehicles
            if (box.road_direction && box.track_id != null) {
                this._drawDirectionArrow(x + w / 2, y + h / 2, box.road_direction, roadColor);
            }

            // Label
            let label = `${box.label} ${(box.confidence * 100).toFixed(0)}%`;
            if (box.road_name) {
                label += ` ${box.road_name}`;
                if (box.road_direction) label += ` ${box.road_direction}`;
            }
            this.ctx.font = '11px monospace';
            this.ctx.fillStyle = roadColor;
            const textW = this.ctx.measureText(label).width;
            this.ctx.fillRect(x, y - 15, textW + 6, 15);
            this.ctx.fillStyle = '#000';
            this.ctx.fillText(label, x + 3, y - 3);
        });
    },

    _drawDirectionArrow(cx, cy, direction, color) {
        const size = 10;
        const angles = {
            'NB': -Math.PI / 2, 'SB': Math.PI / 2,
            'EB': 0, 'WB': Math.PI,
            'NEB': -Math.PI / 4, 'SEB': Math.PI / 4,
            'SWB': 3 * Math.PI / 4, 'NWB': -3 * Math.PI / 4,
        };
        const angle = angles[direction];
        if (angle === undefined) return;

        this.ctx.save();
        this.ctx.translate(cx, cy);
        this.ctx.rotate(angle);
        this.ctx.fillStyle = color;
        this.ctx.globalAlpha = 0.8;
        this.ctx.beginPath();
        this.ctx.moveTo(size, 0);
        this.ctx.lineTo(-size / 2, -size / 2);
        this.ctx.lineTo(-size / 2, size / 2);
        this.ctx.closePath();
        this.ctx.fill();
        this.ctx.globalAlpha = 1.0;
        this.ctx.restore();
    },

    updateStats(cvResult, stations, interval) {
        document.getElementById('cv-count').textContent = cvResult.vehicle_count;
        document.getElementById('cv-occupancy').textContent = Utils.fmt(cvResult.occupancy) + '%';

        // CV speed from interval data
        if (interval && interval.detectors) {
            const speeds = interval.detectors.map(d => d.speed).filter(s => s != null);
            const avgSpeed = speeds.length > 0
                ? speeds.reduce((a, b) => a + b, 0) / speeds.length
                : null;
            document.getElementById('cv-speed').textContent = avgSpeed != null
                ? Utils.fmt(avgSpeed, 0) + ' mph'
                : '--';
        } else {
            document.getElementById('cv-speed').textContent = '--';
        }

        // Station-level detector totals (sum volume, avg occ/speed)
        const stationVols = (stations || []).map(s => s.volume).filter(v => v != null);
        const detVol = stationVols.length > 0 ? stationVols.reduce((a, b) => a + b, 0) : null;
        const detOcc = Utils.avg((stations || []).map(s => s.occupancy));
        const detSpd = Utils.avg((stations || []).map(s => s.speed));

        document.getElementById('det-count').textContent = detVol != null ? Utils.fmt(detVol, 0) : '--';
        document.getElementById('det-occupancy').textContent = detOcc != null ? Utils.fmt(detOcc) + '%' : '--';
        document.getElementById('det-speed').textContent = detSpd != null ? Utils.fmt(detSpd, 0) + ' mph' : '--';
    },

    _countdownTimer: null,
    _countdownStart: null,
    _countdownDuration: 30,  // seconds

    updateStreamInfo(interval) {
        const el = document.getElementById('stream-info');
        if (!interval) {
            el.classList.add('hidden');
            return;
        }
        el.classList.remove('hidden');
        document.getElementById('stream-fps').textContent = `${interval.fps_actual} fps`;
        document.getElementById('stream-frames').textContent = `${interval.frame_count} frames`;

        // Reset countdown on each update
        this._startCountdown();
    },

    _startCountdown() {
        // Clear existing timer
        if (this._countdownTimer) {
            clearInterval(this._countdownTimer);
        }

        this._countdownStart = Date.now();
        const duration = this._countdownDuration;
        const bar = document.getElementById('stream-progress-bar');
        const label = document.getElementById('stream-countdown');

        // Reset bar immediately (no transition for the reset)
        bar.style.transition = 'none';
        bar.style.width = '0%';

        // Force reflow then re-enable transition
        bar.offsetHeight;
        bar.style.transition = 'width 1s linear';

        const tick = () => {
            const elapsed = (Date.now() - this._countdownStart) / 1000;
            const remaining = Math.max(0, duration - elapsed);
            const pct = Math.min(100, (elapsed / duration) * 100);

            bar.style.width = `${pct}%`;
            label.textContent = `Next in ${Math.ceil(remaining)}s`;

            if (remaining <= 0) {
                label.textContent = 'Updating...';
                clearInterval(this._countdownTimer);
                this._countdownTimer = null;
            }
        };

        tick();
        this._countdownTimer = setInterval(tick, 1000);
    },

    /**
     * Update road breakdown badges showing per-road vehicle counts.
     * @param {Array} roadCounts - from CVResult.road_counts
     */
    /**
     * Update road breakdown badges with CV and detector per-direction comparison.
     * @param {Array} roadCounts - from IntervalResult.detectors or CVResult.road_counts
     * @param {Array} stations - StationAggregate array from SSE event
     */
    updateRoadBreakdown(roadCounts, stations) {
        const section = document.getElementById('road-breakdown');
        const container = document.getElementById('road-badges');

        if (!roadCounts || roadCounts.length === 0) {
            section.classList.add('hidden');
            return;
        }

        // Build station lookup by direction
        const stationByDir = {};
        (stations || []).forEach(s => { stationByDir[s.direction] = s; });

        section.classList.remove('hidden');
        container.innerHTML = '';

        roadCounts.forEach(rc => {
            const roadName = rc.road_name;
            const dir = rc.direction;
            const vol = rc.volume != null ? rc.volume : rc.vehicle_count;
            const color = this.roadColorMap[roadName] || '#4ecca3';
            const badge = document.createElement('div');
            badge.className = 'road-badge';
            badge.style.borderLeftColor = color;

            const types = Object.entries(rc.by_type || {})
                .map(([t, c]) => `${c} ${t}${c > 1 ? 's' : ''}`)
                .join(', ');

            const speedStr = rc.speed != null ? ` | spd: ${Utils.fmt(rc.speed, 0)} mph` : '';

            // Match station by direction for side-by-side comparison
            const station = stationByDir[dir];
            let detLine = '';
            if (station) {
                const parts = [];
                if (station.volume != null) parts.push(`vol: ${Utils.fmt(station.volume, 0)}`);
                if (station.occupancy != null) parts.push(`occ: ${Utils.fmt(station.occupancy)}%`);
                if (station.speed != null) parts.push(`spd: ${Utils.fmt(station.speed, 0)} mph`);
                detLine = `<div class="road-badge-detail" style="color:#e94560">Det ${station.station_label}: ${parts.join(' | ')}</div>`;
            }

            badge.innerHTML = `
                <div class="road-badge-header">
                    <span class="road-badge-name" style="color:${color}">${roadName} ${dir}</span>
                    <span class="road-badge-count">${vol}</span>
                </div>
                <div class="road-badge-detail">${types || 'no vehicles'} | occ: ${Utils.fmt(rc.occupancy)}%${speedStr}</div>
                ${detLine}
            `;
            container.appendChild(badge);
        });
    },

    /**
     * Show/update road legend based on color map.
     */
    updateRoadLegend(roadColorMap) {
        const legend = document.getElementById('road-legend');
        const items = document.getElementById('road-legend-items');

        if (!roadColorMap || Object.keys(roadColorMap).length === 0) {
            legend.classList.add('hidden');
            return;
        }

        legend.classList.remove('hidden');
        items.innerHTML = '';

        Object.entries(roadColorMap).forEach(([label, color]) => {
            const item = document.createElement('span');
            item.className = 'road-legend-item';
            item.innerHTML = `<span class="road-legend-swatch" style="background:${color}"></span>${label}`;
            items.appendChild(item);
        });
    },

    _populateDetectors(detectors) {
        const tbody = document.querySelector('#detectors-table tbody');
        tbody.innerHTML = '';

        if (!detectors || detectors.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#888">No matched detectors</td></tr>';
            return;
        }

        detectors.forEach(det => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${det.id}</td>
                <td>${det.label}</td>
                <td>${det.distance_m}</td>
                <td>${det.lane || '-'}</td>
                <td>${det.corridor || '-'}</td>
            `;
            tbody.appendChild(tr);
        });
    },
};
