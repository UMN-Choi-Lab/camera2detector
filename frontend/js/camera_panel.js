/**
 * Camera panel: image display + CV bounding box overlay + road-colored boxes.
 */

const CameraPanel = {
    imageEl: null,
    canvasEl: null,
    ctx: null,
    roadColorMap: {},  // route_label -> color
    currentROIs: null, // CameraROIs object

    init() {
        this.imageEl = document.getElementById('camera-image');
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

        // Load camera image
        this.loadImage(cameraId);

        // Populate detectors table
        this._populateDetectors(detectors);

        // Reset stat cards
        document.getElementById('cv-count').textContent = '--';
        document.getElementById('cv-occupancy').textContent = '--';
        document.getElementById('det-count').textContent = '--';
        document.getElementById('det-occupancy').textContent = '--';

        // Hide road breakdown until data arrives
        document.getElementById('road-breakdown').classList.add('hidden');
    },

    setRoadColorMap(colorMap) {
        this.roadColorMap = colorMap || {};
    },

    loadImage(cameraId) {
        this.imageEl.src = `/api/camera/${cameraId}/image?t=${Date.now()}`;
        this.imageEl.onload = () => {
            this.canvasEl.width = this.imageEl.naturalWidth;
            this.canvasEl.height = this.imageEl.naturalHeight;
            this.clearOverlay();
        };
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

    updateStats(cvResult, detectorSamples) {
        document.getElementById('cv-count').textContent = cvResult.vehicle_count;
        document.getElementById('cv-occupancy').textContent = Utils.fmt(cvResult.occupancy) + '%';

        // Average detector values
        const detVol = Utils.avg(detectorSamples.map(d => d.volume));
        const detOcc = Utils.avg(detectorSamples.map(d => d.occupancy));

        document.getElementById('det-count').textContent = detVol != null ? Utils.fmt(detVol, 0) : '--';
        document.getElementById('det-occupancy').textContent = detOcc != null ? Utils.fmt(detOcc) + '%' : '--';
    },

    /**
     * Update road breakdown badges showing per-road vehicle counts.
     * @param {Array} roadCounts - from CVResult.road_counts
     */
    updateRoadBreakdown(roadCounts) {
        const section = document.getElementById('road-breakdown');
        const container = document.getElementById('road-badges');

        if (!roadCounts || roadCounts.length === 0) {
            section.classList.add('hidden');
            return;
        }

        section.classList.remove('hidden');
        container.innerHTML = '';

        roadCounts.forEach(rc => {
            const color = this.roadColorMap[rc.road_name] || '#4ecca3';
            const badge = document.createElement('div');
            badge.className = 'road-badge';
            badge.style.borderLeftColor = color;

            // Build type breakdown string
            const types = Object.entries(rc.by_type || {})
                .map(([t, c]) => `${c} ${t}${c > 1 ? 's' : ''}`)
                .join(', ');

            badge.innerHTML = `
                <div class="road-badge-header">
                    <span class="road-badge-name" style="color:${color}">${rc.road_name} ${rc.direction}</span>
                    <span class="road-badge-count">${rc.vehicle_count}</span>
                </div>
                <div class="road-badge-detail">${types || 'no vehicles'} | occ: ${Utils.fmt(rc.occupancy)}%</div>
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
