/**
 * Leaflet map with clustered camera markers + road centerlines.
 */

const CameraMap = {
    map: null,
    clusterGroup: null,
    cameraMarkers: {},       // camera_id -> L.marker
    detectorMarkers: [],     // currently visible detector markers
    roadLayers: [],          // currently visible road polylines
    activeMarkerId: null,

    // Road color palette
    ROAD_COLORS: [
        '#3b82f6', // blue
        '#f97316', // orange
        '#a855f7', // purple
        '#14b8a6', // teal
        '#ef4444', // red
        '#eab308', // yellow
        '#ec4899', // pink
        '#22c55e', // green
    ],

    init() {
        this.map = L.map('map', {
            center: [44.96, -93.27],  // Twin Cities metro
            zoom: 11,
            zoomControl: true,
        });

        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
            maxZoom: 19,
        }).addTo(this.map);

        this.clusterGroup = L.markerClusterGroup({
            maxClusterRadius: 40,
            spiderfyOnMaxZoom: true,
            showCoverageOnHover: false,
            iconCreateFunction(cluster) {
                const count = cluster.getChildCount();
                let size = 'small';
                if (count > 50) size = 'large';
                else if (count > 10) size = 'medium';
                return L.divIcon({
                    html: `<div><span>${count}</span></div>`,
                    className: `marker-cluster marker-cluster-${size}`,
                    iconSize: L.point(40, 40),
                });
            },
        });

        this.map.addLayer(this.clusterGroup);
    },

    loadCameras(cameras, onCameraClick) {
        cameras.forEach(cam => {
            const icon = L.divIcon({
                className: 'camera-marker',
                iconSize: [12, 12],
            });

            const marker = L.marker([cam.lat, cam.lon], { icon })
                .bindPopup(`<b>${cam.name}</b><br>${cam.id}`);

            marker.on('click', () => onCameraClick(cam));

            this.clusterGroup.addLayer(marker);
            this.cameraMarkers[cam.id] = marker;
        });
    },

    selectCamera(cameraId, detectors) {
        // Reset previous active marker
        if (this.activeMarkerId && this.cameraMarkers[this.activeMarkerId]) {
            this.cameraMarkers[this.activeMarkerId].setIcon(
                L.divIcon({ className: 'camera-marker', iconSize: [12, 12] })
            );
        }

        // Set new active
        this.activeMarkerId = cameraId;
        const marker = this.cameraMarkers[cameraId];
        if (marker) {
            marker.setIcon(
                L.divIcon({ className: 'camera-marker active', iconSize: [16, 16] })
            );
            this.map.setView(marker.getLatLng(), 15, { animate: true });
        }

        // Clear old detector markers
        this.detectorMarkers.forEach(m => this.map.removeLayer(m));
        this.detectorMarkers = [];

        // Add detector markers
        if (detectors) {
            detectors.forEach(det => {
                const detIcon = L.divIcon({
                    className: 'detector-marker',
                    iconSize: [10, 10],
                });
                const detMarker = L.marker([det.lat, det.lon], { icon: detIcon })
                    .bindPopup(`<b>${det.label}</b><br>${det.id}<br>${det.distance_m}m`)
                    .addTo(this.map);
                this.detectorMarkers.push(detMarker);
            });
        }
    },

    /**
     * Draw road centerlines on the map for nearby roads.
     * @param {Array} roads - array of road objects with geometry_coords
     * @returns {Object} roadColorMap - route_label -> color
     */
    drawRoads(roads) {
        // Clear old road layers
        this.clearRoads();

        const roadColorMap = {};
        let colorIdx = 0;

        roads.forEach(road => {
            const rawLabel = road.route_label || road.route_name;
            if (!rawLabel) return;
            const label = Utils.normalizeRoadName(rawLabel);

            // Assign color per route_label
            if (!roadColorMap[label]) {
                roadColorMap[label] = this.ROAD_COLORS[colorIdx % this.ROAD_COLORS.length];
                colorIdx++;
            }
            const color = roadColorMap[label];

            // geometry_coords is [[lon, lat], ...] — Leaflet wants [[lat, lon], ...]
            if (road.geometry_coords && road.geometry_coords.length >= 2) {
                const latLngs = road.geometry_coords.map(c => [c[1], c[0]]);
                const polyline = L.polyline(latLngs, {
                    color: color,
                    weight: 4,
                    opacity: 0.7,
                    dashArray: null,
                }).addTo(this.map);

                polyline.bindPopup(
                    `<b>${label}</b><br>${road.cardinal || ''} ${road.direction || ''}<br>` +
                    `${road.distance_m}m from camera`
                );

                this.roadLayers.push(polyline);
            }
        });

        return roadColorMap;
    },

    clearRoads() {
        this.roadLayers.forEach(l => this.map.removeLayer(l));
        this.roadLayers = [];
    },
};
