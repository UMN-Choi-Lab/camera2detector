/**
 * Chart.js time-series for count, occupancy, and speed comparison.
 * Includes per-road breakdown and historical ClickHouse overlay.
 */

const Charts = {
    countChart: null,
    occupancyChart: null,
    speedChart: null,
    maxPoints: 20,  // ~10 min at 30s intervals
    roadColorMap: {},
    _roadDatasetIndices: {},  // track which datasets are per-road

    init() {
        const defaultOpts = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 300 },
            scales: {
                x: {
                    ticks: {
                        color: '#888',
                        font: { size: 9 },
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 8,
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: '#888', font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
            },
            plugins: {
                legend: {
                    labels: { color: '#e0e0e0', font: { size: 11 } },
                },
            },
        };

        this.countChart = new Chart(document.getElementById('count-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CV Count',
                        data: [],
                        borderColor: '#4ecca3',
                        backgroundColor: 'rgba(78, 204, 163, 0.1)',
                        tension: 0.3,
                        pointRadius: 3,
                    },
                    {
                        label: 'Det Volume',
                        data: [],
                        borderColor: '#e94560',
                        backgroundColor: 'rgba(233, 69, 96, 0.1)',
                        tension: 0.3,
                        pointRadius: 3,
                    },
                ],
            },
            options: {
                ...defaultOpts,
                plugins: {
                    ...defaultOpts.plugins,
                    title: { display: true, text: 'Count Comparison', color: '#e0e0e0' },
                },
            },
        });

        this.occupancyChart = new Chart(document.getElementById('occupancy-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CV Occupancy %',
                        data: [],
                        borderColor: '#4ecca3',
                        backgroundColor: 'rgba(78, 204, 163, 0.1)',
                        tension: 0.3,
                        pointRadius: 3,
                    },
                    {
                        label: 'Det Occupancy %',
                        data: [],
                        borderColor: '#e94560',
                        backgroundColor: 'rgba(233, 69, 96, 0.1)',
                        tension: 0.3,
                        pointRadius: 3,
                    },
                ],
            },
            options: {
                ...defaultOpts,
                plugins: {
                    ...defaultOpts.plugins,
                    title: { display: true, text: 'Occupancy Comparison', color: '#e0e0e0' },
                },
            },
        });

        this.speedChart = new Chart(document.getElementById('speed-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CV Speed (mph)',
                        data: [],
                        borderColor: '#4ecca3',
                        backgroundColor: 'rgba(78, 204, 163, 0.1)',
                        tension: 0.3,
                        pointRadius: 3,
                    },
                    {
                        label: 'MnDOT Speed (mph)',
                        data: [],
                        borderColor: '#e94560',
                        backgroundColor: 'rgba(233, 69, 96, 0.1)',
                        tension: 0.3,
                        pointRadius: 3,
                    },
                    {
                        label: 'ClearGuide Speed (mph)',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.3,
                        pointRadius: 3,
                        borderDash: [6, 3],
                    },
                ],
            },
            options: {
                ...defaultOpts,
                plugins: {
                    ...defaultOpts.plugins,
                    title: { display: true, text: 'Speed Comparison', color: '#e0e0e0' },
                },
            },
        });
    },

    setRoadColorMap(colorMap) {
        this.roadColorMap = colorMap || {};
    },

    reset() {
        // Remove any extra road/history datasets (keep first 2: CV + Det)
        [this.countChart, this.occupancyChart].forEach(chart => {
            chart.data.labels = [];
            chart.data.datasets = chart.data.datasets.slice(0, 2);
            chart.data.datasets.forEach(ds => { ds.data = []; });
            chart.update('none');
        });
        // Speed chart: keep first 3 datasets (CV + MnDOT + ClearGuide)
        this.speedChart.data.labels = [];
        this.speedChart.data.datasets = this.speedChart.data.datasets.slice(0, 3);
        this.speedChart.data.datasets.forEach(ds => { ds.data = []; });
        this.speedChart.update('none');

        this._roadDatasetIndices = {};
    },

    addDataPoint(timestamp, cvCount, cvOccupancy, detVolume, detOccupancy) {
        const label = Utils.formatTime(timestamp);

        // Count chart
        this.countChart.data.labels.push(label);
        this.countChart.data.datasets[0].data.push(cvCount);
        this.countChart.data.datasets[1].data.push(detVolume);
        if (this.countChart.data.labels.length > this.maxPoints) {
            this.countChart.data.labels.shift();
            this.countChart.data.datasets.forEach(ds => ds.data.shift());
        }
        this.countChart.update();

        // Occupancy chart
        this.occupancyChart.data.labels.push(label);
        this.occupancyChart.data.datasets[0].data.push(cvOccupancy);
        this.occupancyChart.data.datasets[1].data.push(detOccupancy);
        if (this.occupancyChart.data.labels.length > this.maxPoints) {
            this.occupancyChart.data.labels.shift();
            this.occupancyChart.data.datasets.forEach(ds => ds.data.shift());
        }
        this.occupancyChart.update();
    },

    /**
     * Add speed data point to speed chart.
     * @param {string} timestamp
     * @param {number|null} detSpeed - MnDOT detector speed
     * @param {number|null} cvSpeed - CV-estimated speed
     */
    addSpeedDataPoint(timestamp, detSpeed, cvSpeed) {
        const label = Utils.formatTime(timestamp);

        this.speedChart.data.labels.push(label);
        // Dataset 0: CV Speed
        this.speedChart.data.datasets[0].data.push(cvSpeed != null ? Math.round(cvSpeed) : null);
        // Dataset 1: MnDOT Speed
        this.speedChart.data.datasets[1].data.push(detSpeed);
        // Dataset 2: ClearGuide — push null to keep aligned (overlay fills separately)
        if (this.speedChart.data.datasets[2].data.length < this.speedChart.data.labels.length) {
            this.speedChart.data.datasets[2].data.push(null);
        }

        if (this.speedChart.data.labels.length > this.maxPoints) {
            this.speedChart.data.labels.shift();
            this.speedChart.data.datasets.forEach(ds => ds.data.shift());
        }
        this.speedChart.update();
    },

    /**
     * Set ClearGuide speed overlay data.
     * @param {Array} dataPoints - [{ts, speed}]
     */
    setClearGuideOverlay(dataPoints) {
        if (!dataPoints || dataPoints.length === 0) return;

        const cgIdx = 2; // ClearGuide is dataset index 2

        // If speed chart has no labels yet, use ClearGuide timestamps
        if (this.speedChart.data.labels.length === 0) {
            this.speedChart.data.labels = dataPoints.map(p => {
                const d = new Date(typeof p.ts === 'number' ? p.ts * 1000 : p.ts);
                return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            });
            this.speedChart.data.datasets[0].data = new Array(dataPoints.length).fill(null);
            this.speedChart.data.datasets[1].data = new Array(dataPoints.length).fill(null);
            this.speedChart.data.datasets[cgIdx].data = dataPoints.map(p => p.speed);
        } else {
            const cgDs = this.speedChart.data.datasets[cgIdx];
            cgDs.data = new Array(this.speedChart.data.labels.length).fill(null);

            for (const pt of dataPoints) {
                const ptTime = new Date(typeof pt.ts === 'number' ? pt.ts * 1000 : pt.ts);
                const ptLabel = ptTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                const idx = this.speedChart.data.labels.indexOf(ptLabel);
                if (idx >= 0) {
                    cgDs.data[idx] = pt.speed;
                }
            }

            // Fill forward for ClearGuide (5min granularity covers multiple 30s points)
            let lastVal = null;
            for (let i = 0; i < cgDs.data.length; i++) {
                if (cgDs.data[i] != null) {
                    lastVal = cgDs.data[i];
                } else if (lastVal != null) {
                    cgDs.data[i] = lastVal;
                }
            }
        }
        this.speedChart.update();
    },

    /**
     * Add per-road count data to the count chart.
     * @param {Array} roadCounts - from CVResult.road_counts
     */
    addRoadDataPoint(roadCounts) {
        if (!roadCounts || roadCounts.length === 0) return;

        roadCounts.forEach(rc => {
            const key = `${rc.road_name} ${rc.direction}`;
            const color = this.roadColorMap[rc.road_name] || '#888';

            // Find or create dataset for this road
            if (!(key in this._roadDatasetIndices)) {
                const idx = this.countChart.data.datasets.length;
                this.countChart.data.datasets.push({
                    label: key,
                    data: new Array(this.countChart.data.labels.length - 1).fill(null),
                    borderColor: color,
                    backgroundColor: 'transparent',
                    borderDash: [4, 2],
                    tension: 0.3,
                    pointRadius: 2,
                    borderWidth: 1.5,
                });
                this._roadDatasetIndices[key] = idx;
            }

            const dsIdx = this._roadDatasetIndices[key];
            this.countChart.data.datasets[dsIdx].data.push(rc.vehicle_count);

            // Trim to maxPoints
            if (this.countChart.data.datasets[dsIdx].data.length > this.maxPoints) {
                this.countChart.data.datasets[dsIdx].data.shift();
            }
        });

        // Fill null for roads not in this update
        for (const [key, dsIdx] of Object.entries(this._roadDatasetIndices)) {
            const ds = this.countChart.data.datasets[dsIdx];
            if (ds.data.length < this.countChart.data.labels.length) {
                ds.data.push(null);
            }
        }

        this.countChart.update();
    },

    /**
     * Add a historical overlay as dashed line from ClickHouse data.
     * @param {string} label - e.g. "Yesterday v30"
     * @param {Array} dataPoints - [{ts, value}]
     * @param {string} chartType - "count" or "occupancy"
     */
    addHistoricalOverlay(label, dataPoints, chartType) {
        const chart = chartType === 'count' ? this.countChart : this.occupancyChart;

        // Check if dataset already exists
        const existing = chart.data.datasets.findIndex(ds => ds.label === label);
        if (existing >= 0) {
            chart.data.datasets.splice(existing, 1);
        }

        chart.data.datasets.push({
            label: label,
            data: dataPoints.map(p => p.value),
            borderColor: '#fbbf24',
            backgroundColor: 'transparent',
            borderDash: [8, 4],
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2,
        });

        // Use historical timestamps as labels if chart is empty
        if (chart.data.labels.length === 0) {
            chart.data.labels = dataPoints.map(p => {
                const d = new Date(p.ts);
                return d.toLocaleTimeString();
            });
        }

        chart.update();
    },
};
