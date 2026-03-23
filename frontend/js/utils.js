/**
 * Utility helpers: time index calculation, formatting.
 */

const Utils = {
    /**
     * Get the current 30-second index into Mayfly data arrays.
     */
    current30sIndex() {
        const now = new Date();
        return Math.floor((now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds()) / 30);
    },

    /**
     * Format a timestamp string to HH:MM:SS.
     */
    formatTime(isoString) {
        const d = new Date(isoString);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    },

    /**
     * Format a number to fixed decimal places.
     */
    fmt(val, decimals = 1) {
        if (val == null || val === '--') return '--';
        return Number(val).toFixed(decimals);
    },

    /**
     * Normalize road name: "I 35W" -> "I-35W", "US 12" -> "US-12"
     */
    normalizeRoadName(name) {
        if (!name) return name;
        return name.replace(/^(I|US|MN|TH)\s+/, '$1-');
    },

    /**
     * Average an array of numbers, ignoring nulls.
     */
    avg(arr) {
        const valid = arr.filter(v => v != null && v >= 0);
        if (valid.length === 0) return null;
        return valid.reduce((a, b) => a + b, 0) / valid.length;
    },
};
