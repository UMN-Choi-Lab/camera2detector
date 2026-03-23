/**
 * EventSource connection manager for SSE streaming.
 */

const SSEClient = {
    source: null,
    currentCameraId: null,

    connect(cameraId, onUpdate) {
        this.disconnect();
        this.currentCameraId = cameraId;

        this.source = new EventSource(`/api/sse/${cameraId}`);

        this.source.addEventListener('update', (e) => {
            try {
                const data = JSON.parse(e.data);
                onUpdate(data);
            } catch (err) {
                console.error('Failed to parse SSE event:', err);
            }
        });

        this.source.addEventListener('error', (e) => {
            try {
                const data = JSON.parse(e.data);
                console.warn('SSE error event:', data.error);
            } catch {
                // Connection error, EventSource will auto-reconnect
            }
        });

        this.source.onerror = () => {
            console.warn('SSE connection error, will auto-reconnect');
        };
    },

    disconnect() {
        if (this.source) {
            this.source.close();
            this.source = null;
            this.currentCameraId = null;
        }
    },
};
