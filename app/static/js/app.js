/**
 * Thai Coin Counter & Detector
 * Mobile-First Client Application
 */

class CoinCounterApp {
    constructor() {
        // DOM Elements
        this.video = document.getElementById("webcam");
        this.canvas = document.getElementById("overlay-canvas");
        this.ctx = this.canvas.getContext("2d");
        this.hiddenCanvas = document.getElementById("hidden-canvas");
        this.hiddenCtx = this.hiddenCanvas.getContext("2d");

        // UI Displays
        this.totalBahtEl = document.getElementById("total-baht");
        this.count10El = document.getElementById("count-10");
        this.count5El = document.getElementById("count-5");
        this.count1El = document.getElementById("count-1");
        this.totalCoinsEl = document.getElementById("total-coins");
        this.fpsCounterEl = document.getElementById("fps-counter");
        this.cameraLabelEl = document.getElementById("camera-label");
        this.connectionBadge = document.getElementById("connection-badge");
        this.statusText = document.getElementById("status-text");
        this.guideReticle = document.getElementById("guide-reticle");
        this.detectionStatusPill = document.getElementById("detection-status-pill");
        this.engineBadge = document.getElementById("engine-badge");
        this.toastEl = document.getElementById("toast");

        // Engine & Roboflow Controls
        this.btnEngineLocal = document.getElementById("btn-engine-local");
        this.btnEngineYolo = document.getElementById("btn-engine-yolo");
        this.btnEngineRoboflow = document.getElementById("btn-engine-roboflow");
        this.inputRoboflowKey = document.getElementById("input-roboflow-key");
        this.btnSaveRoboflowKey = document.getElementById("btn-save-roboflow-key");
        this.roboflowKeyStatus = document.getElementById("roboflow-key-status");

        // Action Buttons
        this.btnSwitchCamera = document.getElementById("btn-switch-camera");
        this.btnTorch = document.getElementById("btn-torch");
        this.btnSnapshot = document.getElementById("btn-snapshot");
        this.btnToggleGuide = document.getElementById("btn-toggle-guide");
        this.btnToggleSound = document.getElementById("btn-toggle-sound");
        this.soundIcon = document.getElementById("sound-icon");
        this.btnToggleSettings = document.getElementById("btn-toggle-settings");
        this.btnCloseSettings = document.getElementById("btn-close-settings");
        this.btnResetSettings = document.getElementById("btn-reset-settings");
        this.btnShowQr = document.getElementById("btn-show-qr");
        this.btnCloseMobileModal = document.getElementById("btn-close-mobile-modal");
        this.btnCopyUrl = document.getElementById("btn-copy-url");
        this.toggleGuideOnly = document.getElementById("toggle-guide-only");

        // Settings Drawers & Modals
        this.settingsDrawer = document.getElementById("settings-drawer");
        this.mobileInfoModal = document.getElementById("mobile-info-modal");
        this.mobileUrlText = document.getElementById("mobile-url-text");

        // Settings Sliders
        this.sliderSensitivity = document.getElementById("slider-sensitivity");
        this.sliderMinRadius = document.getElementById("slider-min-radius");
        this.sliderMaxRadius = document.getElementById("slider-max-radius");
        this.sliderFps = document.getElementById("slider-fps");
        this.valSensitivity = document.getElementById("val-sensitivity");
        this.valMinRadius = document.getElementById("val-min-radius");
        this.valMaxRadius = document.getElementById("val-max-radius");
        this.valFps = document.getElementById("val-fps");

        // State
        this.stream = null;
        this.videoTrack = null;
        this.currentFacingMode = "environment"; // default to rear camera on mobile
        this.isTorchOn = false;
        this.hasTorch = false;
        this.isSoundEnabled = true;
        this.isStreaming = false;
        this.isWaitingForResponse = false;
        this.ws = null;
        this.wsReconnectTimer = null;
        this.lastFrameTime = performance.now();
        this.fps = 0;
        this.previousCoinsCount = 0;

        // Settings Config
        this.config = {
            sensitivity: 30,
            minRadius: 20,
            maxRadius: 160,
            targetFps: 10,
            guideOnly: false,
            engine: "local",
            hasRoboflowKey: false,
        };

        // Audio Context for Coin Chime
        this.audioCtx = null;

        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.fetchEngineConfig();
        await this.fetchServerInfo();
        await this.startCamera();
        this.connectWebSocket();
        this.startStreamLoop();
    }

    /* -------------------------------------------------------------
       Camera Management & Switching
       ------------------------------------------------------------- */
    async startCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach((track) => track.stop());
        }

        this.showToast("กำลังเปิดกล้อง...");

        const constraints = {
            audio: false,
            video: {
                facingMode: { ideal: this.currentFacingMode },
                width: { ideal: 1280 },
                height: { ideal: 720 },
            },
        };

        try {
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;

            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    this.resizeCanvas();
                    resolve();
                };
            });

            this.videoTrack = this.stream.getVideoTracks()[0];
            const settings = this.videoTrack.getSettings();
            const label = settings.facingMode === "user" ? "กล้องหน้า" : "กล้องหลัง";
            this.cameraLabelEl.innerText = `กล้อง: ${label}`;

            // Check Torch (Flashlight) Support
            await this.checkTorchCapability();
            this.showToast(`เปิด${label}สำเร็จ`);
        } catch (err) {
            console.error("Camera access error:", err);
            this.showToast("ไม่สามารถเปิดกล้องได้: " + err.message);
            this.cameraLabelEl.innerText = "กล้อง: ข้อผิดพลาด";
        }
    }

    async switchCamera() {
        this.currentFacingMode = this.currentFacingMode === "environment" ? "user" : "environment";
        this.isTorchOn = false;
        this.btnTorch.classList.remove("active");
        await this.startCamera();
    }

    async checkTorchCapability() {
        if (!this.videoTrack) return;
        const capabilities = this.videoTrack.getCapabilities ? this.videoTrack.getCapabilities() : {};
        this.hasTorch = !!capabilities.torch;
        if (!this.hasTorch) {
            this.btnTorch.style.opacity = "0.4";
            this.btnTorch.title = "อุปกรณ์นี้ไม่รองรับไฟแฟลช";
        } else {
            this.btnTorch.style.opacity = "1";
            this.btnTorch.title = "เปิด/ปิดไฟแฟลช";
        }
    }

    async toggleTorch() {
        if (!this.hasTorch || !this.videoTrack) {
            this.showToast("อุปกรณ์นี้ไม่รองรับไฟฉาย");
            return;
        }

        try {
            this.isTorchOn = !this.isTorchOn;
            await this.videoTrack.applyConstraints({
                advanced: [{ torch: this.isTorchOn }],
            });
            this.btnTorch.classList.toggle("active", this.isTorchOn);
            this.showToast(this.isTorchOn ? "เปิดไฟฉายแล้ว" : "ปิดไฟฉายแล้ว");
        } catch (err) {
            console.error("Torch error:", err);
            this.showToast("ไม่สามารถควบคุมไฟฉายได้");
        }
    }

    /* -------------------------------------------------------------
       WebSocket Real-Time Pipeline
       ------------------------------------------------------------- */
    connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.updateConnectionStatus("connecting", "กำลังเชื่อมต่อ...");

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log("WebSocket connected.");
            this.updateConnectionStatus("connected", "พร้อมตรวจจับ");
            if (this.wsReconnectTimer) {
                clearTimeout(this.wsReconnectTimer);
                this.wsReconnectTimer = null;
            }
        };

        this.ws.onmessage = (event) => {
            this.isWaitingForResponse = false;
            try {
                const data = JSON.parse(event.data);
                this.handleDetectionResult(data);
            } catch (err) {
                console.error("Message parse error:", err);
            }
        };

        this.ws.onclose = () => {
            this.updateConnectionStatus("disconnected", "การเชื่อมต่อขาดหาย");
            this.isWaitingForResponse = false;
            // Retry connection
            if (!this.wsReconnectTimer) {
                this.wsReconnectTimer = setTimeout(() => this.connectWebSocket(), 2000);
            }
        };

        this.ws.onerror = (err) => {
            console.error("WebSocket error:", err);
            this.ws.close();
        };
    }

    updateConnectionStatus(status, text) {
        this.connectionBadge.className = `badge-status badge-${status}`;
        this.statusText.innerText = text;
    }

    startStreamLoop() {
        const loop = () => {
            const now = performance.now();
            const interval = 1000 / this.config.targetFps;

            if (
                now - this.lastFrameTime >= interval &&
                !this.isWaitingForResponse &&
                this.ws &&
                this.ws.readyState === WebSocket.OPEN &&
                this.video.readyState === this.video.HAVE_ENOUGH_DATA
            ) {
                this.sendFrame();
                this.lastFrameTime = now;
            }

            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }

    sendFrame() {
        const videoW = this.video.videoWidth;
        const videoH = this.video.videoHeight;
        if (!videoW || !videoH) return;

        // Scale down frame for transmission efficiency (e.g. max dimension 640px)
        const maxDim = 640;
        let scale = Math.min(1.0, maxDim / Math.max(videoW, videoH));
        const targetW = Math.round(videoW * scale);
        const targetH = Math.round(videoH * scale);

        this.hiddenCanvas.width = targetW;
        this.hiddenCanvas.height = targetH;

        this.hiddenCtx.drawImage(this.video, 0, 0, targetW, targetH);

        const dataUrl = this.hiddenCanvas.toDataURL("image/jpeg", 0.65);

        const payload = {
            image: dataUrl,
            engine: this.config.engine,
            sensitivity: this.config.sensitivity,
            min_radius: Math.round(this.config.minRadius * scale),
            max_radius: Math.round(this.config.maxRadius * scale),
        };

        if (this.config.guideOnly && this.guideReticle) {
            const rect = this.video.getBoundingClientRect();
            const rRect = this.guideReticle.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                const rx = ((rRect.left + rRect.width / 2) - rect.left) * (targetW / rect.width);
                const ry = ((rRect.top + rRect.height / 2) - rect.top) * (targetH / rect.height);
                const rr = (rRect.width / 2) * (targetW / rect.width);
                payload.guide_center = [Math.round(rx), Math.round(ry)];
                payload.guide_radius = Math.round(rr);
            }
        }

        this.isWaitingForResponse = true;
        this.ws.send(JSON.stringify(payload));
    }

    /* -------------------------------------------------------------
       Result Rendering & Canvas Drawing
       ------------------------------------------------------------- */
    handleDetectionResult(result) {
        // Calculate FPS
        const now = performance.now();
        const delta = (now - (this._lastResultTime || now)) / 1000;
        this._lastResultTime = now;
        if (delta > 0) {
            this.fps = Math.round(1 / delta);
            this.fpsCounterEl.innerText = `FPS: ${this.fps}`;
        }

        const summary = result.summary || {
            count_10: 0,
            count_5: 0,
            count_1: 0,
            total_coins: 0,
            total_baht: 0,
        };

        const totalCoins = summary.total_coins || 0;

        // When NO coins are detected
        if (totalCoins === 0) {
            if (this.detectionStatusPill) {
                this.detectionStatusPill.className = "hud-pill hud-status-idle";
                this.detectionStatusPill.innerText = "⚪ รอวางเหรียญ";
            }
            this.totalBahtEl.innerText = "0";
            this.count10El.innerText = "0";
            this.count5El.innerText = "0";
            this.count1El.innerText = "0";
            this.totalCoinsEl.innerText = "0";
            this.previousCoinsCount = 0;

            // Clear canvas completely so no false rings are drawn
            this.resizeCanvas();
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            return;
        }

        // When coins ARE detected
        if (this.detectionStatusPill) {
            this.detectionStatusPill.className = "hud-pill hud-status-detected";
            this.detectionStatusPill.innerText = `🪙 ตรวจพบ ${totalCoins} เหรียญ (${summary.total_baht}฿)`;
        }

        // Play sound if coin count increased
        if (totalCoins > this.previousCoinsCount) {
            this.playCoinChime();
        }
        this.previousCoinsCount = totalCoins;

        // Update UI counters
        this.totalBahtEl.innerText = summary.total_baht;
        this.count10El.innerText = summary.count_10;
        this.count5El.innerText = summary.count_5;
        this.count1El.innerText = summary.count_1;
        this.totalCoinsEl.innerText = totalCoins;

        // Draw overlay
        this.renderOverlay(result.coins || [], result.frame_width, result.frame_height);
    }

    resizeCanvas() {
        const rect = this.video.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    renderOverlay(coins, frameW, frameH) {
        this.resizeCanvas();
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        this.ctx.clearRect(0, 0, cw, ch);

        if (!coins || coins.length === 0 || !frameW || !frameH) return;

        // Scale factors to map frame coordinates to canvas coordinates
        const scaleX = cw / frameW;
        const scaleY = ch / frameH;

        for (const coin of coins) {
            const cx = coin.x * scaleX;
            const cy = coin.y * scaleY;
            const cr = coin.radius * Math.max(scaleX, scaleY);

            this.ctx.save();

            // 1. Outer Glowing Circle
            this.ctx.beginPath();
            this.ctx.arc(cx, cy, cr, 0, 2 * Math.PI);
            this.ctx.lineWidth = 3.5;
            this.ctx.strokeStyle = coin.color;
            this.ctx.shadowColor = coin.color;
            this.ctx.shadowBlur = 12;
            this.ctx.stroke();

            // 2. Center crosshair dot
            this.ctx.beginPath();
            this.ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
            this.ctx.fillStyle = coin.color;
            this.ctx.fill();

            // 3. Label Badge on top of coin
            const confText = coin.confidence ? ` (${Math.round(coin.confidence * 100)}%)` : "";
            const label = `${coin.label || `${coin.type}฿`}${confText}`;
            this.ctx.font = "bold 12px Prompt, sans-serif";
            const textWidth = this.ctx.measureText(label).width;
            const badgeW = textWidth + 14;
            const badgeH = 22;
            const badgeX = cx - badgeW / 2;
            const badgeY = cy - cr - badgeH - 6;

            // Badge Background
            this.ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
            this.ctx.shadowBlur = 6;
            this.roundRect(this.ctx, badgeX, badgeY, badgeW, badgeH, 6, true, false);

            // Badge Border
            this.ctx.lineWidth = 1.5;
            this.ctx.strokeStyle = coin.color;
            this.roundRect(this.ctx, badgeX, badgeY, badgeW, badgeH, 6, false, true);

            // Badge Text
            this.ctx.fillStyle = "#ffffff";
            this.ctx.textAlign = "center";
            this.ctx.textBaseline = "middle";
            this.ctx.fillText(label, cx, badgeY + badgeH / 2);

            this.ctx.restore();
        }
    }

    roundRect(ctx, x, y, width, height, radius, fill, stroke) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        if (fill) ctx.fill();
        if (stroke) ctx.stroke();
    }

    /* -------------------------------------------------------------
       Snapshot / Freeze Mode
       ------------------------------------------------------------- */
    async takeSnapshot() {
        this.showToast("กำลังจับภาพวิเคราะห์ความละเอียดสูง...");
        const videoW = this.video.videoWidth;
        const videoH = this.video.videoHeight;
        if (!videoW || !videoH) return;

        this.hiddenCanvas.width = videoW;
        this.hiddenCanvas.height = videoH;
        this.hiddenCtx.drawImage(this.video, 0, 0, videoW, videoH);

        const dataUrl = this.hiddenCanvas.toDataURL("image/jpeg", 0.90);

        try {
            const formData = new FormData();
            formData.append("image_base64", dataUrl);
            formData.append("sensitivity", this.config.sensitivity);
            formData.append("min_radius", this.config.minRadius);
            formData.append("max_radius", this.config.maxRadius);

            const res = await fetch("/api/detect", {
                method: "POST",
                body: formData,
            });
            const result = await res.json();
            this.handleDetectionResult(result);
            this.showToast(`ตรวจพบ ${result.summary.total_coins} เหรียญ (${result.summary.total_baht} บาท)`);
        } catch (err) {
            console.error("Snapshot error:", err);
            this.showToast("เกิดข้อผิดพลาดในการวิเคราะห์");
        }
    }

    /* -------------------------------------------------------------
       Sound & Haptic Feedback
       ------------------------------------------------------------- */
    playCoinChime() {
        if (!this.isSoundEnabled) return;

        // Vibrate if mobile
        if ("vibrate" in navigator) {
            navigator.vibrate([40]);
        }

        try {
            if (!this.audioCtx) {
                this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (this.audioCtx.state === "suspended") {
                this.audioCtx.resume();
            }

            const osc = this.audioCtx.createOscillator();
            const gain = this.audioCtx.createGain();

            osc.type = "sine";
            osc.frequency.setValueAtTime(987.77, this.audioCtx.currentTime); // B5
            osc.frequency.exponentialRampToValueAtTime(1318.51, this.audioCtx.currentTime + 0.1); // E6

            gain.gain.setValueAtTime(0.15, this.audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + 0.25);

            osc.connect(gain);
            gain.connect(this.audioCtx.destination);

            osc.start();
            osc.stop(this.audioCtx.currentTime + 0.25);
        } catch (e) {
            // Audio policy prevent
        }
    }

    /* -------------------------------------------------------------
       Engine Configuration & Roboflow AI
       ------------------------------------------------------------- */
    async fetchEngineConfig() {
        try {
            const res = await fetch("/api/config");
            const data = await res.json();
            this.config.engine = data.engine || "local";
            this.config.hasRoboflowKey = data.has_roboflow_key || false;
            this.updateEngineUI();
        } catch (err) {
            console.error("Config fetch error:", err);
        }
    }

    updateEngineUI() {
        const engine = this.config.engine || "local";
        if (this.btnEngineLocal) {
            this.btnEngineLocal.classList.toggle("active", engine === "local");
        }
        if (this.btnEngineYolo) {
            this.btnEngineYolo.classList.toggle("active", engine === "yolo");
        }
        if (this.btnEngineRoboflow) {
            this.btnEngineRoboflow.classList.toggle("active", engine === "roboflow");
        }

        if (this.engineBadge) {
            this.engineBadge.classList.remove("roboflow-active", "yolo-active");
            if (engine === "roboflow") {
                this.engineBadge.innerText = "🌐 Roboflow";
                this.engineBadge.classList.add("roboflow-active");
            } else if (engine === "yolo") {
                this.engineBadge.innerText = "🧠 YOLO Offline";
                this.engineBadge.classList.add("yolo-active");
            } else {
                this.engineBadge.innerText = "⚡ Local";
            }
        }

        if (this.roboflowKeyStatus) {
            if (this.config.hasRoboflowKey) {
                this.roboflowKeyStatus.className = "status-pill status-saved";
                this.roboflowKeyStatus.innerText = "✓ ตั้งค่าคีย์แล้ว";
            } else {
                this.roboflowKeyStatus.className = "status-pill status-missing";
                this.roboflowKeyStatus.innerText = "ยังไม่ใส่คีย์";
            }
        }
    }

    async switchEngine(engine) {
        if (engine === "roboflow" && !this.config.hasRoboflowKey) {
            this.showToast("⚠️ กรุณาใส่ Roboflow API Key ก่อนเปิดใช้งาน Cloud AI ครับ");
            this.settingsDrawer.classList.add("open");
            this.inputRoboflowKey.focus();
            return;
        }

        try {
            const res = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ engine: engine }),
            });
            const data = await res.json();
            if (data.status === "ok") {
                this.config.engine = engine;
                this.updateEngineUI();
                const msg = engine === "roboflow"
                    ? "🌐 สลับไปใช้ Roboflow Cloud AI แล้ว"
                    : engine === "yolo"
                    ? "🧠 สลับไปใช้ YOLO Offline AI แล้ว"
                    : "⚡ สลับไปใช้ Local OpenCV แล้ว";
                this.showToast(msg);
            }
        } catch (err) {
            console.error("Switch engine error:", err);
            this.showToast("เกิดข้อผิดพลาดในการสลับเอนจิน");
        }
    }

    async saveRoboflowKey() {
        const key = this.inputRoboflowKey.value.trim();
        if (!key) {
            this.showToast("กรุณากรอก API Key ก่อนกดบันทึก");
            return;
        }

        try {
            const res = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    roboflow_api_key: key,
                    engine: "roboflow",
                }),
            });
            const data = await res.json();
            if (data.status === "ok") {
                this.config.hasRoboflowKey = true;
                this.config.engine = "roboflow";
                this.inputRoboflowKey.value = "";
                this.updateEngineUI();
                this.showToast("🎉 บันทึก API Key และเปิดใช้งาน Roboflow AI สำเร็จ!");
            }
        } catch (err) {
            console.error("Save key error:", err);
            this.showToast("บันทึก API Key ล้มเหลว");
        }
    }

    /* -------------------------------------------------------------
       Server Info & Mobile Connection
       ------------------------------------------------------------- */
    async fetchServerInfo() {
        try {
            const res = await fetch("/api/info");
            const data = await res.json();
            const port = window.location.port ? `:${window.location.port}` : "";
            const mobileUrl = `${window.location.protocol}//${data.local_ip}${port}`;
            this.mobileUrlText.innerText = mobileUrl;
        } catch (e) {
            this.mobileUrlText.innerText = window.location.href;
        }
    }

    /* -------------------------------------------------------------
       Event Listeners & UI Binding
       ------------------------------------------------------------- */
    setupEventListeners() {
        // Engine Buttons
        if (this.btnEngineLocal) {
            this.btnEngineLocal.addEventListener("click", () => this.switchEngine("local"));
        }
        if (this.btnEngineYolo) {
            this.btnEngineYolo.addEventListener("click", () => this.switchEngine("yolo"));
        }
        if (this.btnEngineRoboflow) {
            this.btnEngineRoboflow.addEventListener("click", () => this.switchEngine("roboflow"));
        }
        if (this.engineBadge) {
            this.engineBadge.addEventListener("click", () => {
                this.settingsDrawer.classList.add("open");
            });
        }
        if (this.btnSaveRoboflowKey) {
            this.btnSaveRoboflowKey.addEventListener("click", () => this.saveRoboflowKey());
        }

        // Switch Camera
        this.btnSwitchCamera.addEventListener("click", () => this.switchCamera());

        // Flashlight Torch
        this.btnTorch.addEventListener("click", () => this.toggleTorch());

        // Snapshot
        this.btnSnapshot.addEventListener("click", () => this.takeSnapshot());

        // Toggle Guide Reticle (Target Zone)
        this.btnToggleGuide.addEventListener("click", () => {
            const isVisible = this.guideReticle.classList.toggle("visible");
            this.btnToggleGuide.classList.toggle("active", isVisible);
            this.config.guideOnly = isVisible;
            if (this.toggleGuideOnly) {
                this.toggleGuideOnly.checked = isVisible;
            }
            this.showToast(isVisible ? "🎯 ล็อกเป้า: ตรวจจับเฉพาะในกรอบเล็ง" : "🌐 ตรวจจับทั้งหน้าจอ");
        });

        // Toggle Sound
        this.btnToggleSound.addEventListener("click", () => {
            this.isSoundEnabled = !this.isSoundEnabled;
            this.soundIcon.innerText = this.isSoundEnabled ? "🔔" : "🔕";
            this.btnToggleSound.classList.toggle("active", this.isSoundEnabled);
            this.showToast(this.isSoundEnabled ? "เปิดเสียงเตือน" : "ปิดเสียงเตือน");
        });

        // Settings Drawer
        this.btnToggleSettings.addEventListener("click", () => {
            this.settingsDrawer.classList.toggle("open");
        });
        this.btnCloseSettings.addEventListener("click", () => {
            this.settingsDrawer.classList.remove("open");
        });

        // Mobile Info Modal
        this.btnShowQr.addEventListener("click", () => {
            this.mobileInfoModal.classList.add("open");
        });
        this.btnCloseMobileModal.addEventListener("click", () => {
            this.mobileInfoModal.classList.remove("open");
        });

        // Copy Mobile URL
        this.btnCopyUrl.addEventListener("click", () => {
            navigator.clipboard.writeText(this.mobileUrlText.innerText);
            this.showToast("คัดลอก URL เรียบร้อยแล้ว");
        });

        // Toggle Guide Only scanning
        if (this.toggleGuideOnly) {
            this.toggleGuideOnly.addEventListener("change", (e) => {
                this.config.guideOnly = e.target.checked;
                this.guideReticle.classList.toggle("visible", this.config.guideOnly);
                this.btnToggleGuide.classList.toggle("active", this.config.guideOnly);
                this.showToast(this.config.guideOnly ? "🎯 ล็อกเป้า: ตรวจจับเฉพาะในกรอบเล็ง" : "🌐 ตรวจจับทั้งหน้าจอ");
            });
        }

        // Sliders
        this.sliderSensitivity.addEventListener("input", (e) => {
            this.config.sensitivity = parseInt(e.target.value);
            this.valSensitivity.innerText = this.config.sensitivity;
        });

        this.sliderMinRadius.addEventListener("input", (e) => {
            this.config.minRadius = parseInt(e.target.value);
            this.valMinRadius.innerText = this.config.minRadius;
        });

        this.sliderMaxRadius.addEventListener("input", (e) => {
            this.config.maxRadius = parseInt(e.target.value);
            this.valMaxRadius.innerText = this.config.maxRadius;
        });

        this.sliderFps.addEventListener("input", (e) => {
            this.config.targetFps = parseInt(e.target.value);
            this.valFps.innerText = this.config.targetFps;
        });

        this.btnResetSettings.addEventListener("click", () => {
            this.config = {
                sensitivity: 30,
                minRadius: 20,
                maxRadius: 160,
                targetFps: 10,
            };
            this.sliderSensitivity.value = 30;
            this.sliderMinRadius.value = 20;
            this.sliderMaxRadius.value = 160;
            this.sliderFps.value = 10;
            this.valSensitivity.innerText = "30";
            this.valMinRadius.innerText = "20";
            this.valMaxRadius.innerText = "160";
            this.valFps.innerText = "10";
            this.showToast("รีเซ็ตการตั้งค่าเรียบร้อย");
        });

        // Window Resize
        window.addEventListener("resize", () => this.resizeCanvas());
    }

    showToast(message) {
        this.toastEl.innerText = message;
        this.toastEl.classList.add("show");
        if (this._toastTimer) clearTimeout(this._toastTimer);
        this._toastTimer = setTimeout(() => {
            this.toastEl.classList.remove("show");
        }, 2200);
    }
}

// Start application when DOM is ready
window.addEventListener("DOMContentLoaded", () => {
    window.coinApp = new CoinCounterApp();
});
