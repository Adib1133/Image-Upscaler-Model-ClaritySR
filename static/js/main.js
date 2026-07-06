document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const uploadPreview = document.getElementById('upload-preview');
    const previewName = document.getElementById('preview-filename');
    const previewSize = document.getElementById('preview-filesize');
    const previewThumb = document.getElementById('preview-img-thumbnail');
    const removeFileBtn = document.getElementById('remove-file-btn');
    const dropzoneContent = dropzone.querySelector('.dropzone-content');
    
    const upscaleForm = document.getElementById('upscale-form');
    const submitBtn = document.getElementById('upscale-submit-btn');
    const stopBtn = document.getElementById('upscale-stop-btn');

    const modelSelect = document.getElementById('model-select');
    const modelDesc = document.getElementById('model-desc');
    const scaleGroup = document.getElementById('scale-group');
    const scaleSelect = document.getElementById('scale-select');
    
    // Viewport views
    const placeholderView = document.getElementById('placeholder-view');
    const loaderView = document.getElementById('loader-view');
    const sliderView = document.getElementById('slider-view');
    const sideView = document.getElementById('side-view');
    
    // Loader details
    const loaderTitle = document.getElementById('loader-title');
    const loaderMsg = document.getElementById('loader-msg');
    const loaderPercent = document.getElementById('loader-percent');
    const loaderProgressBar = document.getElementById('loader-progress-bar');
    const loaderTimer = document.getElementById('loader-timer');
    
    // Slider View components
    const sliderImgOrig = document.getElementById('slider-img-orig');
    const sliderImgUpscaled = document.getElementById('slider-img-upscaled');
    const sliderImgUpscaledContainer = document.getElementById('slider-img-upscaled-container');
    const sliderHandle = document.getElementById('slider-handle');
    const sliderContainer = sliderHandle.parentElement;
    
    // Side-by-Side Zoom components
    const sideViewportOrig = document.getElementById('viewport-orig');
    const sideViewportUpscaled = document.getElementById('viewport-upscaled');
    const zoomImgOrig = document.getElementById('zoom-img-orig');
    const zoomImgUpscaled = document.getElementById('zoom-img-upscaled');
    const zoomSlider = document.getElementById('zoom-slider');
    const zoomVal = document.getElementById('zoom-val');
    
    // Toolbar controls
    const workspaceToolbar = document.getElementById('workspace-toolbar');
    const btnViewSlider = document.getElementById('btn-view-slider');
    const btnViewSide = document.getElementById('btn-view-side');
    const btnDownloadActive = document.getElementById('btn-download-active');
    const btnClearResult = document.getElementById('btn-clear-result');

    // Viewfinder action buttons
    const viewfinderActions = document.getElementById('viewfinder-actions');
    const vfUpscaleBtn = document.getElementById('vf-upscale-btn');
    const vfStopBtn = document.getElementById('vf-stop-btn');
    
    // Console logs & stats
    const consoleLogs = document.getElementById('console-logs');
    const btnClearLogs = document.getElementById('btn-clear-logs');
    const cpuFill = document.getElementById('cpu-fill');
    const cpuVal = document.getElementById('cpu-val');
    const ramFill = document.getElementById('ram-fill');
    const ramVal = document.getElementById('ram-val');
    const systemStatusMsg = document.getElementById('system-status-msg');
    const headerDevice = document.getElementById('header-device');
    const historyContainer = document.getElementById('history-items-container');

    // App state
    let activeFile = null;
    let sseSource = null;
    let timerInterval = null;
    let upscaledFilename = null;
    let activeViewMode = 'slider'; // 'slider' or 'side'
    const MAX_FILE_SIZE = 30 * 1024 * 1024;

    // Error Modal Elements & Functions
    const errorModal = document.getElementById('error-modal');
    const modalErrorTitle = document.getElementById('modal-error-title');
    const modalErrorMessage = document.getElementById('modal-error-message');
    const btnCloseModal = document.getElementById('btn-close-modal');

    function showErrorModal(message, title = 'System Error') {
        modalErrorTitle.innerText = title;
        modalErrorMessage.innerText = message;
        errorModal.classList.add('active');
    }

    btnCloseModal.addEventListener('click', () => {
        errorModal.classList.remove('active');
    });

    errorModal.addEventListener('click', (e) => {
        if (e.target === errorModal) {
            errorModal.classList.remove('active');
        }
    });

    // ==========================================
    // MODEL DESCRIPTIONS
    // ==========================================

    const MODEL_DESCS = {
        'realesrgan-general': 'Real-ESRGAN produces sharp, photo-realistic upscaling on general photography and mixed content.',
        'realesrgan-anime': 'Optimized for anime illustrations, manga, and artwork. 6-block architecture is 4x faster.',
        'realesrgan-x2': 'Same architecture as Real-ESRGAN x4+, native 2x output - ideal for already decent images.',
        'realesrgan-v3': 'Balanced quality-to-speed model. Great all-rounder for mixed content and web images.',
        'esrgan-classic': 'Classic ESRGAN from 2018. Produces very sharp, high-texture detail with a natural look.',
        'bsrgan-x4': 'BSRGAN targets blind degradation - excellent at removing blur, noise, and JPEG compression.',
        'gfpgan-1.4': 'Generative Facial Prior GAN. Restores broken images, artifacts, and highly compressed faces with realistic textures.',
        'detail-enhance': 'Runs restoration at native resolution (1x). No final upscale - it removes artifacts and enhances micro-detail.',
    };

    modelSelect.addEventListener('change', () => {
        const key = modelSelect.value;
        modelDesc.innerText = MODEL_DESCS[key] || '';
        
        // Hide/show scale selector for 1x-only model
        if (key === 'detail-enhance') {
            scaleGroup.style.display = 'none';
            scaleSelect.value = '1';
        } else {
            scaleGroup.style.display = 'block';
            if (scaleSelect.value === '1') scaleSelect.value = '4';
        }
    });

    // ==========================================
    // UTILS & DIAGNOSTIC CONSOLE LOGS
    // ==========================================
    
    function log(message, type = 'system') {
        const line = document.createElement('div');
        line.className = `log-line ${type}-log`;
        
        const timestamp = new Date().toLocaleTimeString();
        line.innerText = `[${timestamp}] ${message}`;
        
        consoleLogs.appendChild(line);
        consoleLogs.scrollTop = consoleLogs.scrollHeight;
    }

    btnClearLogs.addEventListener('click', () => {
        consoleLogs.innerHTML = '';
        log('Console cleared.', 'system');
    });

    // ==========================================
    // SYSTEM MONITOR POLLING
    // ==========================================
    
    const gpuGaugesContainer = document.getElementById('gpu-gauges-container');

    // Track which GPU gauge elements we've already created
    const gpuFills = {};
    const gpuVals  = {};

    function ensureGpuGauge(label, id, fullName) {
        if (gpuFills[id]) return;
        const row = document.createElement('div');
        row.className = 'metric-gauge';
        row.title = fullName || label;
        row.innerHTML = `
            <span class="metric-label">${label}</span>
            <div class="gauge-bar"><div class="gauge-fill" id="gpu-fill-${id}" style="width:0%"></div></div>
            <span class="metric-val" id="gpu-val-${id}">0%</span>`;
        gpuGaugesContainer.appendChild(row);
        gpuFills[id] = document.getElementById(`gpu-fill-${id}`);
        gpuVals[id]  = document.getElementById(`gpu-val-${id}`);
    }

    function updateSystemMetrics() {
        fetch('/api/system_info')
            .then(res => res.json())
            .then(data => {
                headerDevice.innerText = data.device;
                
                cpuVal.innerText = `${data.cpu_usage}%`;
                cpuFill.style.width = `${data.cpu_usage}%`;
                cpuFill.className = 'gauge-fill';
                if (data.cpu_usage > 85) cpuFill.classList.add('critical');
                else if (data.cpu_usage > 60) cpuFill.classList.add('warning');
                
                ramVal.innerText = `${data.ram_usage}% / ${data.ram_total_gb}GB`;
                ramFill.style.width = `${data.ram_usage}%`;
                ramFill.className = 'gauge-fill';
                if (data.ram_usage > 85) ramFill.classList.add('critical');
                else if (data.ram_usage > 65) ramFill.classList.add('warning');

                // Render GPU gauges
                if (data.gpus && data.gpus.length > 0) {
                    data.gpus.forEach((gpu, i) => {
                        const id = `gpu_${i}`;
                        ensureGpuGauge(`${gpu.label} LOAD`, id, gpu.name);
                        const pct = gpu.usage_pct;
                        gpuFills[id].style.width = `${pct}%`;
                        gpuFills[id].className = 'gauge-fill';
                        if (pct > 85) gpuFills[id].classList.add('critical');
                        else if (pct > 60) gpuFills[id].classList.add('warning');
                        gpuVals[id].innerText = `${pct}%`;
                    });
                }
            })
            .catch(err => {
                console.error("Error polling system metrics:", err);
            });
    }

    updateSystemMetrics();
    setInterval(updateSystemMetrics, 3000);

    // ==========================================
    // DRAG AND DROP FILE ACTIONS
    // ==========================================

    // Trigger input browse
    dropzone.addEventListener('click', (e) => {
        // Prevent click bubbling from the file input itself or preview buttons
        if (e.target === fileInput || e.target.closest('#remove-file-btn') || e.target.closest('.upload-preview')) {
            return;
        }
        fileInput.click();
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleSelectedFile(fileInput.files[0]);
        }
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith('image/')) {
            log('Error: Selected file must be an image.', 'error');
            showErrorModal('Please select a valid image file.', 'Invalid File Type');
            return;
        }
        if (file.size > MAX_FILE_SIZE) {
            log('Error: Selected file is larger than 30 MB.', 'error');
            showErrorModal('Please select an image smaller than 30 MB.', 'File Too Large');
            return;
        }

        activeFile = file;

        previewName.innerText = file.name;
        const sizeKB = (file.size / 1024).toFixed(1);
        previewSize.innerText = `${sizeKB} KB`;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewThumb.src = e.target.result;
            dropzoneContent.style.display = 'none';
            uploadPreview.style.display = 'flex';
        };
        reader.readAsDataURL(file);

        submitBtn.disabled = false;
        vfUpscaleBtn.disabled = false;
        viewfinderActions.style.display = 'flex';
        log(`Loaded input image: ${file.name} (${sizeKB} KB)`, 'system');
    }

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUploadField();
    });

    function resetUploadField() {
        activeFile = null;
        fileInput.value = '';
        uploadPreview.style.display = 'none';
        dropzoneContent.style.display = 'block';
        submitBtn.disabled = true;
        vfUpscaleBtn.disabled = true;
        log('Removed input image.', 'system');
    }

    // ==========================================
    // STOP BUTTON
    // ==========================================

    stopBtn.addEventListener('click', () => {
        fetch('/api/stop', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                log(data.message, 'error');
                stopBtn.disabled = true;
                stopBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Stopping...';
                vfStopBtn.disabled = true;
                vfStopBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Stopping...';
            })
            .catch(err => {
                log(`Stop request failed: ${err.message}`, 'error');
            });
    });

    // ==========================================
    // SUBMIT & UPSCALING WORKER ACTIONS
    // ==========================================

    upscaleForm.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!activeFile) return;

        const formData = new FormData(upscaleForm);
        formData.set('image', activeFile, activeFile.name);
        
        // Lock UI controls
        submitBtn.disabled = true;
        stopBtn.disabled = false;
        stopBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop';
        vfUpscaleBtn.disabled = true;
        vfStopBtn.disabled = false;
        vfStopBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop';
        dropzone.style.pointerEvents = 'none';
        removeFileBtn.style.display = 'none';
        
        // Shift views to Loader
        placeholderView.style.display = 'none';
        sliderView.style.display = 'none';
        sideView.style.display = 'none';
        workspaceToolbar.style.display = 'none';
        loaderView.style.display = 'flex';
        
        // Reset loader status
        loaderTitle.innerText = "Initializing upscaler...";
        loaderMsg.innerText = "Pre-processing input image file and loading neural network configurations.";
        loaderPercent.innerText = "0%";
        loaderProgressBar.style.width = "0%";
        loaderTimer.innerText = "0.0s";
        
        // Start client-side timer
        let elapsedSeconds = 0;
        clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            elapsedSeconds += 0.1;
            loaderTimer.innerText = `${elapsedSeconds.toFixed(1)}s`;
        }, 100);

        systemStatusMsg.innerText = "Processing Image...";
        systemStatusMsg.parentElement.querySelector('.indicator').className = 'indicator orange';

        log("Sending upload request to Flask backend...", 'system');

        fetch('/api/upscale', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            upscaledFilename = data.output_filename;
            log(`Upload successful. Input: ${data.input_filename}. Launching model pipeline.`, 'system');
            
            startProgressSSE(data.input_filename, data.output_filename);
        })
        .catch(err => {
            log(`Inference failed: ${err.message}`, 'error');
            showErrorModal(err.message, 'Inference Request Failed');
            stopUpscaleUI(false);
        });
    });

    function startProgressSSE(inputName, outputName) {
        if (sseSource) {
            sseSource.close();
        }
        
        sseSource = new EventSource(`/api/progress?filename=${outputName}`);
        
        sseSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Update loader display from server-side progress
            loaderTitle.innerText = data.message;
            loaderPercent.innerText = `${data.progress}%`;
            loaderProgressBar.style.width = `${data.progress}%`;


            
            // Classify log type
            let logType = 'pytorch';
            if (data.message.includes('Downloading')) {
                logType = 'download';
                loaderMsg.innerText = "Connecting to model repository. Larger models may take up to 2 minutes.";
            } else if (data.message.includes('tile') || data.message.includes('Tile')) {
                logType = 'pytorch';
                loaderMsg.innerText = "Dividing large image into padding blocks for safe CPU memory allocation.";
            } else if (data.message.includes('successfully') || data.message.includes('finished') || data.message.includes('Completed')) {
                logType = 'success';
            } else if (data.message.includes('Error')) {
                logType = 'error';
            } else if (data.message.includes('Stopped')) {
                logType = 'error';
            } else {
                loaderMsg.innerText = "Running PyTorch tensor processing pipeline.";
            }
            
            log(data.message, logType);
            
            // Handle completion
            if (data.progress >= 100 || data.message.includes('finished') || data.message.includes('Completed') || data.message.includes('Error') || data.message.includes('Stopped')) {
                sseSource.close();
                clearInterval(timerInterval);
                
                if (data.message.includes('Error')) {
                    showErrorModal(data.message, 'Processing Error');
                    stopUpscaleUI(false);
                } else if (data.message.includes('Stopped')) {
                    log('Task was stopped by user.', 'error');
                    stopUpscaleUI(false);
                } else {
                    log(`Inference finished. Total execution time: ${loaderTimer.innerText}`, 'success');
                    loadResultImages(inputName, outputName);
                }
            }
        };

        sseSource.onerror = (e) => {
            console.error("SSE connection error: ", e);
            log("Warning: SSE progress stream disconnected. Retrying...", "error");
        };
    }



    function stopUpscaleUI(isSuccess = true) {
        submitBtn.disabled = !activeFile;
        stopBtn.disabled = true;
        stopBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop';
        vfUpscaleBtn.disabled = !activeFile;
        vfStopBtn.disabled = true;
        vfStopBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop';
        dropzone.style.pointerEvents = 'auto';
        removeFileBtn.style.display = 'flex';
        loaderView.style.display = 'none';

        systemStatusMsg.innerText = "Upscaler Ready";
        systemStatusMsg.parentElement.querySelector('.indicator').className = 'indicator green';

        if (!isSuccess) {
            placeholderView.style.display = 'flex';
            workspaceToolbar.style.display = 'none';
            viewfinderActions.style.display = 'none';
        }
    }

    // ==========================================
    // RESULT DISPLAY & INTERACTION SLIDERS
    // ==========================================

    function loadResultImages(inputName, outputName) {
        const inputUrl = `/uploads/${inputName}?t=${Date.now()}`;
        const outputUrl = `/outputs/${outputName}?t=${Date.now()}`;

        // Load images for View 1: Slider
        sliderImgOrig.src = inputUrl;
        sliderImgUpscaled.src = outputUrl;
        
        // Reset slider wipe handle to 50%
        sliderImgUpscaledContainer.style.clipPath = `polygon(0 0, 50% 0, 50% 100%, 0 100%)`;
        sliderHandle.style.left = '50%';

        // Load images for View 2: Side-by-Side Zoom
        zoomImgOrig.src = inputUrl;
        zoomImgUpscaled.src = outputUrl;
        
        // Reset Zoom slider to 2x
        zoomSlider.value = 2;
        zoomVal.innerText = '2.0x';
        applyZoom(2, 0.5, 0.5);
        
        // Set download link
        btnDownloadActive.href = `/api/download/${outputName}`;

        // Wait for images to load to finalize UI swap
        sliderImgUpscaled.onload = () => {
            stopUpscaleUI(true);

            if (activeViewMode === 'slider') {
                sliderView.style.display = 'flex';
            } else {
                sideView.style.display = 'grid';
            }
            workspaceToolbar.style.display = 'flex';
            viewfinderActions.style.display = 'flex';

            loadHistoryItems();
            log(`Loaded reconstruction result: ${outputName}`, 'system');
        };
    }

    // BEFORE/AFTER COMPARISON SLIDER DRAG LOGIC
    let isDraggingSlider = false;

    function moveSlider(clientX) {
        const rect = sliderContainer.getBoundingClientRect();
        let posX = clientX - rect.left;
        
        if (posX < 0) posX = 0;
        if (posX > rect.width) posX = rect.width;
        
        const pct = (posX / rect.width) * 100;
        sliderImgUpscaledContainer.style.clipPath = `polygon(0 0, ${pct}% 0, ${pct}% 100%, 0 100%)`;
        sliderHandle.style.left = `${pct}%`;
    }

    sliderHandle.addEventListener('mousedown', (e) => {
        isDraggingSlider = true;
        e.preventDefault();
    });

    window.addEventListener('mouseup', () => {
        isDraggingSlider = false;
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDraggingSlider) return;
        moveSlider(e.clientX);
    });

    sliderHandle.addEventListener('touchstart', (e) => {
        isDraggingSlider = true;
    });

    window.addEventListener('touchend', () => {
        isDraggingSlider = false;
    });

    window.addEventListener('touchmove', (e) => {
        if (!isDraggingSlider) return;
        if (e.touches.length > 0) {
            moveSlider(e.touches[0].clientX);
        }
    });

    // ==========================================
    // SIDE-BY-SIDE SYNCHRONIZED ZOOM & PAN
    // Fixed: uses CSS transform with correct origin mapping
    // ==========================================

    // Current zoom origin (relative 0 to 1)
    let zoomOriginX = 0.5;
    let zoomOriginY = 0.5;
    let currentZoom = 2;

    function applyZoom(scale, rx, ry) {
        const ox = rx * 100;
        const oy = ry * 100;
        const transformStr = `scale(${scale})`;
        
        zoomImgOrig.style.transformOrigin = `${ox}% ${oy}%`;
        zoomImgUpscaled.style.transformOrigin = `${ox}% ${oy}%`;
        zoomImgOrig.style.transform = transformStr;
        zoomImgUpscaled.style.transform = transformStr;
        
        zoomOriginX = rx;
        zoomOriginY = ry;
        currentZoom = scale;
    }

    zoomSlider.addEventListener('input', () => {
        const val = parseFloat(zoomSlider.value);
        zoomVal.innerText = `${val.toFixed(1)}x`;
        applyZoom(val, zoomOriginX, zoomOriginY);
    });

    function getRelativeMousePos(e, viewport) {
        const rect = viewport.getBoundingClientRect();
        const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
        return { x, y };
    }

    function syncViewportsPan(e, sourceViewport) {
        const { x, y } = getRelativeMousePos(e, sourceViewport);
        const scale = parseFloat(zoomSlider.value);
        applyZoom(scale, x, y);
    }

    function resetViewportsPan() {
        applyZoom(currentZoom, 0.5, 0.5);
    }

    [sideViewportOrig, sideViewportUpscaled].forEach(viewport => {
        viewport.addEventListener('mousemove', (e) => {
            syncViewportsPan(e, viewport);
        });
        
        viewport.addEventListener('mouseleave', () => {
            resetViewportsPan();
        });
    });

    // ==========================================
    // TOOLBAR TOGGLE CONTROL BUTTONS
    // ==========================================

    btnViewSlider.addEventListener('click', () => {
        activeViewMode = 'slider';
        btnViewSlider.classList.add('active');
        btnViewSide.classList.remove('active');
        
        sideView.style.display = 'none';
        sliderView.style.display = 'flex';
    });

    btnViewSide.addEventListener('click', () => {
        activeViewMode = 'side';
        btnViewSide.classList.add('active');
        btnViewSlider.classList.remove('active');

        sliderView.style.display = 'none';
        sideView.style.display = 'grid';
    });

    // Clear result button: remove from viewfinder, show placeholder.
    btnClearResult.addEventListener('click', () => {
        clearWorkspace();
    });

    // Viewfinder floating action buttons mirror left panel buttons.
    vfUpscaleBtn.addEventListener('click', () => {
        upscaleForm.dispatchEvent(new Event('submit'));
    });

    vfStopBtn.addEventListener('click', () => {
        stopBtn.click();
    });

    function clearWorkspace() {
        upscaledFilename = null;
        activeViewMode = 'slider';
        btnViewSlider.classList.add('active');
        btnViewSide.classList.remove('active');

        sliderView.style.display = 'none';
        sideView.style.display = 'none';
        loaderView.style.display = 'none';
        workspaceToolbar.style.display = 'none';
        placeholderView.style.display = 'flex';

        // Keep viewfinder actions visible only if there's still an input file loaded
        if (activeFile) {
            viewfinderActions.style.display = 'flex';
            vfUpscaleBtn.disabled = false;
        } else {
            viewfinderActions.style.display = 'none';
        }

        // Reset slider images to prevent stale content flash
        sliderImgOrig.src = '';
        sliderImgUpscaled.src = '';
        zoomImgOrig.src = '';
        zoomImgUpscaled.src = '';
        btnDownloadActive.href = '#';

        log('Cleared workspace result.', 'system');
    }

    // ==========================================
    // GALLERY HISTORY CARDS LOADER
    // ==========================================

    function loadHistoryItems() {
        fetch('/api/history')
            .then(res => res.json())
            .then(items => {
                if (items.length === 0) {
                    historyContainer.innerHTML = `
                        <div class="history-empty">
                            <i class="fa-regular fa-image"></i>
                            <p>No processed images yet.</p>
                        </div>
                    `;
                    return;
                }

                historyContainer.innerHTML = '';
                
                items.forEach(item => {
                    const card = document.createElement('div');
                    card.className = 'history-card';
                    if (upscaledFilename && upscaledFilename.startsWith(item.id)) {
                        card.classList.add('active');
                    }
                    
                    card.innerHTML = `
                        <img class="history-thumb" src="/outputs/${item.output_file}" alt="Thumbnail">
                        <div class="history-info">
                            <span class="history-name" title="${item.original_name}">${item.original_name}</span>
                            <div class="history-meta">
                                <span>${item.original_resolution} <i class="fa-solid fa-arrow-right"></i> ${item.upscaled_resolution}</span>
                            </div>
                            <span class="history-model">${item.model}</span>
                        </div>
                        <a class="history-action-btn" href="/api/download/${item.output_file}" download title="Download image">
                            <i class="fa-solid fa-download"></i>
                        </a>
                    `;
                    
                    card.addEventListener('click', (e) => {
                        if (e.target.closest('.history-action-btn')) {
                            return;
                        }
                        
                        document.querySelectorAll('.history-card').forEach(c => c.classList.remove('active'));
                        card.classList.add('active');
                        
                        upscaledFilename = item.output_file;
                        
                        placeholderView.style.display = 'none';
                        loaderView.style.display = 'none';
                        
                        loadResultImages(item.input_file, item.output_file);
                    });

                    historyContainer.appendChild(card);
                });
            })
            .catch(err => {
                console.error("Error loading history list:", err);
            });
    }

    // ==========================================
    // PAGE-RELOAD RESILIENCE: RESUME ONGOING SESSION
    // ==========================================
    function resumeActiveSession() {
        fetch('/api/status')
            .then(res => res.json())
            .then(data => {
                if (!data.active || !data.output_filename) return;

                log(`Resuming active upscale session: ${data.output_filename}`, 'system');
                upscaledFilename = data.output_filename;

                // Shift to loader view
                placeholderView.style.display = 'none';
                sliderView.style.display = 'none';
                sideView.style.display = 'none';
                workspaceToolbar.style.display = 'none';
                loaderView.style.display = 'flex';
                viewfinderActions.style.display = 'none';

                // Restore loader status from server state
                loaderTitle.innerText = data.message || 'Processing...';
                loaderPercent.innerText = `${data.progress}%`;
                loaderProgressBar.style.width = `${data.progress}%`;
                loaderTimer.innerText = `${data.elapsed.toFixed(1)}s`;

                // Lock UI
                submitBtn.disabled = true;
                stopBtn.disabled = false;
                stopBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop';
                vfUpscaleBtn.disabled = true;
                vfStopBtn.disabled = false;
                vfStopBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop';
                dropzone.style.pointerEvents = 'none';
                if (removeFileBtn) removeFileBtn.style.display = 'none';

                systemStatusMsg.innerText = 'Processing Image...';
                systemStatusMsg.parentElement.querySelector('.indicator').className = 'indicator orange';

                // Resume client timer from server elapsed
                let elapsed = data.elapsed || 0;
                clearInterval(timerInterval);
                timerInterval = setInterval(() => {
                    elapsed += 0.1;
                    loaderTimer.innerText = `${elapsed.toFixed(1)}s`;
                }, 100);

                // Reconnect SSE
                if (sseSource) sseSource.close();
                sseSource = new EventSource(`/api/progress?filename=${data.output_filename}`);

                sseSource.onmessage = (event) => {
                    const msg = JSON.parse(event.data);

                    loaderTitle.innerText = msg.message;
                    loaderPercent.innerText = `${msg.progress}%`;
                    loaderProgressBar.style.width = `${msg.progress}%`;

                    let logType = 'pytorch';
                    if (msg.message.includes('Downloading')) {
                        logType = 'download';
                        loaderMsg.innerText = 'Connecting to model repository...';
                    } else if (msg.message.includes('tile') || msg.message.includes('Tile')) {
                        logType = 'pytorch';
                        loaderMsg.innerText = 'Processing image tiles...';
                    } else if (msg.message.includes('successfully') || msg.message.includes('Completed')) {
                        logType = 'success';
                    } else if (msg.message.includes('Error')) {
                        logType = 'error';
                    } else if (msg.message.includes('Stopped')) {
                        logType = 'error';
                    } else {
                        loaderMsg.innerText = 'Running PyTorch tensor processing pipeline.';
                    }

                    log(msg.message, logType);

                    if (msg.progress >= 100 || msg.message.includes('Completed') || msg.message.includes('Error') || msg.message.includes('Stopped')) {
                        sseSource.close();
                        clearInterval(timerInterval);

                        if (msg.message.includes('Error')) {
                            showErrorModal(msg.message, 'Processing Error');
                            stopUpscaleUI(false);
                        } else if (msg.message.includes('Stopped')) {
                            log('Task was stopped by user.', 'error');
                            stopUpscaleUI(false);
                        } else {
                            log(`Inference finished. Total time: ${loaderTimer.innerText}`, 'success');
                            loadResultImagesFromOutput(data.output_filename, data.input_filename);
                        }
                    }
                };

                sseSource.onerror = () => {
                    console.error('SSE connection error during resumed session.');
                    log('Warning: SSE progress stream disconnected.', 'error');
                };
            })
            .catch(err => {
                console.error('Error checking active session:', err);
            });
    }

    function loadResultImagesFromOutput(outputName, inputName = '') {
        const outputUrl = `/outputs/${outputName}?t=${Date.now()}`;
        const inputUrl = inputName ? `/uploads/${inputName}?t=${Date.now()}` : outputUrl;

        sliderImgOrig.src = inputUrl;
        sliderImgUpscaled.src = outputUrl;
        sliderImgUpscaledContainer.style.clipPath = `polygon(0 0, 50% 0, 50% 100%, 0 100%)`;
        sliderHandle.style.left = '50%';

        zoomImgOrig.src = inputUrl;
        zoomImgUpscaled.src = outputUrl;
        zoomSlider.value = 2;
        zoomVal.innerText = '2.0x';
        applyZoom(2, 0.5, 0.5);

        btnDownloadActive.href = `/api/download/${outputName}`;

        sliderImgUpscaled.onload = () => {
            stopUpscaleUI(true);
            if (activeViewMode === 'slider') {
                sliderView.style.display = 'flex';
            } else {
                sideView.style.display = 'grid';
            }
            workspaceToolbar.style.display = 'flex';
            viewfinderActions.style.display = 'flex';
            loadHistoryItems();
            log(`Reloaded result: ${outputName}`, 'system');
        };

        sliderImgUpscaled.onerror = () => {
            // Input image may not exist; in that case, show the output.
            stopUpscaleUI(true);
            if (activeViewMode === 'slider') {
                sliderView.style.display = 'flex';
            } else {
                sideView.style.display = 'grid';
            }
            workspaceToolbar.style.display = 'flex';
            viewfinderActions.style.display = 'flex';
            loadHistoryItems();
        };
    }

    // Check for an active session on page load
    resumeActiveSession();
    loadHistoryItems();
    log("Interface components fully ready.", "system");
});
