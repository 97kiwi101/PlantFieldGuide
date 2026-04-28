const video = document.getElementById('camera-feed');
const canvas = document.getElementById('photo-canvas');
const captureBtn = document.getElementById('capture-btn');
const resetBtn = document.getElementById('reset-btn');
const resultArea = document.getElementById('result-area');
const statusText = document.getElementById('status-text');

let plantDatabase = {};
let aiSession = null;

window.addEventListener('load', async () => {
    captureBtn.disabled = true; // Disable camera button until AI is loaded

    try {
        // Load the local JSON database
        const response = await fetch('enriched_plant_database.json');
        plantDatabase = await response.json();
    } catch (error) {
        console.error("Failed to load plant database:", error);
    }

    try {
        // Load the Quantized ONNX Model into the Browser
        // Note: wasm (WebAssembly) is the fastest execution backend for browsers
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        aiSession = await ort.InferenceSession.create('./plant_model.onnx', { executionProviders: ['wasm'] });
        
        statusText.textContent = "AI Ready! Point your camera at a plant.";
        captureBtn.disabled = false;
    } catch (error) {
        console.error("Failed to load AI model:", error);
        statusText.textContent = "Error loading AI model. Check console.";
    }
    
    startCamera();
});

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        video.srcObject = stream;
    } catch (err) {
        console.error("Error accessing camera: ", err);
    }
}

captureBtn.addEventListener('click', async () => {
    if (!aiSession) return;

    captureBtn.classList.add('d-none');
    resultArea.classList.remove('d-none');
    document.getElementById('loading-spinner').classList.remove('d-none');
    document.getElementById('trefle-card').classList.add('d-none');

    // 1. Capture the frame to a hidden 224x224 canvas
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Calculate crop to keep aspect ratio perfect
    const minSize = Math.min(video.videoWidth, video.videoHeight);
    const startX = (video.videoWidth - minSize) / 2;
    const startY = (video.videoHeight - minSize) / 2;
    ctx.drawImage(video, startX, startY, minSize, minSize, 0, 0, 224, 224);

    // 2. Pre-process the image for the AI
    const imgData = ctx.getImageData(0, 0, 224, 224).data;
    const float32Data = new Float32Array(3 * 224 * 224);
    
    // PyTorch Normalization standards
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // Convert RGBA array to CHW (Channel, Height, Width) Float32 format
    for (let i = 0; i < 224 * 224; i++) {
        let r = imgData[i * 4] / 255.0;
        let g = imgData[i * 4 + 1] / 255.0;
        let b = imgData[i * 4 + 2] / 255.0;

        float32Data[i] = (r - mean[0]) / std[0]; // Red Channel
        float32Data[i + 224 * 224] = (g - mean[1]) / std[1]; // Green Channel
        float32Data[i + 2 * 224 * 224] = (b - mean[2]) / std[2]; // Blue Channel
    }

    // 3. Run Inference
    try {
        const tensor = new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
        const results = await aiSession.run({ 'input': tensor });
        
        // 4. Find the highest probability prediction
        const outputArray = results.output.data;
        let maxIndex = 0;
        let maxValue = outputArray[0];

        for (let i = 1; i < outputArray.length; i++) {
            if (outputArray[i] > maxValue) {
                maxValue = outputArray[i];
                maxIndex = i;
            }
        }

        displayResults(maxIndex);
    } catch (err) {
        console.error("AI execution error:", err);
        alert("Failed to process image.");
    }
});

function displayResults(predictedId) {
    document.getElementById('loading-spinner').classList.add('d-none');
    
    // We assume the predicted ID directly maps to the index of your classes
    // Note: Ensure your JSON keys align with the PyTorch output index!
    const plantInfo = plantDatabase[predictedId]; 

    if (plantInfo) {
        document.getElementById('plant-name').textContent = plantInfo.name || "Unknown Name";
        document.getElementById('plant-id').textContent = `Scientific: ${plantInfo.scientificName || "-"}`;
        document.getElementById('plant-family').textContent = plantInfo.family || 'Unknown';
        document.getElementById('plant-genus').textContent = plantInfo.genus || 'Unknown';
    } else {
        document.getElementById('plant-name').textContent = `Species ID: ${predictedId}`;
        document.getElementById('plant-id').textContent = "Details missing from local JSON.";
    }
    
    document.getElementById('plant-name').classList.remove('d-none');
    document.getElementById('plant-id').classList.remove('d-none');
    document.getElementById('trefle-card').classList.remove('d-none');
    resetBtn.classList.remove('d-none');
}

resetBtn.addEventListener('click', () => {
    resultArea.classList.add('d-none');
    captureBtn.classList.remove('d-none');
    resetBtn.classList.add('d-none');
});