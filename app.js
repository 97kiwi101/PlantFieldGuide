const video = document.getElementById('camera-feed');
const canvas = document.getElementById('photo-canvas');
const captureBtn = document.getElementById('capture-btn');
const resetBtn = document.getElementById('reset-btn');
const resultCard = document.getElementById('result-card');
const statusText = document.getElementById('status-text');
const spinner = document.getElementById('loading-spinner');

let plantDatabase = {};
let aiSession = null;

// Initialize App
window.addEventListener('load', async () => {
    captureBtn.disabled = true;

    // 1. Load the JSON database
    try {
        const response = await fetch('enriched_plant_database.json');
        plantDatabase = await response.json();
    } catch (error) {
        console.error("Failed to load JSON database:", error);
    }

    // 2. Load the Uncompressed ONNX Model (84MB)
    try {
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        aiSession = await ort.InferenceSession.create('./plant_model_single.onnx', { executionProviders: ['wasm'] });
        
        statusText.textContent = "AI Ready! Point & Shoot.";
        statusText.classList.replace('text-warning', 'text-success');
        captureBtn.disabled = false;
    } catch (error) {
        console.error("Failed to load AI model:", error);
        statusText.textContent = "Error loading AI.";
        statusText.classList.replace('text-warning', 'text-danger');
    }
    
    // 3. Start Camera
    startCamera();
});

async function startCamera() {
    try {
        // Prioritize the back camera for mobile devices
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        video.srcObject = stream;
    } catch (err) {
        console.error("Error accessing camera: ", err);
        statusText.textContent = "Camera access denied.";
    }
}

captureBtn.addEventListener('click', async () => {
    if (!aiSession) return;

    // UI Changes during processing
    captureBtn.classList.add('d-none');
    resultCard.classList.add('d-none');
    spinner.classList.remove('d-none');
    statusText.textContent = "Analyzing structure...";

    // 1. Capture the frame to a hidden 224x224 canvas
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Calculate center crop to keep aspect ratio perfect
    const minSize = Math.min(video.videoWidth, video.videoHeight);
    const startX = (video.videoWidth - minSize) / 2;
    const startY = (video.videoHeight - minSize) / 2;
    ctx.drawImage(video, startX, startY, minSize, minSize, 0, 0, 224, 224);

    // Pause the video to freeze the frame
    video.pause();

    // 2. Pre-process the image for the AI (Normalize to PyTorch standards)
    const imgData = ctx.getImageData(0, 0, 224, 224).data;
    const float32Data = new Float32Array(3 * 224 * 224);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < 224 * 224; i++) {
        let r = imgData[i * 4] / 255.0;
        let g = imgData[i * 4 + 1] / 255.0;
        let b = imgData[i * 4 + 2] / 255.0;

        float32Data[i] = (r - mean[0]) / std[0]; // Red
        float32Data[i + 224 * 224] = (g - mean[1]) / std[1]; // Green
        float32Data[i + 2 * 224 * 224] = (b - mean[2]) / std[2]; // Blue
    }

    // 3. Run Inference
    try {
        const tensor = new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
        const results = await aiSession.run({ 'input': tensor });
        
        // Find the highest probability prediction
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
        statusText.textContent = "AI calculation failed.";
        spinner.classList.add('d-none');
        captureBtn.classList.remove('d-none');
        video.play();
    }
});

function displayResults(predictedId) {
    spinner.classList.add('d-none');
    statusText.textContent = "Identification Complete";
    
    // Pull the data from the JSON file using the predicted Index
    const plantInfo = plantDatabase[predictedId]; 

    if (plantInfo) {
        document.getElementById('plant-name').textContent = plantInfo.name || "Unknown Name";
        document.getElementById('plant-id').textContent = plantInfo.scientificName || "Unknown Species";
        document.getElementById('plant-family').textContent = plantInfo.family || 'Unknown';
        document.getElementById('plant-genus').textContent = plantInfo.genus || 'Unknown';
    } else {
        document.getElementById('plant-name').textContent = `Species ID: ${predictedId}`;
        document.getElementById('plant-id').textContent = "Not found in local database.";
        document.getElementById('plant-family').textContent = "-";
        document.getElementById('plant-genus').textContent = "-";
    }
    
    // Reveal Results & Reset Button
    resultCard.classList.remove('d-none');
    resetBtn.classList.remove('d-none');
}

resetBtn.addEventListener('click', () => {
    // Reset UI to scan again
    resultCard.classList.add('d-none');
    resetBtn.classList.add('d-none');
    captureBtn.classList.remove('d-none');
    statusText.textContent = "AI Ready! Point & Shoot.";
    
    // Unfreeze camera
    video.play();
});