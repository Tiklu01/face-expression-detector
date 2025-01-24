const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const context = overlay.getContext('2d');

// Load models
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
  faceapi.nets.faceExpressionNet.loadFromUri('./models')
]).then(startVideo).catch(err => {
  console.error("Failed to load models:", err);
});

// Start webcam
function startVideo() {
  navigator.mediaDevices.getUserMedia({ video: {} })
    .then(stream => {
      video.srcObject = stream;
    })
    .catch(err => console.error("Error accessing camera:", err));
}

// Detect face features
video.addEventListener('play', () => {
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(overlay, displaySize);

  setInterval(async () => {
    try {
      const detections = await faceapi
        .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceExpressions();

      context.clearRect(0, 0, overlay.width, overlay.height);
      const resizedDetections = faceapi.resizeResults(detections, displaySize);

      // Draw face detections and landmarks
      faceapi.draw.drawDetections(overlay, resizedDetections);
      faceapi.draw.drawFaceLandmarks(overlay, resizedDetections);
      faceapi.draw.drawFaceExpressions(overlay, resizedDetections);

      // Process each face
      resizedDetections.forEach(det => {
        const landmarks = det.landmarks;
        const expressions = det.expressions;
        const faceShape = classifyFaceShape(landmarks.getJawOutline(), landmarks.getFaceContour());

        // Display face shape and dominant emotion
        const box = det.detection.box;
        const dominantEmotion = getDominantEmotion(expressions);

        context.fillStyle = 'rgba(0, 0, 0, 0.6)';
        context.fillRect(box.x, box.y - 40, box.width, 40);
        context.fillStyle = '#fff';
        context.font = '16px Arial';
        context.fillText(`Shape: ${faceShape}`, box.x + 10, box.y - 25);
        context.fillText(`Emotion: ${dominantEmotion}`, box.x + 10, box.y - 10);
      });
    } catch (err) {
      console.error("Error detecting faces:", err);
    }
  }, 100);
});

// Classify face shape based on jaw and contour landmarks
function classifyFaceShape(jaw, faceContour) {
  const jawWidth = Math.abs(jaw[0].x - jaw[jaw.length - 1].x); // Distance between the ends of the jawline
  const jawHeight = Math.abs(jaw[0].y - jaw[Math.floor(jaw.length / 2)].y); // Distance from chin to middle of jawline
  const cheekboneWidth = Math.abs(jaw[3].x - jaw[jaw.length - 4].x); // Approximate cheekbone width
  const foreheadWidth = Math.abs(faceContour[17].x - faceContour[26].x); // Width of the forehead

  const jawToCheekRatio = jawWidth / cheekboneWidth;
  const jawToHeightRatio = jawWidth / jawHeight;
  const cheekToForeheadRatio = cheekboneWidth / foreheadWidth;

  // Use refined heuristics to classify face shapes
  if (jawToHeightRatio < 1.2 && cheekToForeheadRatio < 1.5) return 'Round';
  if (jawToHeightRatio > 1.5 && cheekToForeheadRatio > 1.4) return 'Square';
  if (jawToHeightRatio > 1.3 && cheekToForeheadRatio < 1.5) return 'Oval';
  if (jaw[0].x < jaw[Math.floor(jaw.length / 2)].x) return 'Heart';
  if (cheekToForeheadRatio < 1.2 && jawToCheekRatio < 1.5) return 'Diamond';
  return 'Undetermined'; // Fallback for unclear classifications
}

// Get the dominant emotion based on probabilities
function getDominantEmotion(expressions) {
  let maxEmotion = '';
  let maxValue = 0;
  for (const [emotion, value] of Object.entries(expressions)) {
    if (value > maxValue) {
      maxValue = value;
      maxEmotion = emotion;
    }
  }
  return maxEmotion || 'Neutral';
}
