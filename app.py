import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template_string
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# 1. Recreate the exact architecture of your trained model
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) 
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 2. Load your weights into the architecture
device = torch.device("cpu")
model = MNISTCNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# 3. Real-Time Vanilla HTML/JS Drawing Interface
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>MNIST Live Classifier</title>
    <style>
        body { font-family: sans-serif; text-align: center; background: #fafafa; padding: 30px; }
        .wrapper { display: flex; justify-content: center; gap: 50px; margin-top: 30px; }
        canvas { border: 3px solid #222; background: #fff; cursor: crosshair; border-radius: 4px; }
        button { padding: 12px 24px; font-size: 16px; font-weight: bold; cursor: pointer; border: none; border-radius: 4px; background: #dc3545; color: white; margin: 5px; }
        button:hover { background: #bd2130; }
        .dashboard { text-align: left; background: white; padding: 25px; border-radius: 6px; border: 1px solid #ddd; width: 280px; }
        .row { display: flex; align-items: center; margin-bottom: 10px; }
        .num { width: 25px; font-weight: bold; font-size: 16px; }
        .track { flex-grow: 1; height: 16px; background: #e9ecef; border-radius: 8px; overflow: hidden; }
        .fill { height: 100%; background: #28a745; width: 0%; transition: width 0.1s ease; }
        .pct { width: 55px; text-align: right; font-size: 14px; color: #666; margin-left: 10px; }
    </style>
</head>
<body>
    <h1>MNIST Live Canvas Classifier</h1>
    <div class="wrapper">
        <div>
            <canvas id="paintCanvas" width="280" height="280"></canvas>
            <div style="margin-top: 15px;">
                <button onclick="resetCanvas()">Clear Canvas</button>
            </div>
        </div>
        <div class="dashboard">
            <h3>Network Confidence</h3>
            <div id="bars-container"></div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('paintCanvas');
        const ctx = canvas.getContext('2d');
        let drawing = false;
        let isThrottled = false;

        // Set up brush settings
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 16;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        resetCanvas();

        // Mouse Event Listeners
        canvas.addEventListener('mousedown', (e) => { drawing = true; paint(e); });
        canvas.addEventListener('mousemove', (e) => {
            paint(e);
            if (drawing) {
                livePredict(); // Triggers dynamically as you draw
            }
        });
        canvas.addEventListener('mouseup', () => { 
            drawing = false; 
            ctx.beginPath(); 
            sendToBackend(); // Run one final prediction on release
        });
        canvas.addEventListener('mouseleave', () => { drawing = false; ctx.beginPath(); });

        function paint(e) {
            if (!drawing) return;
            const boundary = canvas.getBoundingClientRect();
            ctx.lineTo(e.clientX - boundary.left, e.clientY - boundary.top);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(e.clientX - boundary.left, e.clientY - boundary.top);
        }

        function resetCanvas() {
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            // Reset all bars back to 0
            for(let i=0; i<10; i++) {
                const bar = document.getElementById(`bar-${i}`);
                const pct = document.getElementById(`pct-${i}`);
                if (bar && pct) {
                    bar.style.width = '0%';
                    pct.innerText = '0.0%';
                }
            }
        }

        function buildEmptyBars() {
            const container = document.getElementById('bars-container');
            container.innerHTML = '';
            for(let i=0; i<10; i++) {
                container.innerHTML += `
                    <div class="row">
                        <span class="num">${i}</span>
                        <div class="track"><div class="fill" id="bar-${i}"></div></div>
                        <span class="pct" id="pct-${i}">0.0%</span>
                    </div>`;
            }
        }

        // Throttle requests using requestAnimationFrame to protect server performance
        function livePredict() {
            if (isThrottled) return;
            isThrottled = true;
            
            window.requestAnimationFrame(() => {
                sendToBackend();
                isThrottled = false;
            });
        }

        async function sendToBackend() {
            const b64Image = canvas.toDataURL('image/png');
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: b64Image })
                });
                const result = await response.json();
                
                // Update percentage bars smoothly
                result.probabilities.forEach((prob, index) => {
                    const percentage = (prob * 100).toFixed(1);
                    document.getElementById(`bar-${index}`).style.width = percentage + '%';
                    document.getElementById(`pct-${index}`).innerText = percentage + '%';
                });
            } catch (err) {
                console.error("Prediction error:", err);
            }
        }
        
        window.onload = buildEmptyBars;
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    raw_b64 = payload['image'].split(',')[1]
    
    img = Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = (img_array - 0.1307) / 0.3081
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        log_predictions = model(img_tensor)
        probabilities = torch.exp(log_predictions)
        
    return jsonify({'probabilities': probabilities[0].tolist()})

if __name__ == '__main__':
    app.run(debug=True)