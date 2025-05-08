<template>
  <div id="app">
    <header class="app-header">
      <h1><i class="fas fa-shield-alt"></i> 基于YOLOv11的安检X光检测系统</h1>
    </header>
    <section class="image-detection-section">
      <div class="image-container">
        <img v-if="imageUrl" :src="imageUrl" alt="Uploaded Image" ref="uploadedImage" @load="drawDetections">
        <div v-else class="placeholder-image">请上传图片</div>
        <canvas ref="detectionCanvas" :width="imageWidth" :height="imageHeight" style="position: absolute; left: 0; top: 0;"></canvas>
      </div>
      <div class="right-side-container">
        <section class="upload-section">
          <label for="file-upload" class="upload-label">
            <i class="fas fa-cloud-upload-alt"></i> 选择图片
          </label>
          <input id="file-upload" type="file" @change="handleFileUpload" accept="image/*" style="display: none;">
          <button @click="uploadImage" :disabled="!selectedFile || isUploading" class="upload-button">
            <i v-if="isUploading" class="fas fa-spinner fa-spin"></i>
            <span v-else style="font-size:20px"><i class="fas fa-upload"></i> 上传检测</span>
          </button>
      <p v-if="uploadError" class="error-message">{{ uploadError }}</p>
    </section>
        <div class="controls-container">
          <div class="confidence-control">
            <label for="confidenceThreshold"><i class="fas fa-sliders-h"></i> 置信度阈值: {{ confidenceThreshold.toFixed(2) }}</label>
            <input type="range" id="confidenceThreshold" v-model.number="confidenceThreshold" min="0" max="1" step="0.01" class="slider">
          </div>
          <button @click="clearDetections" class="clear-button"><i class="fas fa-eraser"></i> 清除检测结果</button>
        </div>
        <section class="results-section">
          <h2><i class="fas fa-list-alt"></i> 检测结果</h2>
          <ul v-if="filteredDetections.length > 0" class="detection-list">
            <li v-for="(detection, index) in filteredDetections" :key="index" class="detection-item">
              <span class="label"><i class="fas fa-tag"></i> {{ detection.label }}</span>
              <span class="confidence">置信度: {{ detection.confidence.toFixed(2) }}</span>
            </li>
          </ul>
          <p v-else-if="!isUploading && !uploadError">
            <span v-if="imageUrl && detections.length === 0"><i class="fas fa-exclamation-circle"></i> 未检测到任何目标。</span>
            <span v-else-if="imageUrl"><i class="fas fa-filter"></i> 请调整置信度阈值</span>
            <span v-else><i class="fas fa-info-circle"></i> 请上传图片进行检测</span>
          </p>
        </section>
      </div>
    </section>
  </div>
</template>

<style scoped>
.placeholder-image {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  background-color: #f9f9f9;
  color: #ccc;
  border: 1px dashed #ddd;
  border-radius: 8px;
}
</style>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      selectedFile: null,
      imageUrl: null,
      isUploading: false,
      uploadError: null,
      detections: [],
      imageWidth: 0,
      imageHeight: 0,
      backendUrl: 'http://localhost:8000',
      confidenceThreshold: 0.5,
    };
  },
  computed: {
    filteredDetections() {
      return this.detections.filter(detection => detection.confidence >= this.confidenceThreshold);
    },
  },
  methods: {
    handleFileUpload(event) {
      this.selectedFile = event.target.files[0];
      this.imageUrl = URL.createObjectURL(this.selectedFile);
      this.detections = [];
      this.uploadError = null;
    },
    async uploadImage() {
      if (!this.selectedFile) {
        this.uploadError = '请选择一张图片';
        return;
      }

      this.isUploading = true;
      this.uploadError = null;
      this.detections = [];

      const formData = new FormData();
      formData.append('file', this.selectedFile);

      try {
        const response = await axios.post(`${this.backendUrl}/detect/`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        this.detections = response.data.detections;
        this.drawDetections();
      } catch (error) {
        console.error('上传图片失败:', error);
        this.uploadError = '图片上传或检测失败，请检查后端服务。';
      } finally {
        this.isUploading = false;
      }
    },
    clearDetections() {
      this.detections = [];
      const detectionCanvas = this.$refs.detectionCanvas;
      if (detectionCanvas) {
        const ctx = detectionCanvas.getContext('2d');
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
      }
    },
    drawDetections() {
      const uploadedImage = this.$refs.uploadedImage;
      const detectionCanvas = this.$refs.detectionCanvas;
      if (!uploadedImage || !detectionCanvas) {
        return;
      }
      const ctx = detectionCanvas.getContext('2d');
      ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
      this.imageWidth = uploadedImage.naturalWidth;
      this.imageHeight = uploadedImage.naturalHeight;
      detectionCanvas.width = this.imageWidth;
      detectionCanvas.height = this.imageHeight;
      ctx.lineWidth = 3;
      this.filteredDetections.forEach(detection => {
        const [x1, y1, x2, y2] = detection.box;
        const label = `${detection.label} (${detection.confidence.toFixed(2)})`;
        const color = '#28a745';
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.strokeRect(x1*0.6, y1*0.6, (x2 - x1)*0.6, (y2 - y1)*0.6);
        const textPadding = 5;
        const textHeight = 20;
        ctx.font = `bold ${textHeight}px sans-serif`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x1*0.6, (y1 - textHeight - 2 * textPadding)*0.6, (textWidth + 2 * textPadding)*0.6, (textHeight + 2 * textPadding)*0.6);
        ctx.fillStyle = 'white';
        ctx.fillText(label, (x1 + textPadding)*0.6, (y1 - textPadding)*0.6);
      });
    },
  },
};
</script>
<style scoped>
#app {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #333;
  margin: 40px auto;
  max-width: 1200px;
  padding: 30px;
  background-color: #f4f6f8;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  align-items: center;
}

.app-header {
  text-align: center;
  margin-bottom: 40px;
}

.app-header h1 {
  color: #2c3e50;
  font-size: 2.5rem;
  margin-bottom: 10px;
}

.app-header h1 i {
  margin-right: 10px;
}

.upload-section {
  text-align: center;
  margin-bottom: 40px;
  padding: 30px;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}

.upload-label {
  display: inline-block;
  padding: 12px 24px;
  background-color: #007bff;
  color: white;
  border-radius: 6px;
  cursor: pointer;
  font-size: 20px;
  margin-bottom: 15px;
  transition: background-color 0.3s ease;
}

.upload-label:hover {
  background-color: #0056b3;
}

.upload-label i {
  margin-right: 8px;
}

.upload-button {
  padding: 12px 24px;
  background-color: #28a745;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s ease;
}

.upload-button:hover {
  background-color: #1e7e34;
}

.upload-button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.upload-button i {
  margin-right: 8px;
}

.image-detection-section {
  display: flex;
  gap: 30px;
  margin-bottom: 40px;
  width: 100%;
}

.image-container {
  position: relative;
  width: 576px;
  height: 576px;
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}

.image-container img {
  display: block;
  width: 100%;
  height: auto;
}

.right-side-container {
  display: flex;
  flex-direction: column;
  width: 35%;
}

.controls-container {
  padding: 20px;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
  margin-bottom: 30px;
}

.confidence-control {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.controls-container label {
  font-weight: bold;
  font-size: 20px;
  color: #555;
}

.slider {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 8px;
  background: #ddd;
  border-radius: 4px;
  outline: none;
  opacity: 0.7;
  -webkit-transition: .2s;
  transition: opacity .2s;
}

.slider:hover {
  opacity: 1;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  background: #007bff;
  border-radius: 50%;
  cursor: pointer;
}

.slider::-moz-range-thumb {
  width: 16px;
  height: 16px;
  background: #007bff;
  border-radius: 50%;
  cursor: pointer;
}

.clear-button {
  padding: 10px 15px;
  background-color: #6c757d;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s ease;
}

.clear-button:hover {
  background-color: #5a6268;
}

.clear-button i {
  margin-right: 8px;
}

.results-section {
  text-align: left;
  padding: 30px;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}

.results-section h2 {
  color: #28a745;
  font-size: 2rem;
  margin-bottom: 20px;
}

.results-section h2 i {
  margin-right: 10px;
}

.detection-list {
  list-style: none;
  padding: 0;
}

.detection-item {
  margin-bottom: 12px;
  padding: 15px;
  background-color: #f9f9f9;
  border-radius: 6px;
  border: 1px solid #eee;
}

.detection-item .label {
  font-weight: bold;
  color: #007bff;
  margin-right: 10px;
}

.detection-item .label i {
  margin-right: 5px;
}

.detection-item .confidence {
  color: #555;
  margin-right: 10px;
}

.detection-item .coordinates {
  color: #777;
  font-size: 0.9rem;
}

.error-message {
  color: #dc3545;
  margin-top: 15px;
  font-size: 16px;
}

</style>

