<template>
  <div id="app">
    <section class="image-detection-section">
      <div class="image-container">
        <img v-if="imageUrl" :src="imageUrl" alt="Uploaded Image" ref="uploadedImage" @load="drawDetections">
        <div v-else class="placeholder-image">请上传图片</div>
        <canvas ref="detectionCanvas" :width="imageWidth" :height="imageHeight" style="position: absolute; left: 0; top: 0;"></canvas>
      </div>
      <div class="right-side-container">
        <fieldset class="upload-section">
          <legend class="fieldset-legend">操作</legend>
          <label for="file-upload" class="upload-label">
            <i class="fas fa-cloud-upload-alt"></i> 图像
          </label>
          <input id="file-upload" type="file" @change="handleFileUpload" accept="image/*" style="display: none;">
          <label for="vedio-upload" class="upload-vedio">
            <i class="fas fa-cloud-upload-alt"></i> 视频
          </label>
          <input id="vedio-upload" type="file" @change="handleFileUpload" accept="video/*" style="display: none;">
          <label for="stream-upload" class="upload-stream">
            <i class="fas fa-cloud-upload-alt"></i> 视频流
          </label>
          <input id="stream-upload" type="file" @change="handleFileUpload" accept="video/*" style="display: none;">
          <div class="upload-model" @click.stop>
            <button type="button" @click="toggleModelDropdown" class="model-select-button">
              <i class="fas fa-brain"></i> <span>选择模型</span>
              <i :class="isModelDropdownOpen ? 'fas fa-chevron-up' : 'fas fa-chevron-down'" class="dropdown-arrow"></i>
            </button>
            <ul v-if="isModelDropdownOpen" class="model-dropdown-list">
              <li v-for="model in availableModels"
                  :key="model.value"
                  @click="selectModel(model)"
                  :class="{ 'selected': model.value === selectedModelValue }">
                {{ model.name }}
              </li>
            </ul>
          </div>
          <button @click="uploadImage" :disabled="!selectedFile || isUploading" class="upload-button">
            <i :class="isUploading ? 'fas fa-spinner fa-spin' : 'fas fa-upload'"></i>
            <span>{{ isUploading ? '上传中...' : '上传检测' }}</span>
          </button>
          <button @click="clearDetections" class="clear-button">
            <i class="fas fa-eraser"></i> 清除
          </button>
          <button @click="selectSaveLocation" class="save-location-button">
            <i class="fas fa-save"></i> 保存
          </button>
          <p v-if="uploadError && !isUploading" class="error-message">{{ uploadError }}</p>
        </fieldset>
        <section>
          <fieldset class="controls-fieldset">
            <legend class="controls-legend">参数</legend>
            <div class="confidence-control">
              <label for="confidenceThreshold"><i class="fas fa-sliders-h"></i> 置信度阈值: {{ confidenceThreshold.toFixed(2) }}</label>
              <input type="range" id="confidenceThreshold" v-model.number="confidenceThreshold" min="0" max="1" step="0.01" class="slider">
            </div>
            <div class="iou-control">
              <label for="iouThreshold"><i class="fas fa-sliders-h"></i> IoU阈值: {{ iouThreshold.toFixed(2) }}</label>
              <input type="range" id="iouThreshold" v-model.number="iouThreshold" min="0" max="1" step="0.01" class="slider">
            </div>
          </fieldset>
        </section>
      </div>
    </section>

    <fieldset class="results-section">
      <legend class="result-legend">检测结果</legend>
      <div v-if="imageUrl && detections.length > 0 && !isUploading && !uploadError" class="results-summary">
        <p><i class="fas fa-clock"></i> <strong>检测耗时(ms):</strong> {{ detectionTime || 'N/A' }}</p>
        <p><i class="fas fa-clipboard-list"></i> <strong>目标总数:</strong> 共发现 {{ tableDetections.length }} 个目标 </p>
      </div>

      <div v-if="tableDetections.length > 0 && !isUploading && !uploadError" class="table-responsive">
        <table class="detection-table">
          <thead>
            <tr>
              <th><i class="fas fa-tag"></i> 目标</th>
              <th><i class="fas fa-percentage"></i> 置信度</th>
              <th><i class="fas fa-arrows-alt-h"></i> X中心</th>
              <th><i class="fas fa-arrows-alt-v"></i> Y中心</th>
              <th><i class="fas fa-ruler-horizontal"></i> 宽度</th>
              <th><i class="fas fa-ruler-vertical"></i> 高度</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(detection, index) in tableDetections" :key="index" class="detection-item-row">
              <td>{{ detection.label }}</td>
              <td>{{ detection.confidence.toFixed(3) }}</td>
              <td>{{ detection.x_center.toFixed(1) }}</td>
              <td>{{ detection.y_center.toFixed(1) }}</td>
              <td>{{ detection.width.toFixed(1) }}</td>
              <td>{{ detection.height.toFixed(1) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else-if="!isUploading && !uploadError" class="no-results-message">
        <span v-if="imageUrl && detections.length === 0"><i class="fas fa-exclamation-circle"></i> 未检测到任何目标。</span>
        <span v-else-if="imageUrl && filteredDetections.length === 0 && detections.length > 0"><i class="fas fa-filter"></i> 未筛选出结果，请尝试调整置信度阈值。</span>
        <span v-else-if="!imageUrl"><i class="fas fa-info-circle"></i> 请上传图片进行检测。</span>
      </p>
      </fieldset>
  </div>
</template>

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
      selectedModelValue: 'yolo11n',
      isModelDropdownOpen: false,
      availableModels: [
        { name: 'YOLOv11 Nano', value: 'yolo11n' },
        { name: 'YOLOv11 Small', value: 'yolo11s' },
        { name: 'YOLOv11 Medium', value: 'yolo11m' },
        { name: 'YOLOv11 Large', value: 'yolo11l' },
        { name: 'YOLOv11 Xtra', value: 'yolo11x' },
      ],
      confidenceThreshold: 0.5,
      iouThreshold: 0.5,
      saveDirectoryHandle: null,
      backendAnnotatedImageBase64: null,
      canUseFileSystemAccessAPI: ('showDirectoryPicker' in window),
      detectionTime: null,
    };
  },
  computed: {
    selectedModelDisplayName() {
      const model = this.availableModels.find(m => m.value === this.selectedModelValue);
      return model ? model.name : '选择模型';
    },
    filteredDetections() {
      return this.detections.filter(d => d.confidence >= this.confidenceThreshold);
    },
    tableDetections() {
      return this.filteredDetections.map(det => {
        const [x1, y1, x2, y2] = det.box;
        const width = x2 - x1;
        const height = y2 - y1;
        return {
          label: det.label,
          confidence: det.confidence,
          x_center: x1 + width / 2,
          y_center: y1 + height / 2,
          width: width,
          height: height,
        };
      });
    },
  },
  methods: {
    handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      if (this.imageUrl) {
        URL.revokeObjectURL(this.imageUrl);
      }

      this.selectedFile = file;
      this.imageUrl = URL.createObjectURL(file);
      this.detections = [];
      this.uploadError = null;
      this.backendAnnotatedImageBase64 = null;
      this.detectionTime = null;

      const canvas = this.$refs.detectionCanvas;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    },
    toggleModelDropdown() {
      this.isModelDropdownOpen = !this.isModelDropdownOpen;
    },
    selectModel(model) {
      this.selectedModelValue = model.value;
      this.isModelDropdownOpen = false;
      console.log('已选择模型:', this.selectedModelValue);
    },
    async selectSaveLocation() {
      if (!this.canUseFileSystemAccessAPI) {
        alert('当前浏览器不支持文件夹选择API。');
        return;
      }
      try {
        this.saveDirectoryHandle = await window.showDirectoryPicker();
        alert(`已选择保存目录：${this.saveDirectoryHandle.name}`);
      } catch (e) {
        console.warn('用户取消或拒绝权限', e);
        this.saveDirectoryHandle = null;
      }
    },
    async uploadImage() {
      if (!this.selectedFile) {
        this.uploadError = '请选择一张图片。';
        return;
      }
      if (!this.selectedModelValue) {
        this.uploadError = '请选择一个模型。';
        return;
      }
      this.isUploading = true;
      this.uploadError = null;
      this.detections = [];
      this.backendAnnotatedImageBase64 = null;
      this.detectionTime = null;

      const formData = new FormData();
      formData.append('file', this.selectedFile);
      formData.append('confidence_threshold', this.confidenceThreshold);
      formData.append('iou_threshold', this.iouThreshold);
      formData.append('model_name', this.selectedModelValue);

      try {
        const res = await axios.post(`${this.backendUrl}/detect/`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        this.detections = res.data.detections;
        this.backendAnnotatedImageBase64 = res.data.annotated_image_base64;
        this.detectionTime = res.data.detection_time;
        this.$nextTick(async () => {
          this.drawDetections();
          if (this.saveDirectoryHandle && this.detections.length > 0 && this.backendAnnotatedImageBase64) {
            await this.saveResultsToSelectedDirectory();
          }
        });
      } catch (err) {
        console.error('上传或检测失败:', err);
        this.uploadError = `检测服务出错: ${err.response ? (err.response.data.detail || err.message) : err.message}`;
      } finally {
        this.isUploading = false;
      }
    },
    async saveResultsToSelectedDirectory() {
      if (!this.detections.length || !this.backendAnnotatedImageBase64) {
        return console.warn('没有检测结果可保存');
      }
      const now = new Date();
      const datePart = now.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' }).replace(/\//g, '');
      const timePart = now.toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }).replace(/:/g, '');
      const timestamp = `${datePart}_${timePart}`;

      try {
        const subDir = await this.saveDirectoryHandle.getDirectoryHandle(timestamp, { create: true });

        const blob = this.base64ToBlob(this.backendAnnotatedImageBase64, 'image/jpeg');
        const imgName = `${timestamp}_annotated.jpg`;
        const imgFileHandle = await subDir.getFileHandle(imgName, { create: true });
        const imgWritable = await imgFileHandle.createWritable();
        await imgWritable.write(blob);
        await imgWritable.close();
        console.log('已保存标注图片：', imgName);
        const jsonToSave = this.filteredDetections;
        const jsonBlob = new Blob([JSON.stringify(jsonToSave, null, 2)], { type: 'application/json' });
        const jsonName = `${timestamp}_detections.json`;
        const jsonFileHandle = await subDir.getFileHandle(jsonName, { create: true });
        const jsonWritable = await jsonFileHandle.createWritable();
        await jsonWritable.write(jsonBlob);
        await jsonWritable.close();
        console.log('已保存 JSON：', jsonName);
        alert(`结果已保存到本地目录 ${this.saveDirectoryHandle.name} 下的 ${timestamp} 文件夹中。`);

      } catch(e) {
          console.error("保存文件失败:", e);
          alert("保存文件失败");
      }
    },
    clearDetections() {
      if (this.imageUrl) {
        URL.revokeObjectURL(this.imageUrl);
      }
      this.selectedFile = null;
      this.imageUrl = null;
      this.detections = [];
      this.uploadError = null;
      this.backendAnnotatedImageBase64 = null;
      this.detectionTime = null;
      const fileInputIds = ['file-upload', 'vedio-upload', 'stream-upload'];
      fileInputIds.forEach(id => {
        const inputElement = document.getElementById(id);
        if (inputElement) inputElement.value = '';
      });
      const canvas = this.$refs.detectionCanvas;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    },
    drawDetections() {
      const imgEl = this.$refs.uploadedImage;
      const canvas = this.$refs.detectionCanvas;

      if (!imgEl || !canvas || !this.imageUrl) {
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        return;
      }
      if (!imgEl.complete || imgEl.naturalWidth === 0 || imgEl.naturalHeight === 0) {
        console.warn("drawDetections called before image loaded or image has no dimensions.");
        return;
      }
      const origW = imgEl.naturalWidth;
      const origH = imgEl.naturalHeight;
      const dispW = imgEl.clientWidth;
      const dispH = imgEl.clientHeight;
      if (this.imageWidth !== origW || this.imageHeight !== origH) {
          this.imageWidth = origW;
          this.imageHeight = origH;
      }
      if (canvas.width !== dispW || canvas.height !== dispH) {
          canvas.width = dispW;
          canvas.height = dispH;
      }
      const scaleX = dispW / origW;
      const scaleY = dispH / origH;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, dispW, dispH);
      if (this.filteredDetections.length === 0) return;
      ctx.lineWidth = Math.max(1, Math.min(dispW, dispH) / 300); // Dynamic line width
      const fontSize = Math.max(10, Math.min(dispW, dispH) / 40); // Dynamic font size
      ctx.font = `bold ${fontSize}px sans-serif`;
      this.filteredDetections.forEach(det => {
        const [x1, y1, x2, y2] = det.box;
        const label = `${det.label} (${det.confidence.toFixed(2)})`;
        const rx1 = x1 * scaleX;
        const ry1 = y1 * scaleY;
        const rw = (x2 - x1) * scaleX;
        const rh = (y2 - y1) * scaleY;
        ctx.strokeStyle = '#28a745';
        ctx.strokeRect(rx1, ry1, rw, rh);
        const padding = fontSize * 0.3;
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = fontSize * 1.2;
        ctx.fillStyle = '#28a745';
        ctx.fillRect(
          rx1,
          ry1 - textHeight,
          textWidth + padding * 2,
          textHeight
        );
        ctx.fillStyle = 'white';
        ctx.fillText(label, rx1 + padding, ry1 - padding * 0.5);
      });
    },
    base64ToBlob(base64, type = 'application/octet-stream') {
      try {
        const parts = base64.split(',');
        const b64Data = parts.length > 1 ? parts[1] : parts[0];
        const byteCharacters = atob(b64Data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type });
      } catch (e) {
        console.error("Base64 to Blob conversion failed:", e);
        return new Blob([]);
      }
    },
    handleClickOutside(event) {
      const dropdownContainer = this.$el.querySelector('.upload-model');
      if (dropdownContainer && !dropdownContainer.contains(event.target)) {
        this.isModelDropdownOpen = false;
      }
    },
  },
  mounted() {
    document.addEventListener('click', this.handleClickOutside);
  },
  beforeUnmount() {
    document.removeEventListener('click', this.handleClickOutside);
    if (this.imageUrl) {
      URL.revokeObjectURL(this.imageUrl);
    }
  },
};
</script>

<style scoped>
#app {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #333;
  margin: 20px auto;
  max-width: 1200px;
  padding: 20px;
  background-color: #f8f4f6;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  align-items: stretch;
  width: 100%;
  min-height: 90vh;
  height: auto;
  box-sizing: border-box;
}
.image-detection-section {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  margin-bottom: 20px;
  width: 100%;
}
.image-container {
  position: relative;
  flex: 1 1 550px;
  min-width: 300px;
  aspect-ratio: 1 / 1;
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #f0f0f0;
}
.image-container img {
  display: block;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}
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
  font-size: 1.2rem;
}
.right-side-container {
  display: flex;
  flex-direction: column;
  flex: 1 1 400px;
  min-width: 280px;
  gap: 20px;
}
.upload-section,
.controls-fieldset,
.results-section {
  background-color: #fff;
  border: 1px solid #e0e0e0; /* Softer border */
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
  padding: 20px; /* Consistent padding */
  height: auto; /* Allow height to be auto */
}

.upload-section {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); /* Responsive grid columns */
  gap: 1rem;
  min-height: auto; /* Remove fixed min-height if not needed */
}


.fieldset-legend,
.controls-legend,
.result-legend {
  font-weight: bold;
  padding: 0 8px;
  margin-left: 10px;
  font-size: 1.25rem;
}
.fieldset-legend, .controls-legend { color: #333; }
.result-legend { color: #28a745; margin-bottom: 15px; }
.upload-section > label,
.upload-section > .upload-button,
.upload-section > .clear-button,
.upload-section > .save-location-button,
.upload-model > .model-select-button {
  padding: 0.65rem 0.8rem;
  border: 1px solid #bcccdc;
  border-radius: 6px;
  text-align: center;
  background-color: #76a0af;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  box-sizing: border-box;
  font-size: 0.95rem;
  transition: background-color 0.2s ease, box-shadow 0.2s ease;
  line-height: 1.4;
}
.upload-section > label:hover,
.upload-section > .upload-button:not(:disabled):hover,
.upload-section > .clear-button:hover,
.upload-section > .save-location-button:hover,
.upload-model {
  position: relative; /* Necessary for the absolute positioning of the dropdown list */
  /* This div is a grid item, so its width is determined by the grid.
     It doesn't need its own padding or background if the button inside handles that. */
}

/* Styles for the model select button itself */
.upload-model > .model-select-button {
  /* Inherit or define common button styles */
  padding: 0.65rem 0.8rem;
  border: 1px solid #bcccdc; /* Default border, can be overridden by specific button styles */
  border-radius: 6px;
  background-color: #608a99; /* Your specific background for this button */
  color: white;
  cursor: pointer;
  width: 100%;
  box-sizing: border-box;
  font-size: 0.95rem;
  transition: background-color 0.2s ease, box-shadow 0.2s ease;
  line-height: 1.4;

  /* --- Key Flexbox properties for centering text --- */
  display: flex;
  align-items: center;
  /* Remove justify-content: center; or space-between; from here if you want text to truly center */
}

.upload-model > .model-select-button:hover {
  background-color: #0056b3; /* Your specified hover background */
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Style for the brain icon (left icon) */
.upload-model > .model-select-button > i:first-child {
  margin-right: 0.5em; /* Space between brain icon and text */
  flex-shrink: 0;     /* Prevent this icon from shrinking if space is tight */
}

/* Style for the span containing "选择模型" text */
.upload-model > .model-select-button > span {
  flex-grow: 1;       /* Allow the text span to take up available flexible space */
  text-align: center; /* Center the text within the span */
  /* If you want to ensure it doesn't get pushed by margins of icons: */
  /* margin-left: auto; */
  /* margin-right: auto; */ /* This might work if icons are fixed width or also flex items */
}

/* Style for the dropdown arrow icon (right icon) */
.upload-model > .model-select-button > .dropdown-arrow {
  margin-left: 0.5em;  /* Space between text and arrow icon */
  font-size: 0.8em;
  transition: transform 0.2s ease-in-out;
  flex-shrink: 0;      /* Prevent this icon from shrinking */
}

.upload-model > .model-select-button > .dropdown-arrow.fa-chevron-up {
  transform: rotate(180deg);
}

/* Dropdown list styles (should remain as they were) */
.model-dropdown-list {
  position: absolute;
  top: calc(100% + 2px);
  left: 0;
  right: 0;
  background-color: #fff;
  border: 1px solid #ccc;
  border-top: none;
  border-radius: 0 0 6px 6px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  list-style: none;
  padding: 0;
  margin: 0;
  z-index: 1000;
  max-height: 200px;
  overflow-y: auto;
}

.model-dropdown-list li {
  padding: 10px 15px;
  cursor: pointer;
  font-size: 0.9rem;
  color: #333;
  text-align: left;
}

.model-dropdown-list li:not(:last-child) {
  border-bottom: 1px solid #eee;
}

.model-dropdown-list li:hover {
  background-color: #f0f0f0;
}

.model-dropdown-list li.selected {
  background-color: #608a99; /* Match button background or a distinct selection color */
  color: white;
  font-weight: bold;
}
.upload-section > .upload-button:disabled {
  background-color: #c0c0c0;
  cursor: not-allowed;
  opacity: 0.7;
}
.upload-label,
.upload-vedio,
.upload-stream {
  background-color: #007bff;
}
.upload-label:hover,
.upload-vedio:hover,
.upload-stream:hover {
  background-color: #0056b3;
}
.upload-button {
  background-color: #28a745;
  border-color: #1e7e34;
}
.upload-button:hover:not(:disabled) {
  background-color: #1e7e34;
}
.clear-button {
  background-color: #6c757d;
  border-color: #5a6268;
}
.clear-button:hover {
  background-color: #5a6268;
}

.save-location-button {
  background-color: #17a2b8;
  border-color: #117a8b;
}
.save-location-button:hover {
  background-color: #117a8b;
}
.upload-section > label i,
.upload-section > .upload-button i,
.upload-section > .clear-button i,
.upload-section > .save-location-button i,

.upload-section > .save-location-button,
.upload-section > .error-message {
  grid-column: 1 / -1;
}

.model-dropdown-list {
  position: absolute;
  top: calc(100% + 2px);
  left: 0;
  right: 0;
  background-color: #fff;
  border: 1px solid #ccc;
  border-top: none;
  border-radius: 0 0 6px 6px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  list-style: none;
  padding: 0;
  margin: 0;
  z-index: 1000;
  max-height: 200px;
  overflow-y: auto;
}
.model-dropdown-list li {
  padding: 10px 15px;
  cursor: pointer;
  font-size: 0.9rem;
  color: #333;
}
.model-dropdown-list li:not(:last-child) {
  border-bottom: 1px solid #eee;
}
.model-dropdown-list li:hover {
  background-color: #f0f0f0;
}
.model-dropdown-list li.selected {
  background-color: #007bff;
  color: white;
  font-weight: bold;
}
.controls-fieldset {
  margin-top: 0;
}
.confidence-control,
.iou-control {
  background-color: #f9f9f9;
  padding: 15px;
  border-radius: 6px;
  margin-bottom: 15px;
}
.confidence-control:last-child,
.iou-control:last-child {
  margin-bottom: 0;
}
.confidence-control label,
.iou-control label {
  display: flex;
  align-items: center;
  font-size: 1rem;
  color: #34495e;
  margin-bottom: 10px;
  font-weight: 500;
}
.confidence-control label i,
.iou-control label i {
  margin-right: 10px;
  color: #3498db;
  font-size: 1.1rem;
  min-width: 20px;
  text-align: center;
}
.slider {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 10px;
  background: #e0e6ed;
  border-radius: 5px;
  border: 1px solid #d4dbe0;
  cursor: pointer;
  outline: none;
}
.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 22px;
  height: 22px;
  background: #3498db;
  border-radius: 50%;
  border: 3px solid #ffffff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
  margin-top: -7px;
}
.slider::-moz-range-thumb {
  width: 22px;
  height: 22px;
  background: #3498db;
  border-radius: 50%;
  border: 3px solid #ffffff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
.slider:focus::-webkit-slider-thumb {
  box-shadow: 0 0 0 4px rgba(52, 152, 219, 0.3);
}
.slider:focus::-moz-range-thumb {
  box-shadow: 0 0 0 4px rgba(52, 152, 219, 0.3);
}
.slider::-webkit-slider-thumb:hover { background: #2980b9; }
.slider::-moz-range-thumb:hover { background: #2980b9; }
.results-section {
  margin-top: 0;
}
.results-summary {
  margin-bottom: 15px;
  padding: 12px 15px;
  background-color: #e9f5e9;
  border-left: 4px solid #28a745;
  border-radius: 4px;
}
.results-summary p {
  margin: 6px 0;
  color: #333;
  font-size: 0.95rem;
}
.results-summary p i {
  margin-right: 8px;
  color: #28a745;
}
.table-responsive {
  overflow-x: auto;
  width: 100%;
  border: 1px solid #ddd;
  border-radius: 6px;
}
.detection-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}
.detection-table th,
.detection-table td {
  border: none;
  border-bottom: 1px solid #ddd;
  padding: 10px 12px;
  text-align: left;
  vertical-align: middle;
}
.detection-table th {
  background-color: #f2f2f2;
  font-weight: 600;
  color: #333;
  border-bottom-width: 2px;
}
.detection-table th i {
  margin-right: 6px;
  color: #555;
}
.detection-table tbody tr:nth-child(even) {
  background-color: #f9f9f9;
}
.detection-table tbody tr:hover {
  background-color: #eef6ee;
}
.detection-table td:first-child {
  font-weight: bold;
  color: #007bff;
}
.no-results-message {
  padding: 20px;
  text-align: center;
  color: #555;
  background-color: #f9f9f9;
  border-radius: 6px;
  margin-top: 10px;
  border: 1px dashed #ddd;
}
.no-results-message i {
  margin-right: 8px;
  color: #777;
}
.error-message {
  color: #721c24;
  background-color: #f8d7da;
  border: 1px solid #f5c6cb;
  border-radius: 6px;
  padding: 0.75rem 1rem;
  margin-top: 1rem;
  text-align: center;
  width: 100%;
  box-sizing: border-box;
  font-size: 0.95rem;
}
</style>