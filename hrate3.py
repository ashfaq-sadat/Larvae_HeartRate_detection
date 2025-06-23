import sys
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        fig.tight_layout()

    def plot(self, raw_signal, filtered_signal, peaks):
        self.ax.clear()
        self.ax.plot(raw_signal, label='Raw Intensity')
        if filtered_signal is not None:
            self.ax.plot(filtered_signal, label='Filtered Intensity')
            self.ax.plot(peaks, filtered_signal[peaks], 'rx', label='Detected Beats')
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Intensity')
        self.ax.legend()
        self.draw()

class VideoLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.rect = QRect()
        self.frame = None
        self.scaled_pixmap = None
        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0

    def set_frame(self, frame):
        self.frame = frame
        self.update_display()

    def update_display(self):
        if self.frame is None:
            return

        disp_frame = self.frame.copy()
        img = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
        h_img, w_img, ch = img.shape
        bytes_per_line = ch * w_img
        qt_img = QImage(img.data, w_img, h_img, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.scaled_pixmap = pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
        self.setPixmap(self.scaled_pixmap)

        # Update scale and offsets
        self.scale = self.scaled_pixmap.width() / self.frame.shape[1]
        self.offset_x = (self.width() - self.scaled_pixmap.width()) / 2
        self.offset_y = (self.height() - self.scaled_pixmap.height()) / 2

        # Draw ROI rectangle overlay on widget (in paintEvent)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.rect.isNull():
            painter = QPainter(self)
            pen = QPen(Qt.green, 2)
            painter.setPen(pen)
            # Convert ROI rect image coords -> widget coords
            x_img, y_img, w_img, h_img = self.rect.left(), self.rect.top(), self.rect.width(), self.rect.height()
            x = int(x_img * self.scale + self.offset_x)
            y = int(y_img * self.scale + self.offset_y)
            w = int(w_img * self.scale)
            h = int(h_img * self.scale)
            painter.drawRect(x, y, w, h)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = self.widget_to_image_coords(event.pos())
            self.end_point = self.start_point
            self.rect = QRect(self.start_point, self.end_point)
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = self.widget_to_image_coords(event.pos())
            self.rect = QRect(self.start_point, self.end_point).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.end_point = self.widget_to_image_coords(event.pos())
            self.rect = QRect(self.start_point, self.end_point).normalized()
            self.update()

    def widget_to_image_coords(self, pos):
        x = pos.x()
        y = pos.y()
        x -= self.offset_x
        y -= self.offset_y
        x_img = int(x / self.scale)
        y_img = int(y / self.scale)
        x_img = max(0, min(self.frame.shape[1] - 1, x_img))
        y_img = max(0, min(self.frame.shape[0] - 1, y_img))
        return QPoint(x_img, y_img)

    def get_roi(self):
        if self.rect.isNull():
            return None
        return (self.rect.left(), self.rect.top(), self.rect.width(), self.rect.height())

class HeartRateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Larvae Heart Rate Tracker - Manual ROI")
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.fps = 30
        self.intensity_signal = []
        self.filtered_signal = None
        self.peaks = []
        self.frame_index = 0
        self.roi = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.video_label = VideoLabel()
        self.video_label.setFixedSize(640, 480)
        layout.addWidget(self.video_label)

        controls = QHBoxLayout()

        self.btn_load = QPushButton("Load Video")
        self.btn_load.clicked.connect(self.load_video)
        controls.addWidget(self.btn_load)

        self.btn_start = QPushButton("Start Tracking")
        self.btn_start.clicked.connect(self.start_tracking)
        self.btn_start.setEnabled(False)
        controls.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop Tracking")
        self.btn_stop.clicked.connect(self.stop_tracking)
        self.btn_stop.setEnabled(False)
        controls.addWidget(self.btn_stop)

        self.btn_export = QPushButton("Export CSV")
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_export.setEnabled(False)
        controls.addWidget(self.btn_export)

        layout.addLayout(controls)

        self.plot_canvas = PlotCanvas(self, width=6, height=3)
        layout.addWidget(self.plot_canvas)

        self.setLayout(layout)

    def load_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not fname:
            return
        self.cap = cv2.VideoCapture(fname)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video.")
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30  # fallback
        print(f"Video FPS: {self.fps:.2f}")

        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.critical(self, "Error", "Could not read video.")
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.intensity_signal = []
        self.filtered_signal = None
        self.peaks = []
        self.frame_index = 0
        self.roi = None
        self.video_label.set_frame(frame)
        self.btn_start.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.btn_stop.setEnabled(False)

    def start_tracking(self):
        roi = self.video_label.get_roi()
        if roi is None:
            QMessageBox.warning(self, "ROI not set", "Please draw a ROI rectangle on the video before starting.")
            return
        self.roi = roi
        self.intensity_signal = []
        self.filtered_signal = None
        self.peaks = []
        self.frame_index = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.timer.start(int(1000 / self.fps))
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)

    def stop_tracking(self):
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(True)
        self.show_results()

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_tracking()
            QMessageBox.information(self, "End of Video", "Video has ended.")
            return

        x, y, w, h = self.roi
        roi_frame = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        self.intensity_signal.append(mean_intensity)

        if len(self.intensity_signal) >= int(self.fps * 2):  # after 2 seconds
            data = np.array(self.intensity_signal)
            # Bandpass filter tuned for larvae heart rate ~0.5-6Hz
            self.filtered_signal = bandpass_filter(data, 0.5, 6, self.fps, order=3)
            self.peaks, _ = find_peaks(self.filtered_signal, distance=int(self.fps * 0.2), prominence=0.3)

        disp_frame = frame.copy()
        color = (0, 255, 0)
        cv2.rectangle(disp_frame, (x, y), (x+w, y+h), color, 2)

        self.video_label.set_frame(disp_frame)
        self.frame_index += 1

    def show_results(self):
        if self.filtered_signal is None or len(self.peaks) == 0:
            QMessageBox.warning(self, "No beats detected", "No heart beats detected in ROI.")
            return

        bpm = len(self.peaks) / (len(self.filtered_signal) / self.fps) * 60
        QMessageBox.information(self, "Heart Rate", f"Estimated Heart Rate: {bpm:.2f} BPM")
        self.plot_canvas.plot(self.intensity_signal, self.filtered_signal, self.peaks)

    def export_csv(self):
        if self.filtered_signal is None or len(self.peaks) == 0:
            QMessageBox.warning(self, "No data", "No heart rate data to export.")
            return

        data = {
            'Frame': np.arange(len(self.filtered_signal)),
            'Filtered_Intensity': self.filtered_signal,
            'Peak': [1 if i in self.peaks else 0 for i in range(len(self.filtered_signal))]
        }
        df = pd.DataFrame(data)
        save_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "heart_rate.csv", "CSV Files (*.csv)")
        if save_path:
            df.to_csv(save_path, index=False)
            QMessageBox.information(self, "Export Success", f"Data saved to {save_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeartRateApp()
    window.resize(700, 700)
    window.show()
    sys.exit(app.exec())
