import sys
import cv2
import numpy as np
import pandas as pd
import json
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

        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.rect.isNull():
            painter = QPainter(self)
            pen = QPen(Qt.green, 2)
            painter.setPen(pen)
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

    def set_roi(self, roi):
        """Set ROI rectangle from tuple (x, y, w, h)"""
        x, y, w, h = roi
        self.rect = QRect(x, y, w, h)
        self.update()

class HeartRateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Larvae Heart Rate Tracker - Manual ROI & Export")
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

        self.btn_export_data = QPushButton("Export Data CSV")
        self.btn_export_data.clicked.connect(self.export_csv)
        self.btn_export_data.setEnabled(False)
        controls.addWidget(self.btn_export_data)

        self.btn_save_plot = QPushButton("Save Plot Image")
        self.btn_save_plot.clicked.connect(self.save_plot_image)
        self.btn_save_plot.setEnabled(False)
        controls.addWidget(self.btn_save_plot)

        self.btn_export_summary = QPushButton("Export Summary CSV")
        self.btn_export_summary.clicked.connect(self.export_summary_csv)
        self.btn_export_summary.setEnabled(False)
        controls.addWidget(self.btn_export_summary)

        # New buttons to save/load ROI
        self.btn_save_roi = QPushButton("Save ROI")
        self.btn_save_roi.clicked.connect(self.save_roi)
        self.btn_save_roi.setEnabled(False)
        controls.addWidget(self.btn_save_roi)

        self.btn_load_roi = QPushButton("Load ROI")
        self.btn_load_roi.clicked.connect(self.load_roi)
        self.btn_load_roi.setEnabled(False)
        controls.addWidget(self.btn_load_roi)

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
        self.btn_export_data.setEnabled(False)
        self.btn_save_plot.setEnabled(False)
        self.btn_export_summary.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_save_roi.setEnabled(True)
        self.btn_load_roi.setEnabled(True)

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
        self.btn_export_data.setEnabled(False)
        self.btn_save_plot.setEnabled(False)
        self.btn_export_summary.setEnabled(False)

    def stop_tracking(self):
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export_data.setEnabled(True)
        self.btn_save_plot.setEnabled(True)
        self.btn_export_summary.setEnabled(True)
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

        if len(self.intensity_signal) >= int(self.fps * 2):  # after 2 seconds buffer
            data = np.array(self.intensity_signal)
            # Adjusted bandpass to match original script 1-4 Hz
            self.filtered_signal = bandpass_filter(data, 1, 4, self.fps, order=3)
            self.peaks, _ = find_peaks(self.filtered_signal,
                                       distance=int(self.fps * 0.3),
                                       prominence=0.5)

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
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Data CSV", "heart_rate_data.csv", "CSV Files (*.csv)")
        if save_path:
            df.to_csv(save_path, index=False)
            QMessageBox.information(self, "Saved", f"Data CSV saved as {save_path}")

    def save_plot_image(self):
        if self.filtered_signal is None or len(self.peaks) == 0:
            QMessageBox.warning(self, "No Data", "No data to save plot image.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Plot Image", "plot.png", "PNG Image (*.png)")
        if filename:
            self.plot_canvas.fig.savefig(filename)
            QMessageBox.information(self, "Saved", f"Plot saved as {filename}")

    def export_summary_csv(self):
        if self.filtered_signal is None or len(self.peaks) == 0:
            QMessageBox.warning(self, "No Data", "No data to export.")
            return
        bpm = len(self.peaks) / (len(self.filtered_signal) / self.fps) * 60
        intervals = np.diff(self.peaks) / self.fps  # seconds between beats
        mean_interval = np.mean(intervals) if len(intervals) > 0 else 0
        std_interval = np.std(intervals) if len(intervals) > 0 else 0
        df = pd.DataFrame({
            'Metric': ['Estimated BPM', 'Mean Interval (s)', 'Std Interval (s)'],
            'Value': [bpm, mean_interval, std_interval]
        })
        filename, _ = QFileDialog.getSaveFileName(self, "Save Summary CSV", "summary.csv", "CSV Files (*.csv)")
        if filename:
            df.to_csv(filename, index=False)
            QMessageBox.information(self, "Saved", f"Summary CSV saved as {filename}")

    def save_roi(self):
        roi = self.video_label.get_roi()
        if roi is None:
            QMessageBox.warning(self, "No ROI", "Draw ROI before saving.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save ROI", "roi.json", "JSON Files (*.json)")
        if not filename:
            return
        try:
            with open(filename, 'w') as f:
                json.dump({'roi': roi}, f)
            QMessageBox.information(self, "Saved", f"ROI saved as {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save ROI: {e}")

    def load_roi(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load ROI", "", "JSON Files (*.json)")
        if not filename:
            return
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            roi = tuple(data['roi'])
            self.video_label.set_roi(roi)
            QMessageBox.information(self, "Loaded", f"ROI loaded from {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load ROI: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeartRateApp()
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec())
