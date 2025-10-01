from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QStackedLayout,QLabel, QProgressBar
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt,QThread, pyqtSignal
from PyQt5.QtGui import QFont

import sys
import os

class DataProcessingThread(QThread):
    finished = pyqtSignal()

    def run(self):
        # 在这里处理你的数据，比如读取、对齐、计算等
        import time
        time.sleep(5)  # 模拟耗时操作
        # 数据处理完成后发出信号
        self.finished.emit()
class EvalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Streamflow Evaluation Tool")
        self.setFixedSize(1600, 800)
        self.model_files = []
        self.standard_folder = []
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
                QWidget {
                    background-color: #f5f7fa;
                    font-family: 'Segoe UI', 'Fira Sans', 'Helvetica Neue', sans-serif;
                    font-size: 11pt;
                    color: #333;
                }

                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px;
                    border-radius: 8px;
                    font-weight: bold;
                }

                QPushButton:hover {
                    background-color: #45a049;
                }

                QLabel#titleLabel {
                    font-size: 16pt;
                    font-weight: bold;
                    color: #2c3e50;
                }

                QProgressBar {
                    border: 2px solid #bbb;
                    border-radius: 10px;
                    text-align: center;
                    background-color: #e0e0e0;
                    min-height: 25px;
                }

                QProgressBar::chunk {
                    background-color: qlineargradient(
                        spread:pad, x1:0, y1:0, x2:1, y2:1,
                        stop:0 #4CAF50, stop:1 #81C784
                    );
                    border-radius: 10px;
                    margin: 0.5px;
                }
            """)
        self.btn_default_style = """
            QPushButton {
                background-color: #e0f7e9;
                color: #2e7d32;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c8e6c9;
            }
        """

        self.btn_selected_style = """
            QPushButton {
                background-color: #2e7d32;
                color: white;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            }
        """

        self.stack = QStackedLayout()

        # Page 1 - Main interface
        self.page1 = QWidget()
        main_layout = QHBoxLayout()
        # ==== Left side (Buttons) ====
        left_layout = QVBoxLayout()
        self.model_btn = QPushButton('Select Model Simulation Data')
        self.model_btn.clicked.connect(self.choose_model_file)

        self.standard_btn = QPushButton('Select Reference Observation Data')
        self.standard_btn.clicked.connect(self.choose_standard_file)

        self.eval_btn = QPushButton('Start Evaluation')
        self.eval_btn.setEnabled(False)
        self.eval_btn.clicked.connect(self.start_evaluation)
        self.model_btn.setStyleSheet(self.btn_default_style)
        self.standard_btn.setStyleSheet(self.btn_default_style)
        self.eval_btn.setStyleSheet(self.btn_default_style)

        left_layout.addStretch()
        left_layout.addWidget(self.model_btn)
        left_layout.addWidget(self.standard_btn)
        left_layout.addWidget(self.eval_btn)
        left_layout.addStretch()

        # ==== Right side (Image + Footer) ====
        right_layout = QVBoxLayout()

        # Image on top
        self.image_label = QLabel()
        pixmap = QPixmap("./steamflowbg.jpg")
        pixmap = pixmap.scaledToWidth(800, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Footer info
        self.footer_label = QLabel(
            "Created on April 24, 08:42 2025\n"
            "@Author: Qingliang Li: liqingliang@ccsfu.edu.cn (Email)\n"
            "@Co-author1: Cheng Zhang\n"
            "@Co-author2: Kaixuan Cai\n"
            "@Co-author3: Zhongwang Wei\n"
            "@Co-author4: Zenghui Liu\n"
        )
        self.footer_label.setAlignment(Qt.AlignCenter)

        right_layout.addWidget(self.image_label)
        right_layout.addWidget(self.footer_label)

        # Add to main layout
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

        self.page1.setLayout(main_layout)

        # Page 2 - Evaluation screen
        self.page2 = QWidget()
        layout2 = QVBoxLayout()
        layout2.setAlignment(Qt.AlignCenter)  # 整体居中布局

        # 美观字体设置
        self.label = QLabel("Preprocessing observational data to match simulation settings. Please wait...")
        font = QFont("Segoe UI", 12)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)  # 文本居中对齐
        self.label.setWordWrap(True)
        self.label.setStyleSheet("""
            QLabel {
                color: #2E2E2E;
                padding: 10px;
            }
        """)

        # 优化后的进度条样式
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedSize(600, 30)  # 更合理的尺寸
        self.progress_bar.setRange(0, 0)  # 无限滚动
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #888;
                border-radius: 10px;
                text-align: center;
                background-color: #f0f0f0;
            }

            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 10px;
                width: 20px;
                margin: 0.5px;
            }
        """)

        # 添加控件
        layout2.addWidget(self.label)
        layout2.addWidget(self.progress_bar)

        # 设置页面布局
        self.page2.setLayout(layout2)

        # 添加页面到堆栈中
        self.stack.addWidget(self.page1)
        self.stack.addWidget(self.page2)

        # 应用堆栈布局
        self.setLayout(self.stack)





    def choose_model_file(self):
        fnames, _ = QFileDialog.getOpenFileNames(
            self, 'Select Model Simulation Data Files', '', 'NetCDF files (*.nc)'
        )
        if fnames:
            self.model_files = fnames
            if len(fnames) == 1:
                name_display = fnames[0].split('/')[-1]
            else:
                name_display = f"{len(fnames)} files selected"
            self.model_btn.setText(f"Model: {name_display}")
            self.update_selection_status()

    def choose_standard_file(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Select Reference Observation Data Folder', ''
        )
        if folder:
            self.standard_folder = folder
            name_display = os.path.basename(folder)
            self.standard_btn.setText(f"Reference Folder: {name_display}")
            self.update_selection_status()

    def update_selection_status(self):
        if self.model_files and self.standard_folder:
            self.eval_btn.setEnabled(True)
            self.eval_btn.setText("File selection completed, please start evaluation")

    def start_evaluation(self):
        self.stack.setCurrentWidget(self.page2)

        # 创建线程实例
        self.processing_thread = DataProcessingThread()
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def on_processing_finished(self):
        from PyQt5.QtWidgets import QMessageBox

        msg = QMessageBox(self)
        msg.setWindowTitle("✨ Processing Complete")
        msg.setText("✅ Data processing has finished successfully.\n\nYou may now return to the main screen to proceed.")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)

        # 设置艺术气息的字体与配色
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #f5f5f5;
                font-family: 'Segoe UI';
                font-size: 12pt;
                color: #333333;
            }
            QPushButton {
                min-width: 80px;
                padding: 6px;
                border-radius: 6px;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
        """)

        msg.exec_()

        # 返回主页面.......................................................................需要新加界面, 做评估图
        self.stack.setCurrentWidget(self.page1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = EvalApp()
    win.show()
    sys.exit(app.exec_())
