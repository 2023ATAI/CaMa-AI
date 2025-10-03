import sys
import os
import time
import pandas as pd
import yaml
import numpy as np
import xarray as xr
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QStackedLayout, QComboBox, QMessageBox, QListWidget)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from partd.file import filename

from process_simulations import process_simulations, plot_timeseries, plot_spatial_map, plot_river_network_spatial, plot_water_level_spatial, \
    plot_flooded_area_timeseries, plot_flood_depth_timeseries  # 显式导入

class DataProcessingThread(QThread):
    finished = pyqtSignal()

    def run(self):
        time.sleep(5)  # 模拟处理时间，可替换为实际处理逻辑
        self.finished.emit()

class EvalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Streamflow Evaluation Tool")
        self.setFixedSize(1600, 800)
        self.model_files = []
        self.standard_folder = []
        self.sim_dirs = []
        self.variables = ["Discharge"]
        self.stack = QStackedLayout()
        self.init_main_ui()
        self.init_eval_ui()
        self.init_processing_ui()
        self.setLayout(self.stack)

    def init_main_ui(self):
        self.page1 = QWidget()
        main_layout = QHBoxLayout()
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f5f3;
                font-family: "Segoe UI", "Helvetica Neue", sans-serif;
                font-size: 11pt;
                color: #2c3e50;
            }
            QComboBox, QListWidget {
                background-color: #ffffff;
                border: 1.5px solid #4CAF50;
                border-radius: 8px;
                padding: 6px 10px;
                min-width: 160px;
                color: #2e7d32;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            QComboBox:hover, QListWidget:hover {
                border: 1.5px solid #388e3c;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left: 1px solid #4CAF50;
                background-color: #4CAF50;
                border-top-right-radius: 7px;
                border-bottom-right-radius: 7px;
            }
            QComboBox::down-arrow {
                image: url(:/qt-project.org/styles/commonstyle/images/arrow-down.png);
                width: 14px;
                height: 14px;
            }
            QComboBox QAbstractItemView, QListWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f5faff);
                selection-background-color: #81c784;
                selection-color: #ffffff;
                border: 1px solid #4CAF50;
                outline: none;
                font-size: 11pt;
                color: #2e7d32;
            }
            QComboBox QAbstractItemView::item:selected, QListWidget::item:selected {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #66bb6a, stop:1 #4CAF50);
                color: #ffffff;
                border-radius: 4px;
                padding: 2px;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #e0e0e0;
            }
            QListWidget::item:hover {
                background-color: #e8f5e9;
                border-radius: 4px;
            }
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #66bb6a, stop:1 #4CAF50
                );
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #388e3c
                );
            }
            QPushButton:pressed {
                background: #388e3c;
            }
            QLabel {
                color: #2c3e50;
                font-weight: 500;
            }
            QLabel#resultLabel {
                background-color: #ffffff;
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        left_layout = QVBoxLayout()
        self.model_btn = QPushButton('Select Config File')
        self.model_btn.clicked.connect(self.choose_model_file)

        self.standard_btn = QPushButton('Select Output Folder')
        self.standard_btn.clicked.connect(self.choose_standard_file)

        self.preprocess_btn = QPushButton('Preprocess')
        self.preprocess_btn.clicked.connect(self.start_preprocessing)
        self.preprocess_btn.setEnabled(False)

        self.eval_btn = QPushButton('Go Evaluation')
        self.eval_btn.setEnabled(False)
        self.eval_btn.clicked.connect(self.go_evaluation)

        left_layout.addStretch()
        left_layout.addWidget(self.model_btn)
        left_layout.addWidget(self.standard_btn)
        left_layout.addWidget(self.preprocess_btn)
        left_layout.addWidget(self.eval_btn)
        left_layout.addStretch()

        right_layout = QVBoxLayout()
        self.image_label = QLabel()
        pixmap = QPixmap("./steamflowbg.jpg")
        pixmap = pixmap.scaledToWidth(800, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.footer_label = QLabel(
            "Created on April 24, 08:42 2025\n"
            "@Author: Qingliang Li: liqingliang@ccsfu.edu.cn (Email)\n"
            "@Co-author1: Cheng Zhang\n"
            "@Co-author2: Kaixuan Cai\n"
            "@Co-author3: Zhongwang Wei\n"
        )
        self.footer_label.setAlignment(Qt.AlignCenter)

        right_layout.addWidget(self.image_label)
        right_layout.addWidget(self.footer_label)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)
        self.page1.setLayout(main_layout)
        self.stack.addWidget(self.page1)

    def init_eval_ui(self):
        self.page2 = QWidget()
        layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        self.var_label = QLabel("Select Variable:")
        self.var_combo = QComboBox()
        self.var_combo.addItems(self.variables)
        self.var_combo.currentTextChanged.connect(self.update_options)

        self.plot_type_label = QLabel("Select Plot Type:")
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Time Series", "Spatial Map"])
        self.plot_type_combo.currentTextChanged.connect(self.update_sub_options)

        self.sub_option_label = QLabel("Select Comparison:")
        self.sub_option_combo = QComboBox()

        self.sim_data_label = QLabel("Select Sims:")
        self.sim_data_list = QListWidget()
        self.sim_data_list.setObjectName("sim_data_list")
        self.sim_data_list.setSizeAdjustPolicy(QListWidget.AdjustToContents)
        self.sim_data_list.setSelectionMode(QListWidget.MultiSelection)
        self.sim_data_list.setMinimumHeight(20)
        self.sim_data_list.setMaximumHeight(80)
        self.sim_data_combo = QComboBox()

        self.indicator_label = QLabel("Select Indicator:")
        self.indicator_combo = QComboBox()

        self.station_label = QLabel("Select Station:")
        self.station_combo = QComboBox()

        self.generate_btn = QPushButton("Generate Evaluation")
        self.generate_btn.clicked.connect(self.generate_evaluation)

        self.back_btn = QPushButton("Back to Main")
        self.back_btn.clicked.connect(self.return_to_main)

        left_layout.addStretch()
        left_layout.addWidget(self.var_label)
        left_layout.addWidget(self.var_combo)
        left_layout.addWidget(self.plot_type_label)
        self.plot_type_combo.setMinimumWidth(120)
        left_layout.addWidget(self.plot_type_combo)
        left_layout.addWidget(self.sub_option_label)
        left_layout.addWidget(self.sub_option_combo)
        left_layout.addWidget(self.sim_data_label)
        left_layout.addWidget(self.sim_data_list)
        left_layout.addWidget(self.sim_data_combo)
        left_layout.addWidget(self.indicator_label)
        left_layout.addWidget(self.indicator_combo)
        left_layout.addWidget(self.station_label)
        left_layout.addWidget(self.station_combo)
        left_layout.addWidget(self.generate_btn)
        left_layout.addWidget(self.back_btn)
        left_layout.addStretch()

        right_layout = QVBoxLayout()
        self.result_label = QLabel("Evaluation Result")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setObjectName("resultLabel")
        self.result_label.setPixmap(
            QPixmap("./steamflowbg.jpg").scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        right_layout.addWidget(self.result_label)

        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 2)
        self.page2.setLayout(layout)
        self.stack.addWidget(self.page2)

        self.var_combo.setCurrentText("Discharge")
        self.plot_type_combo.setCurrentText("Time Series")
        self.update_options("Discharge")
        self.update_sub_options("Time Series")

    def init_processing_ui(self):
        self.processing_page = QWidget()
        layout = QVBoxLayout()
        self.processing_label = QLabel("Processing, please wait...")
        self.processing_label.setAlignment(Qt.AlignCenter)
        self.processing_label.setStyleSheet("font-size: 14pt; color: #2c3e50; font-weight: bold;")
        layout.addStretch()
        layout.addWidget(self.processing_label)
        layout.addStretch()
        self.processing_page.setLayout(layout)
        self.stack.addWidget(self.processing_page)

    def choose_model_file(self):
        self.choosen_var=None
        print('choose_model_file')
        fnames, _ = QFileDialog.getOpenFileNames(self, 'Select Config File', '', 'YAML Files (*.yml *.yaml)')
        if fnames:
            self.model_files = fnames
            self.model_btn.setText(f"Selected {len(fnames)} file(s)")
            self.sim_dirs = []
            self.variables = []
            for fname in fnames:
                try:
                    with open(fname, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        for sim_key in ['SIMDIR']:
                            sim_dir = config.get(sim_key)
                            if isinstance(sim_dir, list):
                                self.sim_dirs.extend([os.path.basename(os.path.normpath(path)) for path in sim_dir if path])
                            elif isinstance(sim_dir, str):
                                self.sim_dirs.append(os.path.basename(os.path.normpath(sim_dir)))
                        variables = config.get('Variables', [])
                        if isinstance(variables, list):
                            self.variables.extend([var for var in variables if var and isinstance(var, str)])
                except Exception as e:
                    print(f"Failed to read {fname}: {str(e)}")
                    QMessageBox.warning(self, "Error", f"Could not read SIMDIR or Variables from {fname}: {str(e)}")
            if not self.sim_dirs:
                self.sim_dirs = ["No Model Data"]
            if not self.variables:
                self.variables = ["Discharge"]
            print(f"Loaded sim_dirs: {self.sim_dirs}, variables: {self.variables}")
            self.var_combo.clear()
            self.var_combo.addItems(self.variables)
            self.var_combo.setCurrentText(
                "Discharge" if "Discharge" in self.variables else self.variables[0] if self.variables else "Discharge")
            self.update_selection_status()
            self.update_sub_options(self.plot_type_combo.currentText())


            self.var_combo.currentTextChanged.connect(self.on_combobox_changed)
            # process_simulations(config, self.standard_folder, self.choosen_var)  # 实际调用预处理

            # if self.var_combo:
            #     for i in range(self.var_combo.count()):
            #         item_text = self.var_combo.item(i).text()
            #         if item_text in ['Discharge', 'maximum flood deepth', 'flooded area', 'river network', 'water level']:
            #             self.var_combo.item(i).setSelected(True)
            # print(self.var_combo.selectedItems())
    def on_combobox_changed(self, text):
        print("当前选中：", text)  # 控制台打印出选中的值
        self.choosen_var=text
        print(' self.choosen_var', self.choosen_var)

    def choose_standard_file(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder', '')
        if folder:
            self.standard_folder = folder
            self.standard_btn.setText(f"Folder: {os.path.basename(folder)}")
            self.update_selection_status()

    def update_selection_status(self):
        if self.model_files and self.standard_folder:
            self.eval_btn.setEnabled(True)
            self.preprocess_btn.setEnabled(True)

    def start_preprocessing(self):
        self.stack.setCurrentWidget(self.processing_page)
        print('ee')
        try:
            with open(self.model_files[0], 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            choosen_var='flood'
            process_simulations(config, self.standard_folder,choosen_var)  # 实际调用预处理
            self.on_preprocessing_finished()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Preprocessing failed: {str(e)}")
            self.stack.setCurrentWidget(self.page1)

    def on_preprocessing_finished(self):
        QMessageBox.information(self, "Processing Complete", "Preprocessing is done. Click OK to proceed.",
                                QMessageBox.Ok)
        self.stack.setCurrentWidget(self.page2)

    def go_evaluation(self):
        print("evaluation")
        self.stack.setCurrentWidget(self.page2)
        self.processing_thread = DataProcessingThread()
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def on_processing_finished(self):
        QMessageBox.information(self, "Done", "Evaluation preparation complete.")

    def load_station_data(self):
        try:
            df = pd.read_csv("stn_list.txt")
            return df['ID'].astype(str).tolist()
        except:
            return ["No Station Data"]

    def update_options(self, var):
        if var in self.variables:
            # 特殊情况：river network 和 water level 只能画空间图
            if var in ["river network", "water level"]:
                self.plot_type_combo.setCurrentText("Spatial Map")
                self.plot_type_combo.setEnabled(False)  # 禁用切换
            else:
                self.plot_type_combo.setEnabled(True)

            self.plot_type_combo.show()
            self.sub_option_label.hide()
            self.sub_option_combo.hide()
            self.sim_data_list.clear()
            self.sim_data_combo.clear()
            self.indicator_label.hide()
            self.indicator_combo.hide()
            self.station_combo.hide()
            self.update_sub_options(self.plot_type_combo.currentText())

    def update_sub_options(self, plot_type):
        self.current_plot_type = plot_type
        self.sub_option_combo.clear()
        self.sim_data_list.clear()
        self.sim_data_combo.clear()
        self.indicator_combo.clear()
        self.station_combo.clear()

        try:
            self.sub_option_combo.currentTextChanged.disconnect(self.update_metrics)
        except:
            pass

        variable = self.var_combo.currentText()

        if plot_type == "Time Series":
            self.sub_option_label.setText("Select Comparison:")
            self.sub_option_label.show()

            # 只有 Discharge 才有 Residual
            if variable == "Discharge":
                self.sub_option_combo.addItems(["Numerical Comparison", "Residual Comparison"])
            else:
                self.sub_option_combo.addItems(["Numerical Comparison"])

            self.sub_option_combo.setCurrentIndex(0)
            self.sub_option_combo.show()

            self.sim_data_label.setText("Select Sims:")
            self.sim_data_list.addItems(self.sim_dirs)
            if self.sim_dirs:
                for i in range(self.sim_data_list.count()):
                    item_text = self.sim_data_list.item(i).text()
                    if item_text in ["Fortran", "Pytorch"]:
                        self.sim_data_list.item(i).setSelected(True)
            self.sim_data_list.show()
            self.sim_data_combo.hide()
            self.indicator_label.hide()
            self.indicator_combo.hide()

            # station 只对 Discharge 有效，其余变量隐藏
            if variable == "Discharge":
                stations = self.load_station_data()
                self.station_label.setText("Select Station:")
                self.station_combo.addItems(stations)
                if stations:
                    self.station_combo.setCurrentIndex(0)
                self.station_combo.show()
            else:
                self.station_label.hide()
                self.station_combo.hide()

        elif plot_type == "Spatial Map":
            self.sub_option_label.hide()
            self.sub_option_combo.hide()
            self.sim_data_label.setText("Select Sims:")
            self.sim_data_combo.addItems(self.sim_dirs)
            if self.sim_dirs:
                self.sim_data_combo.setCurrentIndex(0)
            self.sim_data_list.hide()
            self.sim_data_combo.show()

            # 只有 Discharge 才显示指标选择
            if variable == "Discharge":
                self.indicator_label.setText("Select Indicator:")
                self.indicator_combo.addItems(["RMSE", "KGESS", "BIAS", "CORRELATION"])
                if self.indicator_combo.count() > 0:
                    self.indicator_combo.setCurrentIndex(0)
                self.indicator_combo.show()
                self.indicator_label.show()
            else:
                self.indicator_label.hide()
                self.indicator_combo.hide()

            self.station_label.hide()
            self.station_combo.hide()

        self.generate_btn.setEnabled(True)

    def update_metrics(self, model):
        if self.current_plot_type == "Spatial Map":
            pass

    def generate_evaluation(self, sim_full_dir=None):
        print('generate_evaluation')

        variable = self.var_combo.currentText()

        # 时序图
        if self.current_plot_type == "Time Series":
            selected_sims = [item.text() for item in self.sim_data_list.selectedItems()]
            if not selected_sims or selected_sims == ["No Model Data"]:
                QMessageBox.warning(self, "Warning", "Please select at least one simulation dataset.")
                return

            # ---- Discharge ----
            if variable == "Discharge":
                case_plot = 'value' if self.sub_option_combo.currentText() == "Numerical Comparison" else 'residual'
                station_id = self.station_combo.currentText()
                if not station_id or station_id == "No Station Data":
                    QMessageBox.warning(self, "Warning", "Please select a valid station.")
                    return

                try:
                    with open(self.model_files[0], 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    output_folder = self.standard_folder
                    ref_dir = config.get('REFDIR', '') + 'GRDC_Day\\'
                    fig_dir = os.path.join(output_folder, 'figures')

                    plot_timeseries(
                        sim_dirs=selected_sims,
                        ref_dir=ref_dir,
                        fig_dir=fig_dir,
                        case_plot=case_plot,
                        station_ID=int(station_id),
                        OUTPUTFOLDER=output_folder,
                        STARTYEAR=config.get('STARTYEAR', 2000),
                        ENDYEAR=config.get('ENDYEAR', 2020)
                    )
                    plot_path = os.path.join(fig_dir, f'station_{station_id}_{case_plot}.png')

                    if plot_path and os.path.exists(plot_path):
                        pixmap = QPixmap(plot_path)
                        self.result_label.clear()
                        self.result_label.setPixmap(
                            pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        )
                        self.result_label.setAlignment(Qt.AlignCenter)
                    else:
                        self.result_label.setText("Failed to generate time series plot")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to generate discharge time series: {str(e)}")
                    self.result_label.setText("Failed to generate discharge time series")

            else:
                try:
                    with open(self.model_files[0], 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    output_folder = self.standard_folder
                    fig_dir = os.path.join(output_folder, "figures")

                    if variable == "flooded area":
                        from process_simulations import plot_fldaremap
                        plot_path = plot_fldaremap(
                            sim_full_dir=sim_full_dir,
                            fig_dir=fig_dir,
                            start_year=config.get('STARTYEAR', 2000),
                            end_year=config.get('ENDYEAR', 2020),
                            min_lon=config.get('MIN_LON', -125.0),
                            max_lon=config.get('MAX_LON', -66.0),
                            min_lat=config.get('MIN_LAT', 22.0),
                            max_lat=config.get('MAX_LAT', 54.0)
                        )

                    else:
                        plot_path = plot_flood_depth_timeseries(
                            sim_dirs=selected_sims,
                            fig_dir=fig_dir,
                            case_plot="value",
                            OUTPUTFOLDER=output_folder,
                            STARTYEAR=config.get("STARTYEAR", 2000),
                            ENDYEAR=config.get("ENDYEAR", 2020)
                        )

                    if plot_path and os.path.exists(plot_path):
                        pixmap = QPixmap(plot_path)
                        self.result_label.clear()
                        self.result_label.setPixmap(
                            pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        )
                        self.result_label.setAlignment(Qt.AlignCenter)
                    else:
                        self.result_label.setText("Failed to generate variable time series")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to generate variable time series: {str(e)}")
                    self.result_label.setText("Failed to generate variable time series")

        # 空间图
        elif self.current_plot_type == "Spatial Map":
            selected_sim = self.sim_data_combo.currentText()
            indicator = self.indicator_combo.currentText() if variable == "Discharge" else None
            print(
                f"Selected sim: {selected_sim}, Indicator: {indicator}, Combo visible: {self.indicator_combo.isVisible()}")

            if not selected_sim or selected_sim == "No Model Data":
                QMessageBox.warning(self, "Warning", "Please select a simulation dataset.")
                return

            try:
                with open(self.model_files[0], 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                output_folder = self.standard_folder
                fig_dir = os.path.join(output_folder, 'figures')

                if variable == "flooded area":
                    plot_path = os.path.join(fig_dir, f'flooded_area_spatial_{selected_sim}.png')
                    if not os.path.exists(plot_path):
                        years = list(range(config.get('STARTYEAR', 2000), config.get('ENDYEAR', 2020) + 1))
                        sim_full_dir = None
                        for sim_dir in config.get('SIMDIR', []):
                            if os.path.basename(os.path.normpath(sim_dir)) == selected_sim:
                                sim_full_dir = sim_dir
                                break
                        if not sim_full_dir:
                            raise FileNotFoundError(f"Simulation directory not found: {selected_sim}")

                        filenames = [os.path.join(sim_full_dir, f'o_fldare{year}.nc') for year in years]
                        valid_filenames = [f for f in filenames if os.path.exists(f)]
                        if not valid_filenames:
                            raise FileNotFoundError(
                                f"No flooded-area files found in {sim_full_dir}; please verify the data exist or re-run pre-processing."
                            )
                        sim_dataset = xr.open_mfdataset(valid_filenames, combine='nested', concat_dim='time')
                        if 'fldare' not in sim_dataset.variables:
                            raise KeyError(f"'fldare' variable missing in data files: {valid_filenames}")
                        average_fldare = np.average(sim_dataset['fldare'].values, axis=0)[::-1]
                        title = f'Average Flooded Area ({selected_sim})'
                        from process_simulations import plot_fldaremap
                        plot_path = plot_fldaremap(
                            data=average_fldare,
                            title=title,
                            filename=plot_path,
                            min_lon=config.get('MIN_LON', -125.0),
                            max_lon=config.get('MAX_LON', -66.0),
                            min_lat=config.get('MIN_LAT', 22.0),
                            max_lat=config.get('MAX_LAT', 54.0)
                        )
                elif variable == "maximum flood deepth":
                    plot_path = os.path.join(fig_dir, f'flooded_deepth_spatial_{selected_sim}.png')
                    if not os.path.exists(plot_path):
                        years = list(range(config.get('STARTYEAR', 2000), config.get('ENDYEAR', 2020) + 1))
                        sim_full_dir = None
                        for sim_dir in config.get('SIMDIR', []):
                            if os.path.basename(os.path.normpath(sim_dir)) == selected_sim:
                                sim_full_dir = sim_dir
                                break
                        if not sim_full_dir:
                            raise FileNotFoundError(f"Simulation directory not found: {selected_sim}")

                        filenames = [os.path.join(sim_full_dir, f'o_maxdph{year}.nc') for year in years]
                        valid_filenames = [f for f in filenames if os.path.exists(f)]
                        if not valid_filenames:
                            raise FileNotFoundError(
                                f"No flooded-deepth files found in {sim_full_dir}; please verify the data exist or re-run pre-processing."
                            )
                        sim_dataset = xr.open_mfdataset(valid_filenames, combine='nested', concat_dim='time')
                        if 'maxdph' not in sim_dataset.variables:
                            raise KeyError(f"'maxdph' variable missing in data files: {valid_filenames}")
                        average_maxdph = np.average(sim_dataset['maxdph'].values, axis=0)[::-1]
                        title = f'Average Max Flood Depth ({selected_sim})'
                        from process_simulations import plot_fldaremap
                        plot_path = plot_fldaremap(
                            data=average_maxdph,
                            title=title,
                            filename=plot_path,
                            min_lon=config.get('MIN_LON', -125.0),
                            max_lon=config.get('MAX_LON', -66.0),
                            min_lat=config.get('MIN_LAT', 22.0),
                            max_lat=config.get('MAX_LAT', 54.0)
                        )
                elif variable == "river network":
                    plot_path = plot_river_network_spatial(
                        sim_full_dir=sim_full_dir,
                        fig_dir=fig_dir,
                        start_year=config.get('STARTYEAR', 2000),
                        end_year=config.get('ENDYEAR', 2020),
                        min_lon=config.get('MIN_LON', -180),
                        max_lon=config.get('MAX_LON', 180),
                        min_lat=config.get('MIN_LAT', -90),
                        max_lat=config.get('MAX_LAT', 90)
                    )
                elif variable == "water level":
                    plot_path = plot_water_level_spatial(
                        sim_full_dir=sim_full_dir,
                        fig_dir=fig_dir,
                        start_year=config.get('STARTYEAR', 2000),
                        end_year=config.get('ENDYEAR', 2020),
                        min_lon=config.get('MIN_LON', -180),
                        max_lon=config.get('MAX_LON', 180),
                        min_lat=config.get('MIN_LAT', -90),
                        max_lat=config.get('MAX_LAT', 90)
                    )
                else:
                    if not indicator or indicator == "":
                        QMessageBox.warning(self, "Warning",
                                            "Please select a valid indicator. Current value: '{}'".format(
                                                indicator if indicator else "None"))
                        return
                    metrics_path = os.path.join(output_folder, selected_sim, 'metrics.csv')
                    if not os.path.exists(metrics_path):
                        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
                    df = pd.read_csv(metrics_path)
                    valid_col = indicator.upper()
                    if valid_col not in df.columns:
                        raise ValueError(f"Indicator {indicator} not found in metrics.")
                    ind0 = df[df[valid_col] > -9999].index
                    data_select = df.loc[ind0]
                    if valid_col in ['KGESS', 'CORRELATION']:
                        ind1 = data_select[data_select[valid_col].between(-1, 1)].index
                        data_select = data_select.loc[ind1]

                    stn_lon = data_select['lon'].values
                    stn_lat = data_select['lat'].values
                    metric = data_select[valid_col].values

                    min_lon = config.get('MIN_LON', -125.0)
                    max_lon = config.get('MAX_LON', -66.0)
                    min_lat = config.get('MIN_LAT', 22.0)
                    max_lat = config.get('MAX_LAT', 54.0)

                    plot_spatial_map(
                        stn_lon, stn_lat, metric, indicator, fig_dir, selected_sim,
                        min_lon, max_lon, min_lat, max_lat
                    )

                    plot_path = os.path.join(fig_dir, f'spatial_{selected_sim}_{indicator}.png')

                if os.path.exists(plot_path):
                    pixmap = QPixmap(plot_path)
                    self.result_label.clear()
                    self.result_label.setPixmap(
                        pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    )
                    self.result_label.setAlignment(Qt.AlignCenter)
                else:
                    self.result_label.setText("Failed to generate spatial map")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to generate spatial map: {str(e)}")
                self.result_label.setText("Failed to generate spatial map")

    def return_to_main(self):
        self.stack.setCurrentWidget(self.page1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EvalApp()
    window.show()
    sys.exit(app.exec_())