import sys
import cv2
import mediapipe as mp
import threading
import time
from pynput.keyboard import Controller, Key
import json
import os
import webbrowser
import subprocess
import queue
from PIL import Image, ImageTk
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QFrame, QGridLayout, QTextEdit, QLineEdit, QMessageBox,
    QScrollArea, QDialog, QListWidget, QAbstractItemView, QAction
)
from PyQt5.QtGui import QPixmap, QIcon, QImage, QPainter, QFont
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal

CONFIG_FILE = "gestures_macros_config.json"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

keyboard = Controller()


class ConfigManager:
    @staticmethod
    def load():
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            QMessageBox.critical(None, "Config Error", f"Failed to load config: {str(e)}")
            return {}

    @staticmethod
    def save(config):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(None, "Config Error", f"Failed to save config: {str(e)}")


class GestureRecognizer:
    @staticmethod
    def fingers_up(landmarks):
        tips_ids = [4, 8, 12, 16, 20]
        fingers = []

        if landmarks:
            thumb_tip = landmarks[tips_ids[0]]
            thumb_ip = landmarks[tips_ids[0] - 1]
            fingers.append(thumb_tip.x < thumb_ip.x - 0.02)

            for i in range(1, 5):
                tip = landmarks[tips_ids[i]]
                dip = landmarks[tips_ids[i] - 2]
                fingers.append(tip.y < dip.y + 0.03)

        return fingers

    @staticmethod
    def get_signature(fingers):
        return ''.join(['1' if f else '0' for f in fingers]) if fingers else ''


class MacroExecutor:
    @staticmethod
    def execute(actions):
        try:
            for action in actions:
                action = action.strip()
                if not action or action.startswith("#"):
                    continue

                if action.startswith("STRING:"):
                    MacroExecutor._type_string(action[7:])
                elif action.startswith("KEY:"):
                    MacroExecutor._press_key(action[4:])
                elif action.startswith("OPEN:"):
                    MacroExecutor._open_url(action[5:])
                elif action.startswith("WAIT:"):
                    MacroExecutor._wait(action[5:])
                elif action.startswith("CMD:"):
                    MacroExecutor._run_command(action[4:])
                else:
                    keyboard.type(action)

                time.sleep(0.05)
        except Exception as e:
            print(f"Critical error: {str(e)}")
            raise # Перенаправляем исключение в вызывающий код
    @staticmethod
    def _type_string(text):
        keyboard.type(text)

    @staticmethod
    def _press_key(keyname):
        try:
            key = getattr(Key, keyname)
        except AttributeError:
            key = keyname
        keyboard.press(key)
        keyboard.release(key)

    @staticmethod
    def _open_url(url):
        if url.strip():
            webbrowser.open(url.strip())

    @staticmethod
    def _wait(seconds):
        try:
            time.sleep(max(0, float(seconds.strip())))
        except:
            pass

    @staticmethod
    def _run_command(cmd):
        if cmd.strip():
            subprocess.Popen(cmd.strip(), shell=True)


class CameraHandler(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.running = False
        self.last_action = 0
        self.cooldown = 2
        self.timer = QTimer(self)  # Инициализация таймера в конструкторе
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def start(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            time.sleep(0.2)
        
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print("Ошибка инициализации камеры")
                return
            
            self.running = True
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # Теперь таймер гарантированно существует
        except Exception as e:
            print(f"Ошибка камеры: {str(e)}")

    def stop(self):
        self.running = False
        if hasattr(self, 'timer'):
            self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def update_frame(self):
        success, frame = self.cap.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7) as hands:

            results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            fingers = GestureRecognizer.fingers_up(landmarks)
            signature = GestureRecognizer.get_signature(fingers)
            
            # Добавьте проверку времени cooldown
            if time.time() - self.last_action > self.cooldown:
                self._handle_gesture(signature, frame, results)  # Было без аргументо

                mp_drawing.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))

    def _handle_gesture(self, signature, frame, results):  # Добавьте аргументы
        config = ConfigManager.load()
        if signature in config:
            self.last_action = time.time()
            # Добавляем проверку на активный поток
            if not hasattr(self, '_active_thread') or not self._active_thread.is_alive():
                self._active_thread = threading.Thread(
                    target=self.execute_macro_safe,
                    args=(config[signature]["actions"], config[signature]['name']),
                    daemon=True
                )
                self._active_thread.start()
            
    def execute_macro_safe(self, actions, name):
        try:
            MacroExecutor.execute(actions)
            # Используем лямбду для безопасного обновления GUI
            self.parent().status_bar.showMessage(f"Выполнено: {name}")  
        except Exception as e:
            self.parent().status_bar.showMessage(f"Ошибка: {str(e)}")       


class GestureMacroApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GestureMacro")
        self.setGeometry(100, 100, 900, 600)
        self.setStyleSheet("""
            background-color: #1E1E1E; 
            color: white; 
            font-family: Arial;
        """)

        self.camera = CameraHandler(self)
        self.init_ui()
        self.camera.start()

    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header
        header = QHBoxLayout()
        title = QLabel("GestureMacro")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        # Settings button
        settings_btn = QPushButton("⚙️")
        settings_btn.setFixedSize(30, 30)
        settings_btn.setStyleSheet("background: transparent; font-size: 16px;")
        settings_btn.clicked.connect(self.show_settings)
        header.addWidget(settings_btn)

        main_layout.addLayout(header)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                background: #1E1E1E;
                padding: 10px 20px;
                color: white;
            }
            QTabBar::tab:selected {
                border-bottom: 2px solid #2D8CFF;
            }
        """)

        # Macros Tab
        macros_tab = QWidget()
        macros_tab.setLayout(self.create_macros_tab())
        self.tabs.addTab(macros_tab, "Макросы")

        # Learning Tab
        learning_tab = QWidget()
        learning_tab.setLayout(self.create_learning_tab())
        self.tabs.addTab(learning_tab, "Режим обучения")

        main_layout.addWidget(self.tabs)

        # Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("color: #AAAAAA;")
        self.status_bar.showMessage("Ready")

    def create_macros_tab(self):
        layout = QHBoxLayout()

        # Left side - Camera
        left = QVBoxLayout()
        self.camera.setMinimumSize(300, 200)
        left.addWidget(self.camera)
        left.addStretch()
        layout.addLayout(left)

        # Right side - Macros table and buttons
        right = QVBoxLayout()

        self.macros_table = QTableWidget()
        self.macros_table.setColumnCount(3)
        self.macros_table.setHorizontalHeaderLabels(["Жест", "Описание", "Действие"])
        self.macros_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.macros_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Configure table style
        self.macros_table.setStyleSheet("""
            QTableWidget {
                background-color: #2B2B2B;
                color: white;
                border: none;
                gridline-color: #3E3E3E;
            }
            QHeaderView::section {
                background-color: #1E1E1E;
                font-weight: bold;
                padding: 5px;
                border: none;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)

        # Populate table
        self.update_macros_table()

        # Buttons layout - now includes all three buttons
        btn_layout = QHBoxLayout()

        add_btn = QPushButton("+ Добавить макрос")
        add_btn.setStyleSheet("""
            background-color: #2D8CFF; 
            color: white; 
            padding: 8px; 
            font-size: 14px;
            border-radius: 4px;
        """)
        add_btn.clicked.connect(self.show_add_dialog)
        btn_layout.addWidget(add_btn)

        edit_btn = QPushButton("Редактировать")
        edit_btn.setStyleSheet("""
            background-color: #3E3E3E;
            padding: 8px;
            border-radius: 4px;
        """)
        edit_btn.clicked.connect(self.show_edit_dialog)
        btn_layout.addWidget(edit_btn)

        delete_btn = QPushButton("Удалить")
        delete_btn.setStyleSheet("""
            background-color: #5E1E1E;
            padding: 8px;
            border-radius: 4px;
        """)
        delete_btn.clicked.connect(self.delete_macro)
        btn_layout.addWidget(delete_btn)

        right.addWidget(self.macros_table)
        right.addLayout(btn_layout)
        layout.addLayout(right)

        return layout

    def create_learning_tab(self):
        layout = QVBoxLayout()

        # Add content for learning tab
        label = QLabel("Режим обучения: покажите жесты камере для распознавания")
        label.setStyleSheet("font-size: 16px;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        return layout

    def update_macros_table(self):
        config = ConfigManager.load()
        self.macros_table.setRowCount(len(config))
        
        for row, (sig, data) in enumerate(config.items()):
            # Защита от пустых действий
            actions = data.get('actions', ['Нет действий'])
            action_preview = actions[0][:50] + "..." if actions else "Нет действий"
            
            self.macros_table.setItem(row, 0, QTableWidgetItem(data['name']))
            self.macros_table.setItem(row, 1, QTableWidgetItem(sig))
            self.macros_table.setItem(row, 2, QTableWidgetItem(action_preview))
        
        self.macros_table.resizeColumnsToContents()

    def show_add_dialog(self):
        dialog = AddGestureDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.update_macros_table()

    def show_edit_dialog(self):
        selected = self.macros_table.currentRow()
        if selected == -1:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите макрос для редактирования")
            return

        config = ConfigManager.load()
        sig = list(config.keys())[selected]
        name = config[sig]['name']

        dialog = EditMacroDialog(self, sig, name)
        if dialog.exec_() == QDialog.Accepted:
            self.update_macros_table()

    def delete_macro(self):
        selected = self.macros_table.currentRow()
        if selected == -1:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите макрос для удаления")
            return

        config = ConfigManager.load()
        sig = list(config.keys())[selected]
        name = config[sig]['name']

        reply = QMessageBox.question(
            self, 'Подтверждение',
            f"Вы уверены, что хотите удалить макрос '{name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            del config[sig]
            ConfigManager.save(config)
            self.update_macros_table()

    def show_settings(self):
        QMessageBox.information(self, "Настройки", "Здесь будут настройки приложения")

    def update_status(self, message):  # Обычный метод без сигналов
        self.status_bar.showMessage(message)

    def closeEvent(self, event):
        self.camera.stop()
        event.accept()


class AddGestureDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Добавить новый жест")
        self.setModal(True)
        self.setStyleSheet("background-color: #2B2B2B; color: white;")
        self.setFixedSize(400, 300)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Name input
        name_label = QLabel("Название жеста:")
        name_label.setStyleSheet("font-size: 14px;")
        self.name_input = QLineEdit()
        self.name_input.setStyleSheet("""
            background-color: #3E3E3E;
            color: white;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #4E4E4E;
        """)

        # Camera preview
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black;")

        # Buttons
        btn_layout = QHBoxLayout()
        record_btn = QPushButton("Записать жест")
        record_btn.setStyleSheet("""
            background-color: #2D8CFF;
            color: white;
            padding: 10px;
            font-size: 14px;
            border-radius: 4px;
        """)
        record_btn.clicked.connect(self.start_recording)

        cancel_btn = QPushButton("Отмена")
        cancel_btn.setStyleSheet("""
            background-color: #5E1E1E;
            color: white;
            padding: 10px;
            font-size: 14px;
            border-radius: 4px;
        """)
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(record_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addWidget(name_label)
        layout.addWidget(self.name_input)
        layout.addWidget(self.camera_label)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def start_recording(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, введите название жеста")
            return
        recorder = GestureRecorder(self, self.name_input.text())
        recorder.finished.connect(self._handle_recorder_close)
        recorder.open()

    def _handle_recorder_close(self, result):
        if result == QDialog.Accepted:
            # Принудительное обновление таблицы
            self.parent().update_macros_table()
            # Явный перезапуск камеры
            QTimer.singleShot(200, self.parent().camera.start)
        self.show()



class GestureRecorder(QDialog):
    def __init__(self, parent, name):
        super().__init__(parent)
        self.name = name
        self.signature = None
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Ошибка", "Не удалось открыть камеру")
            self.reject()
            return

        self.setWindowTitle("Запись жеста")
        self.setModal(True)
        self.setStyleSheet("background-color: #2B2B2B; color: white;")
        self.setFixedSize(640, 480)

        self.init_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):
        layout = QVBoxLayout()

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black;")

        self.status_label = QLabel("Покажите жест камере")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px;")

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Сохранить")
        save_btn.setStyleSheet("""
            background-color: #2D8CFF;
            color: white;
            padding: 10px;
            font-size: 14px;
            border-radius: 4px;
        """)
        save_btn.clicked.connect(self.save_gesture)

        cancel_btn = QPushButton("Отмена")
        cancel_btn.setStyleSheet("""
            background-color: #5E1E1E;
            color: white;
            padding: 10px;
            font-size: 14px;
            border-radius: 4px;
        """)
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addWidget(self.camera_label)
        layout.addWidget(self.status_label)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7) as hands:

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                fingers = GestureRecognizer.fingers_up(landmarks)
                self.signature = GestureRecognizer.get_signature(fingers)

                mp_drawing.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                self.status_label.setText("Жест распознан! Нажмите 'Сохранить'")

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.size(), Qt.KeepAspectRatio))

    def save_gesture(self):
        if not self.signature:
            QMessageBox.warning(self, "Ошибка", "Жест не распознан")
            return

        config = ConfigManager.load()
        if self.signature in config:
            QMessageBox.warning(self, "Ошибка", "Этот жест уже существует")
            return

        # Сразу создаем запись с пустыми действиями
        config[self.signature] = {
            "name": self.name,
            "actions": ["# Добавьте действия через EditMacroDialog"]
        }
        ConfigManager.save(config)
        
        dialog = EditMacroDialog(self, self.signature, self.name)
        if dialog.exec_() == QDialog.Accepted:
            self.accept()
        else:
            # Удаляем временную запись если пользователь отменил редактирование
            del config[self.signature]
            ConfigManager.save(config)

    def _restart_camera(self):
        try:
            self.parent().camera.stop()
            time.sleep(0.3)
            self.parent().camera.start()
        except Exception as e:
            print(f"Camera restart failed: {str(e)}")

    def closeEvent(self, event):
        try:
            self.cap.release()
        except:
            pass
        
        # Явный перезапуск камеры с проверкой
        if self.parent() and hasattr(self.parent(), 'camera'):
            QTimer.singleShot(100, self._safe_restart)
        
        event.accept()
    
    def _safe_restart(self):
        try:
            self.parent().camera.stop()
            time.sleep(0.3)
            self.parent().camera.start()
        except Exception as e:
            print(f"Camera restart error: {str(e)}")

class EditMacroDialog(QDialog):
    def __init__(self, parent, signature, name):
        super().__init__(parent)
        self.signature = signature
        self.name = name

        self.setWindowTitle(f"Редактирование: {name}")
        self.setModal(True)
        self.setStyleSheet("background-color: #2B2B2B; color: white;")
        self.setFixedSize(600, 500)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Name and signature
        info_layout = QHBoxLayout()
        name_label = QLabel("Название:")
        self.name_input = QLineEdit(self.name)
        self.name_input.setStyleSheet("""
            background-color: #3E3E3E;
            color: white;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #4E4E4E;
        """)

        sig_label = QLabel(f"Сигнатура: {self.signature}")
        sig_label.setStyleSheet("font-family: monospace;")

        info_layout.addWidget(name_label)
        info_layout.addWidget(self.name_input)
        info_layout.addWidget(sig_label)
        info_layout.addStretch()

        layout.addLayout(info_layout)

        # Actions editor
        actions_label = QLabel("Действия:")
        self.actions_edit = QTextEdit()
        self.actions_edit.setStyleSheet("""
            background-color: #3E3E3E;
            color: white;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #4E4E4E;
        """)

        # Load existing actions if editing
        config = ConfigManager.load()
        if self.signature in config:
            self.actions_edit.setPlainText("\n".join(config[self.signature]["actions"]))

        # Help button
        help_btn = QPushButton("Справка по командам")
        help_btn.setStyleSheet("""
            background-color: #3E3E3E;
            color: white;
            padding: 8px;
            font-size: 12px;
            border-radius: 4px;
        """)
        help_btn.clicked.connect(self.show_help)

        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Сохранить")
        save_btn.setStyleSheet("""
            background-color: #2D8CFF;
            color: white;
            padding: 10px;
            font-size: 14px;
            border-radius: 4px;
        """)
        save_btn.clicked.connect(self.save_macro)

        cancel_btn = QPushButton("Отмена")
        cancel_btn.setStyleSheet("""
            background-color: #5E1E1E;
            color: white;
            padding: 10px;
            font-size: 14px;
            border-radius: 4px;
        """)
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(help_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addWidget(actions_label)
        layout.addWidget(self.actions_edit)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def show_help(self):
        help_text = """Доступные команды:

STRING: <текст> - Напечатать текст
  Пример: STRING: Привет, мир!

KEY: <клавиша> - Нажать клавишу
  Примеры: KEY: ctrl+c, KEY: alt+f4

OPEN: <URL> - Открыть веб-страницу
  Пример: OPEN: https://google.com

WAIT: <секунды> - Пауза
  Пример: WAIT: 0.5

CMD: <команда> - Выполнить команду
  Пример: CMD: notepad.exe"""

        QMessageBox.information(self, "Справка по командам", help_text)

    def save_macro(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, введите название")
            return

        actions = self.actions_edit.toPlainText().strip().split('\n')
        valid_actions = [a.strip() for a in actions if a.strip()]
        
        if not valid_actions:  # Добавляем проверку на пустые действия
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, введите хотя бы одно действие")
            return

        config = ConfigManager.load()
        config[self.signature] = {
            "name": name,
            "actions": valid_actions
        }
        ConfigManager.save(config)
        self.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GestureMacroApp()
    window.show()
    sys.exit(app.exec_())