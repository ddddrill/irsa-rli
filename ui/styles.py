APP_STYLE = """
    QMainWindow {
        background-color: #f5f6fa;
        font-family: "Segoe UI", "Arial", sans-serif;
    }

    QGroupBox {
        font-size: 13px;
        font-weight: bold;
        color: #2c3e50;
        border: 1px solid #dcdde1;
        border-radius: 8px;
        margin-top: 12px;
        padding: 16px 10px 10px 10px;
        background-color: #ffffff;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: #3498db;
    }

    QComboBox {
        border: 1px solid #dcdde1;
        border-radius: 5px;
        padding: 5px 8px;
        background: #ffffff;
        font-size: 12px;
        min-height: 24px;
    }
    QComboBox:hover {
        border-color: #3498db;
    }
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    QComboBox QAbstractItemView {
        background-color: #ffffff;
        color: #2c3e50;
        selection-background-color: #3498db;
        selection-color: #ffffff;
        border: 1px solid #dcdde1;
        padding: 2px;
    }
    QComboBox QAbstractItemView::item {
        padding: 4px 8px;
        min-height: 24px;
    }
    QComboBox QAbstractItemView::item:hover {
        background-color: #ebf5fb;
        color: #2c3e50;
    }

    QLineEdit {
        border: 1px solid #dcdde1;
        border-radius: 5px;
        padding: 5px 8px;
        background: #ffffff;
        font-size: 12px;
        min-height: 24px;
    }
    QLineEdit:focus {
        border-color: #3498db;
    }

    QPushButton {
        background-color: #3498db;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 12px;
        font-weight: bold;
        min-height: 28px;
    }
    QPushButton:hover {
        background-color: #2980b9;
    }
    QPushButton:pressed {
        background-color: #1a6fa0;
    }
    QPushButton:disabled {
        background-color: #bdc3c7;
        color: #7f8c8d;
    }

    QPushButton#btn_secondary {
        background-color: #27ae60;
    }
    QPushButton#btn_secondary:hover {
        background-color: #219a52;
    }
    QPushButton#btn_secondary:pressed {
        background-color: #1a7a40;
    }

    QPushButton#btn_action {
        background-color: #e74c3c;
    }
    QPushButton#btn_action:hover {
        background-color: #c0392b;
    }
    QPushButton#btn_action:pressed {
        background-color: #a33025;
    }

    QProgressBar {
        border: 1px solid #dcdde1;
        border-radius: 6px;
        text-align: center;
        background-color: #ecf0f1;
        font-size: 11px;
        min-height: 18px;
    }
    QProgressBar::chunk {
        background-color: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 #3498db, stop:1 #2ecc71
        );
        border-radius: 5px;
    }

    QLabel {
        color: #2c3e50;
        font-size: 12px;
    }

    QLabel#title_label {
        font-size: 14px;
        font-weight: bold;
        color: #2c3e50;
        padding: 4px 0;
    }

    QLabel#sat_info {
        background-color: #fafafa;
        border: 1px solid #dcdde1;
        border-radius: 6px;
        padding: 10px;
        font-size: 12px;
        line-height: 1.5;
    }

    QLabel#status_info {
        color: #7f8c8d;
        font-size: 11px;
        font-style: italic;
    }

    QSplitter::handle {
        background-color: #dcdde1;
        width: 2px;
    }
    QSplitter::handle:hover {
        background-color: #3498db;
    }

    QStatusBar {
        background-color: #ecf0f1;
        color: #7f8c8d;
        font-size: 11px;
        border-top: 1px solid #dcdde1;
    }

    QScrollArea {
        border: none;
        background-color: transparent;
    }

    QScrollBar:vertical {
        border: none;
        background: #f0f0f0;
        width: 8px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: #bdc3c7;
        border-radius: 4px;
        min-height: 30px;
    }
    QScrollBar::handle:vertical:hover {
        background: #95a5a6;
    }
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {
        height: 0;
    }
"""
