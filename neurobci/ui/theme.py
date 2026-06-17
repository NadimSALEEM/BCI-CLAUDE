"""Dark theme stylesheet for the application."""

DARK_QSS = """
QWidget { background-color: #14181c; color: #cdd4da; font-size: 13px; }
QGroupBox {
    border: 1px solid #2a3038; border-radius: 6px; margin-top: 10px; padding-top: 6px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #8fa3b0; }
QPushButton {
    background: #232b33; border: 1px solid #323b45; border-radius: 4px; padding: 5px 14px;
}
QPushButton:hover { background: #2c353f; }
QPushButton:disabled { color: #5a636b; background: #1b2127; }
QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox {
    background: #1b2127; border: 1px solid #323b45; border-radius: 4px; padding: 3px 6px;
}
QTabWidget::pane { border: 1px solid #2a3038; }
QTabBar::tab {
    background: #1b2127; padding: 7px 16px; border: 1px solid #2a3038; border-bottom: none;
}
QTabBar::tab:selected { background: #232b33; color: #ffffff; }
QHeaderView::section { background: #1b2127; padding: 4px; border: 1px solid #2a3038; }
QTableWidget { gridline-color: #2a3038; }
QToolBar { background: #1b2127; border-bottom: 1px solid #2a3038; spacing: 6px; padding: 4px; }
QStatusBar { background: #1b2127; }
"""
