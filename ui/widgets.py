from functools import wraps

from PyQt5.QtWidgets import QLabel, QPushButton
from PyQt5.QtGui import QFont


def label_decorator(font_name="Arial", font_size=10, bold=True):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            label = func(self, *args, **kwargs)
            font = QFont(font_name, font_size)
            font.setBold(bold)
            label.setFont(font)
            return label
        return wrapper
    return decorator


def label_img_decorator(wind_x=10, wind_y=10):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            label = func(self, *args, **kwargs)
            label.resize(wind_x, wind_y)
            if "x" in kwargs and "y" in kwargs:
                label.move(kwargs["x"], kwargs["y"])
            self.layout.addWidget(label)
            return label
        return wrapper
    return decorator


def button_decorator(text, x, y, width=150, height=26):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            button = func(self, *args, **kwargs)
            button.setText(text)
            button.move(x, y)
            button.setFixedSize(width, height)
            self.layout.addWidget(button)
            return button
        return wrapper
    return decorator


class LabelFactory:
    @staticmethod
    def create_title(parent, text, x, y, width, height):
        label = QLabel(text, parent)
        label.move(x, y)
        label.setFixedSize(width, height)
        font = QFont("Arial", 12)
        font.setBold(True)
        label.setFont(font)
        return label

    @staticmethod
    def create_common(parent, text, x, y, width, height):
        label = QLabel(text, parent)
        label.move(x, y)
        label.setFixedSize(width, height)
        font = QFont("Arial", 11)
        font.setBold(True)
        label.setFont(font)
        return label

    @staticmethod
    def create_info(parent, text, x, y, width, height):
        label = QLabel(text, parent)
        label.move(x, y)
        label.setFixedSize(width, height)
        font = QFont("Arial", 11)
        font.setBold(False)
        label.setFont(font)
        return label
