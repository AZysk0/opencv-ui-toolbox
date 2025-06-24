from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QStackedWidget, QListWidget, QListWidgetItem,
    QFormLayout, QSpinBox, QMainWindow
)
from PySide6.QtCore import Signal, Qt, QMimeData
from PySide6.QtGui import QImage, QPixmap, QDrag

import cv2
import numpy as np
import sys

# ======= Drag-Drop widgets ==========================
class DragItemBase(QWidget):
    def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setContentsMargins(25, 5, 25, 5)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setStyleSheet("border: 1px solid green;")

    def set_data(self, data):
        self.data = data

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            drag.setMimeData(mime)

            # Render at x2 pixel ratio to avoid blur on Retina screens.
            pixmap = QPixmap(self.size().width() * 2, self.size().height() * 2)
            pixmap.setDevicePixelRatio(2)
            self.render(pixmap)
            drag.setPixmap(pixmap)

            drag.exec(Qt.DropAction.MoveAction)
            self.show() # Show this widget again, if it's dropped outside.


class DragLabelItem(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setContentsMargins(25, 5, 25, 5)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid black;")
        # Store data separately from display label, but use label for default.
        self.data = self.text()

    def set_data(self, data):
        self.data = data

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            drag.setMimeData(mime)

            # Render at x2 pixel ratio to avoid blur on Retina screens.
            pixmap = QPixmap(self.size().width() * 2, self.size().height() * 2)
            pixmap.setDevicePixelRatio(2)
            self.render(pixmap)
            drag.setPixmap(pixmap)

            drag.exec(Qt.DropAction.MoveAction)
            self.show() # Show this widget again, if it's dropped outside.


class DragTargetIndicator(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(25, 5, 25, 5)
        self.setStyleSheet(
            "QLabel { background-color: #ccc; border: 1px solid black; }"
        )


class DragWidget(QWidget):
    """
    Generic list sorting handler.
    """

    orderChanged = Signal(list)

    def __init__(self, *args, orientation=Qt.Orientation.Vertical, **kwargs):
        super().__init__()
        self.setAcceptDrops(True)

        # Store the orientation for drag checks later.
        self.orientation = orientation

        if self.orientation == Qt.Orientation.Vertical:
            self.blayout = QVBoxLayout()
        else:
            self.blayout = QHBoxLayout()

        # Add the drag target indicator. This is invisible by default,
        # we show it and move it around while the drag is active.
        self._drag_target_indicator = DragTargetIndicator()
        self.blayout.addWidget(self._drag_target_indicator)
        self._drag_target_indicator.hide()

        self.setLayout(self.blayout)

    def dragEnterEvent(self, e):
        e.accept()

    def dragLeaveEvent(self, e):
        self._drag_target_indicator.hide()
        e.accept()

    def dragMoveEvent(self, e):
        # Find the correct location of the drop target, so we can move it there.
        index = self._find_drop_location(e)
        if index is not None:
            # Inserting moves the item if its alreaady in the layout.
            self.blayout.insertWidget(index, self._drag_target_indicator)
            # Hide the item being dragged.
            e.source().hide()
            # Show the target.
            self._drag_target_indicator.show()
        e.accept()

    def dropEvent(self, e):
        widget = e.source()
        # Use drop target location for destination, then remove it.
        self._drag_target_indicator.hide()
        index = self.blayout.indexOf(self._drag_target_indicator)
        if index is not None:
            self.blayout.insertWidget(index, widget)
            self.orderChanged.emit(self.get_item_data())
            widget.show()
            self.blayout.activate()
        e.accept()

    def _find_drop_location(self, e):
        pos = e.position()
        spacing = self.blayout.spacing() / 2

        for n in range(self.blayout.count()):
            # Get the widget at each index in turn.
            w = self.blayout.itemAt(n).widget()

            if self.orientation == Qt.Orientation.Vertical:
                # Drag drop vertically.
                drop_here = (
                    pos.y() >= w.y() - spacing
                    and pos.y() <= w.y() + w.size().height() + spacing
                )
            else:
                # Drag drop horizontally.
                drop_here = (
                    pos.x() >= w.x() - spacing
                    and pos.x() <= w.x() + w.size().width() + spacing
                )

            if drop_here:
                # Drop over this target.
                break

        return n

    def add_item(self, item):
        self.blayout.addWidget(item)

    def get_item_data(self):
        data = []
        for n in range(self.blayout.count()):
            # Get the widget at each index in turn.
            w = self.blayout.itemAt(n).widget()
            if w != self._drag_target_indicator:
                # The target indicator has no data.
                data.append(w.data)
        return data


# ============== Specific OpenCV functions (buttons) ===================
class GaussianBlur(DragItemBase):
    def __init__(self):
        super().__init__()
    
class MeanBlur(DragItemBase):
    def __init__(self):
        super().__init__()

class CannyEdges(DragItemBase):
    def __init__(self):
        super().__init__()

class Threshold(DragItemBase):
    def __init__(self):
        super().__init__()

class Resize(DragItemBase):
    def __init__(self):
        super().__init__()



# =====================
class ButtonAdd(QPushButton):
    def __init__(self):
        super().__init__("+ Add Operation")
        self.setToolTip("Add a new vision function")
        self.setStyleSheet("""
            QPushButton {
                background-color: #d0f0d0;
                border: 2px dashed #50a060;
                color: #305030;
                font-weight: bold;
                padding: 8px;
                margin-top: 10px;
                margin-bottom: 10px;
            }
            QPushButton:hover {
                background-color: #b0e0b0;
            }
        """)
    # def mousePressEvent(self, e):
    #     e.accept()
    
    def mousePressEvent(self, e):
        super().mousePressEvent(e)  # This will emit the .clicked signal


class ListlikeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.drag = DragWidget(orientation=Qt.Orientation.Vertical)
        for n, l in enumerate(["A", "B", "C", "D"]):
            item = DragLabelItem(l)
            item.set_data(n)  # Store the data.
            self.drag.add_item(item)

        self.drag.orderChanged.connect(print)

        layout = QVBoxLayout(self)  # set layout on self
        layout.addWidget(self.drag)


class PipelinePanel(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("Pipeline"))
        
        # self.layout.addWidget(GaussianBlur())

        # Add button at the end
        self.add_button = ButtonAdd()
        # self.add_button.clicked.connect(self.add_new_function)
        self.add_button.clicked.connect(self.debug)
        self.layout.addWidget(self.add_button)
        self.layout.addStretch()
    
    def add_new_function(self):
        func = GaussianBlur()
        self.layout.insertWidget(self.layout.count() - 2, func)  # Insert before button + stretch

    def debug(self):
        print('test add button')
    

class ImagePanelSample(QLabel):
    def __init__(self):
        super().__init__()
        self.setScaledContents(True)

    def set_cv_image(self, cv_img):
        if len(cv_img.shape) == 2:
            qimg = QImage(cv_img.data, cv_img.shape[1], cv_img.shape[0], QImage.Format_Grayscale8)
        else:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            qimg = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)

        self.setPixmap(QPixmap.fromImage(qimg))
    
    def add_image(self):
        ...
        
    def remove_image(self):
        ...


class ImagePanel(QWidget):
    def __init__(self):
        super().__init__()



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV UI Tool")
        self.resize(900, 600)

        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QHBoxLayout(container)

        # LEFT SIDE
        # self.listlike_widget = ListlikeWidget()
        self.pipeline_widget = PipelinePanel()
        main_layout.addWidget(self.pipeline_widget)

        # RIGHT SIDE
        image_panel = QLabel("ImagePanel")
        image_panel.setAlignment(Qt.AlignCenter)
        image_panel.setStyleSheet("background-color: #f0f0f0;")
        main_layout.addWidget(image_panel)

        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 2)
        
        # self.drag = DragWidget(orientation=Qt.Orientation.Vertical)
        # for n, l in enumerate(["A", "B", "C", "D"]):
        #     item = DragItem(l)
        #     item.set_data(n)  # Store the data.
        #     self.drag.add_item(item)

        # # Print out the changed order.
        # self.drag.orderChanged.connect(print)

        # container = QWidget()
        # layout = QVBoxLayout()
        # layout.addStretch(1)
        # layout.addWidget(self.drag)
        # layout.addStretch(1)
        # container.setLayout(layout)

        # self.setCentralWidget(container)

    def load_sample_image(self):
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(img, 'Pipeline Tool', (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 3, cv2.LINE_AA)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())

