import sys
import warnings
import PyQtImportFix    # Runs the script to ad QTCore to PATH (called on import)
warnings.filterwarnings('ignore')

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtCore import QTimer


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Directory Selector'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480

    def openDirectoryDialog(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directoryName = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.show()
        return directoryName


def getDir():
    app = QApplication(sys.argv)
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)
    ex = App()
    path = ex.openDirectoryDialog()
    return path
