import sys
from PySide6.QtWidgets import QApplication,QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl
from PySide6.QtWebEngineCore import QWebEngineSettings
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("X射线安保检测系统")
        self.setGeometry(100,100,1024,768)
        self.webview = QWebEngineView()
        settings=self.webview.page().settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled,True)
        self.webview.setUrl(QUrl("http://localhost:8080"))
        self.setCentralWidget(self.webview)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin=MainWindow()
    mainWin.show()
    sys.exit(app.exec())