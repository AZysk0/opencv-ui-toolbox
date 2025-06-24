import sys
from ui import QApplication, MainWindow


def main ():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    return 0


if __name__ == "__main__":
    main()




