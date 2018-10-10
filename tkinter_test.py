from PyQt5.QtWidgets import *
import sys

app = QApplication(sys.argv)

w = QWidget()
w.resize(250, 150)
w.move(300, 300)
w.setWindowTitle('Simple')

btn = QPushButton('Button', w)
btn.setToolTip('This is a <b>QPushButton</b> widget');
btn.resize(btn.sizeHint())
btn.move(50, 50)

w.show()

sys.exit(app.exec_())