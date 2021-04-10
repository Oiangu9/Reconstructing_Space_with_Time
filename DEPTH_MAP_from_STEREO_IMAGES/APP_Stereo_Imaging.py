from UI.Design_ui import *
from SOURCE.Stereo_Calibrator import Stereo_Calibrator
import os
import sys
# pyuic5 -x Design.ui -o Design_ui.py

"""
TODOS
----

- Probeu aadoan
- Arregleu ia kamarien psoiziñoa zan problemie
- Gehitu opzxiñoa depth mapak eitzeko en general (quiza con disparity ya baste)
- Lortu ez arren perfektoa dale audioa kontroleteie ordenadoretik eta record-sound-record-sound eitzie pythonetik ia al badozun.
- Ikusi ia ondo printietako gai zaren matplotlibegaz edo opencvgaz edo ia zer, Todorrenerako be ondo etorko da eta

Proyektu Todor
-----------
- Plantieu GUIxe, timer bat etabar erakustie
- Ein funkiñoa i607
- Implementeu rotaziñoan bersiño lehena
- Histogramien bersiño lehena
- Roitaziñoa stokatsiko
- Histogramiena azalerakaz
- Saiatu betie eitzen fletxitana

"""

import logging


# To be able to log into the QPlainTextEdit widget directly
class QPlainTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()

        self.widget = parent
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)



class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        # set the working directory
        self.working_directory.setText(os.getcwd())

        # Set up logging to use your widget as a handler
        log_handler = QPlainTextEditLogger(self.log_text)
        # You can format what is printed to text box

        # connect with logger
        logging.getLogger().addHandler(log_handler)
        # You can control the logging level
        logging.getLogger().setLevel(logging.DEBUG)

        # Generate a code signaler
        #self.waitingPhotos = QtCore.pyqtSignal()

        # We connect the events with their actions
        self.start_calibration.clicked.connect(self.executeCalibration)



    def clickToPhotos(self):
        ret = QtWidgets.QMessageBox.question(self, 'Take Photos', "Press OK when\n you are ready\n to take photos", QtWidgets.QMessageBox.Ok)
        return ret

    def executeCalibration(self):
        # block the button
        self.start_calibration.setEnabled(False)
        # We gather all the arguments chosen by the user in a dictionary .isChecked() and .text() .setText()
        user_defined_parameters={}
        user_defined_parameters["working_directory"] = self.working_directory.text()
        user_defined_parameters["num_photos_calibration"] = int(self.num_photos_calibration.value())
        user_defined_parameters["num_photos_test"] = int(self.num_photos_test.value())
        user_defined_parameters["allow_choosing"] = self.allow_choosing.isChecked()
        user_defined_parameters["previs_ms"] = int(self.previs_ms.text())
        user_defined_parameters["chess_size_x"] = int(self.chess_size_x.text())
        user_defined_parameters["chess_size_y"] = int(self.chess_size_y.text())
        user_defined_parameters["chess_square_side"] = float(self.chess_square_side.text())
        user_defined_parameters["corner_criteria_its"] = int(self.corner_criteria_its.text())
        user_defined_parameters["corner_criteria_eps"] = float(self.corner_criteria_eps.text())
        user_defined_parameters["camera_criteria_its"] = int(self.camera_criteria_its.text())
        user_defined_parameters["camera_criteria_eps"] = float(self.camera_criteria_eps.text())
        user_defined_parameters['CALIB_USE_INTRINSIC_GUESS'] = self.CALIB_USE_INTRINSIC_GUESS.isChecked()
        user_defined_parameters['CALIB_FIX_INTRINSIC'] = self.CALIB_FIX_INTRINSIC.isChecked()
        user_defined_parameters['CALIB_FIX_PRINCIPAL_POINT'] = self.CALIB_FIX_PRINCIPAL_POINT.isChecked()
        user_defined_parameters['CALIB_FIX_FOCAL_LENGTH'] = self.CALIB_FIX_FOCAL_LENGTH.isChecked()
        user_defined_parameters['CALIB_FIX_ASPECT_RATIO'] = self.CALIB_FIX_ASPECT_RATIO.isChecked()
        user_defined_parameters['CALIB_SAME_FOCAL_LENGTH'] = self.CALIB_SAME_FOCAL_LENGTH.isChecked()
        user_defined_parameters['CALIB_ZERO_TANGENT_DIST'] = self.CALIB_ZERO_TANGENT_DIST.isChecked()
        user_defined_parameters['CALIB_FIX_K1'] = self.CALIB_FIX_K1.isChecked()
        user_defined_parameters['CALIB_FIX_K2'] = self.CALIB_FIX_K2.isChecked()
        user_defined_parameters['CALIB_FIX_K3'] = self.CALIB_FIX_K3.isChecked()
        user_defined_parameters['CALIB_FIX_K4'] = self.CALIB_FIX_K4.isChecked()
        user_defined_parameters['CALIB_FIX_K5'] = self.CALIB_FIX_K5.isChecked()
        user_defined_parameters['CALIB_FIX_K6'] = self.CALIB_FIX_K6.isChecked()
        user_defined_parameters['CALIB_ZERO_DISPARITY'] = self.CALIB_ZERO_DISPARITY.isChecked()
        user_defined_parameters["stereocalib_criteria_its"] = int(self.stereocalib_criteria_its.text())
        user_defined_parameters["stereocalib_criteria_eps"] = float(self.stereocalib_criteria_eps.text())
        user_defined_parameters["alpha"] = float(self.alpha.text())
        user_defined_parameters["block_size"]= float(self.block_size.text())
        user_defined_parameters["min_disp"]= float(self.min_disp.text())
        user_defined_parameters["max_disp"]= float(self.max_disp.text())
        user_defined_parameters["num_disp"]= float(self.num_disp.text())
        user_defined_parameters["uniquenessRatio"]= float(self.uniquenessRatio.text())
        user_defined_parameters["speckleWindowSize"]= float(self.speckleWindowSize.text())
        user_defined_parameters["disp12MaxDiff"]= float(self.disp12MaxDiff.text())
        user_defined_parameters["speckleRange"]= float(self.speckleRange.text())
        user_defined_parameters["lmbda"]= float(self.lmbda.text())
        user_defined_parameters["sigma"]= float(self.sigma.text())
        user_defined_parameters["visual_multiplier"]= float(self.visual_multiplier.text())

        # We initialize an instance of the Setero_Calibrator object
        stereo_Calibrator = Stereo_Calibrator( user_defined_parameters )

        # We execute all the pipeline, notifying the user in between
        os.chdir(user_defined_parameters["working_directory"])
        self.progressBar_Calibrate.setValue(1)

        ret = stereo_Calibrator.setCameras(int(self.cam_L_idx.value()),
                                            int(self.cam_R_idx.value())) #%5
        if (ret==1):
            logging.error("\nTry readjusting cameras and Start Calibration Again!")
            self.progressBar_Calibrate.setValue(0)
            self.start_calibration.setEnabled(True)
            return 1

        self.progressBar_Calibrate.setValue(5)

        stereo_Calibrator.clean_directories_build_new() #%6
        self.progressBar_Calibrate.setValue(6)

        ret = self.clickToPhotos()
        logging.info(f"A que {ret}")
        stereo_Calibrator.take_chess_photos_compute_points() #%15
        self.progressBar_Calibrate.setValue(15)

        stereo_Calibrator.calibrate_cameras() #%25
        self.progressBar_Calibrate.setValue(25)

        stereo_Calibrator.compute_Fundamental_Matrix() #%40
        self.progressBar_Calibrate.setValue(40)

        stereo_Calibrator.draw_Epilines() #%55
        self.progressBar_Calibrate.setValue(55)

        stereo_Calibrator.compute_Stereo_Rectification() # %65
        self.progressBar_Calibrate.setValue(65)

        stereo_Calibrator.rectify_chess() #%75
        self.progressBar_Calibrate.setValue(75)

        stereo_Calibrator.compute_Disparity_Map() #%85
        self.progressBar_Calibrate.setValue(85)

        self.clickToPhotos()
        stereo_Calibrator.do_Test() #%100
        self.progressBar_Calibrate.setValue(100)

        # reset the button
        self.start_calibration.setEnabled(True)


if __name__ == "__main__":

    # Initialize and execute app
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
