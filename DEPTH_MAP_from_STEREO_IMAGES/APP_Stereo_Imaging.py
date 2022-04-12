from GUI.Design_ui import *
from SOURCE.Stereo_Calibrator import Stereo_Calibrator
from SOURCE.Depth_Mapper import Depth_Mapper
from SOURCE.Sound_Echo_and_Depth_Mapper import Sound_Echo_and_Depth_Mapper
import os
import sys
import numpy as np
import cv2
import logging
import glob
import matplotlib as plt

try:
    import pyrealsense2 as rs
    disable_realsense_cam = False
except:
    disable_realsense_cam = True

# pyuic5 -x Design.ui -o Design_ui.py
# sudo modprobe v4l2loopback
# v4l2-ctl --list-devices



"""
# To be able to log into the QPlainTextEdit widget directly
# Very simple and beautiful way but it is blocking
class QPlainTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()

        self.widget = parent
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)
"""

class QPlainTextEditLogger_NonBlockong(logging.Handler, QtCore.QObject):
    sigLog = QtCore.Signal(str)
    def __init__(self, widget):
        logging.Handler.__init__(self)
        QtCore.QObject.__init__(self)
        self.widget=widget
        self.widget.setReadOnly(True)

        self.sigLog.connect(self.widget.appendPlainText)

    def emit(self, logRecord):
        message = str(logRecord.getMessage())
        self.sigLog.emit(message)

class Worker(QtCore.QThread):
    def __init__(self, func, args):
        super(Worker, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args) # si pones *self.args se desacoplan todos los argumentos


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    # Create a progress bar updater signal
    barUpdate_Calibrate = QtCore.Signal(int)
    barUpdate_Life = QtCore.Signal(int)

    # Create the cv2 plotter signal given : This is necessary to do it because gui stuff (like the
    # cv2 calls to qt) cnanot be handled from secondary threads!!! In a blocking non-threaded
    # version of the code this was not at all a problem, but here is indeed
    plotter_cv2 = QtCore.Signal(np.ndarray, int, str)
    # should be the array to plot, an int with the time to waitKey and a string with the label to show

    # signal to prompt the user from the child thread for approval
    approval = QtCore.Signal()


    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        # and connect it to the progress bars # asi ahora si se llama self.barUpdate_Calibrate.emit(10) ba yasta en thread save mode
        self.barUpdate_Calibrate.connect(self.progressBar_Calibrate.setValue)
        self.barUpdate_Life.connect(self.progressBar_Life.setValue)

        # connect the signal to the plotting cv2 function
        self.plotter_cv2.connect(self.show_cv2_image, type=QtCore.Qt.BlockingQueuedConnection)

        # connect the signal to the approval prompt box
        self.approval.connect(self.clickToPhotos, type=QtCore.Qt.BlockingQueuedConnection)

        # set the working directory
        self.working_directory.setText(os.getcwd())

        # Set up logging to use your widget as a handler
        log_handler = QPlainTextEditLogger_NonBlockong(self.log_text)
        # You can format what is printed to text box
        # connect with logger
        logging.getLogger().addHandler(log_handler)
        # You can control the logging level
        logging.getLogger().setLevel(logging.DEBUG)

        # Generate a code signaler
        #self.waitingPhotos = QtCore.pyqtSignal()

        # disable realsese camera if not installed
        if disable_realsense_cam:
            self.use_realsense_cameras.setChecked(False)
            self.use_realsense_cameras.setEnabled(False)
            self.use_standard_cameras.setChecked(True)

        # We connect the events with their actions
        self.start_calibration.clicked.connect(self.executeCalibration)
        self.start_take.clicked.connect(self.executeLifeTake)
        self.run_echo_depth.clicked.connect(self.executeEchoDepthTake)
        self.run_selection_echo_depth.clicked.connect(self.selectEchoDepth)

        # when user clicks to choose differnt paths
        self.change_stereo_calib_path.clicked.connect(lambda:
                    self.choose_file("Choose StereoCalib parameter file",
                        "Stereo_Calibrate_rms_KL_DL_KR_DR_R_T_E_F.npz",
                        "Numpy array container (*.npz)", self.stereo_calibrate_param_path))
        self.change_stereo_rectify_path.clicked.connect(lambda:
                    self.choose_file("Choose StereoRectify parameter file",
                        "Stereo_Rectify_R1_R2_P1_P2_Q_roi_left_roi_right.npz",
                        "Numpy array container (*.npz)", self.stereo_rectify_param_path))
        self.change_stereo_calib_path_echo_depth.clicked.connect(lambda:
                    self.choose_file("Choose StereoCalib parameter file",
                        "Stereo_Calibrate_rms_KL_DL_KR_DR_R_T_E_F.npz",
                        "Numpy array container (*.npz)", self.stereo_calibrate_param_path))
        self.change_stereo_rectify_path_echo_depth.clicked.connect(lambda:
                    self.choose_file("Choose StereoRectify parameter file",
                        "Stereo_Rectify_R1_R2_P1_P2_Q_roi_left_roi_right.npz",
                        "Numpy array container (*.npz)", self.stereo_rectify_param_path))
        self.change_working_directory.clicked.connect(lambda:
                    self.choose_directory("Choose Working Directory", self.working_directory))

        # Set default seterocalibration and recification parameter file paths
        self.stereo_calibrate_param_path.setText( "./OUTPUTS/CALIBRATION/COMMON/Calibrated_Parameters/Stereo_Calibrate_rms_KL_DL_KR_DR_R_T_E_F.npz" )
        self.stereo_rectify_param_path.setText( "./OUTPUTS/CALIBRATION/COMMON/Calibrated_Parameters/Stereo_Rectify_R1_R2_P1_P2_Q_roi_left_roi_right.npz" )
        self.stereo_calibrate_param_path_echo_depth.setText( "./OUTPUTS/CALIBRATION/COMMON/Calibrated_Parameters/Stereo_Calibrate_rms_KL_DL_KR_DR_R_T_E_F.npz" )
        self.stereo_rectify_param_path_echo_depth.setText( "./OUTPUTS/CALIBRATION/COMMON/Calibrated_Parameters/Stereo_Rectify_R1_R2_P1_P2_Q_roi_left_roi_right.npz" )
        self.echo_depth_output_path.setText("./OUTPUTS")

        # create the workers for the methods
        self.calibrator_worker = Worker( self._executeCalibration_Pipeline, ())
        self.calibrator_worker.finished.connect(lambda: self.setUserInteraction(True))

        self.life_take_worker = Worker( self._executeLifeTake_Pipeline, ())
        self.life_take_worker.finished.connect(lambda: self.setUserInteraction(True))
        #self.calibrator_worker.terminated.connect(lambda: self.setUserInteraction(True))

        self.echo_depth_worker = Worker( self._executeEchoDepthTake_Pipeline, ())
        self.echo_depth_worker.finished.connect(lambda: self.setUserInteraction(True))

    def show_cv2_image(self, image_array, t, label):
        cv2.imshow(label, image_array)
        ok = cv2.waitKey(t)
        cv2.destroyAllWindows()
        return ok


    def choose_file(self, label, guess, extension, display_widget):
        """
            Prompts the user to choose a file, this path will be saved in the text of
            the display_widget.
        """
        filepath = QtWidgets.QFileDialog.getOpenFileName(self, label, guess, extension)
        display_widget.setText(filepath[0])

    def choose_directory(self, label, display_widget):
        """
            Prompts the user to choose a directory, this path will be saved in the text of
            the display_widget.
        """
        folderpath = QtWidgets.QFileDialog.getExistingDirectory(self, label)
        display_widget.setText(folderpath)


    def clickToPhotos(self):
        ret = QtWidgets.QMessageBox.question(self, 'Take Photos', "Press OK when\n you are ready\n to take photos", QtWidgets.QMessageBox.Ok)
        return ret

    def setUserInteraction(self, state):
        self.start_take.setEnabled(state)
        self.start_calibration.setEnabled(state)
        self.run_echo_depth.setEnabled(state)
        self.run_selection_echo_depth.setEnabled(state)


    def executeCalibration(self):
        # block the button
        self.setUserInteraction(False)
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
        user_defined_parameters["block_size"]= int(self.block_size.text())
        user_defined_parameters["min_disp"]= int(self.min_disp.text())
        user_defined_parameters["num_disp"]= int(self.num_disp.text())
        user_defined_parameters["uniquenessRatio"]= int(self.uniquenessRatio.text())
        user_defined_parameters["speckleWindowSize"]= int(self.speckleWindowSize.text())
        user_defined_parameters["disp12MaxDiff"]= int(self.disp12MaxDiff.text())
        user_defined_parameters["speckleRange"]= int(self.speckleRange.text())
        user_defined_parameters["lmbda"]= int(self.lmbda.text())
        user_defined_parameters["sigma"]= float(self.sigma.text())
        user_defined_parameters["visual_multiplier"]= int(self.visual_multiplier.text())
        user_defined_parameters["is_realsense"] = self.use_realsense_cameras.isChecked()
        user_defined_parameters["width"] = int(self.width.text())
        user_defined_parameters["height"] = int(self.height.text())
        user_defined_parameters["laser"]=self.laser.isChecked()

        self.calibrator_worker.args=(user_defined_parameters,)
        self.calibrator_worker.start()

        """
        self.bee.finished.connect(self.restoreUi)
        self.bee.terminated.connect(self.restoreUi)
        """

    def _executeCalibration_Pipeline(self, user_defined_parameters):
        # We initialize an instance of the Setero_Calibrator object
        logging.info("Executing Calibration...")
        stereo_Calibrator = Stereo_Calibrator( user_defined_parameters, self.plotter_cv2 )

        # We execute all the pipeline, notifying the user in between
        os.chdir(user_defined_parameters["working_directory"])
        self.barUpdate_Calibrate.emit(1)

        if (not self.use_taken_photos_test.isChecked()) or (not self.use_taken_photos.isChecked()): # only then need for cameras
            ret = stereo_Calibrator.setCameras(int(self.cam_L_idx.value()),
                                                int(self.cam_R_idx.value())) #%5
            if (ret==1):
                logging.error("\nTry readjusting cameras and Start Calibration Again!")
                self.barUpdate_Calibrate.emit(0)
                return 1

        self.barUpdate_Calibrate.emit(5)

        ret = stereo_Calibrator.clean_directories_build_new(self.remove_old_data.isChecked()) #%6
        if ret==1:
            logging.error("\n[ERROR] If old data is to use, there should exist an ./OUTPUT directory in working directory!")
            self.barUpdate_Calibrate.emit(0)
            return 1
        self.barUpdate_Calibrate.emit(6)

        if self.use_taken_photos.isChecked():
            stereo_Calibrator.use_given_photos_compute_points()
        else:
            self.approval.emit()
            stereo_Calibrator.take_chess_photos_compute_points() #%15
            self.barUpdate_Calibrate.emit(15)

        stereo_Calibrator.calibrate_cameras() #%25
        self.barUpdate_Calibrate.emit(15)

        stereo_Calibrator.compute_Fundamental_Matrix() #%40
        self.barUpdate_Calibrate.emit(40)

        stereo_Calibrator.draw_Epilines() #%55
        self.barUpdate_Calibrate.emit(55)

        stereo_Calibrator.compute_Stereo_Rectification() # %65
        self.barUpdate_Calibrate.emit(65)

        stereo_Calibrator.rectify_chess() #%75
        self.barUpdate_Calibrate.emit(75)

        stereo_Calibrator.compute_Disparity_Map() #%85
        self.barUpdate_Calibrate.emit(85)

        self.approval.emit()
        stereo_Calibrator.do_Test(self.use_taken_photos_test.isChecked()) #%100
        self.barUpdate_Calibrate.emit(100)

        logging.info("\nCALIBRATION FINISHED!!!\n")
        # reset the button

    def executeLifeTake(self):
        self.setUserInteraction(False)

        user_defined_parameters={}
        user_defined_parameters["working_directory"] = self.working_directory.text()
        user_defined_parameters["stereo_calibrate_param_path"] = self.stereo_calibrate_param_path.text()
        user_defined_parameters["stereo_rectify_param_path"] = self.stereo_rectify_param_path.text()
        user_defined_parameters["frame_number"] = int(self.frame_number.text())
        user_defined_parameters["time_between_frames"] = float(self.time_between_frames.text())
        user_defined_parameters["show_life_frames"] = self.show_life_frames.isChecked()
        user_defined_parameters["use_taken_photos_life"] = self.use_taken_photos_life.isChecked()
        user_defined_parameters["block_size"]= int(self.block_size.text())
        user_defined_parameters["min_disp"]= int(self.min_disp.text())
        user_defined_parameters["num_disp"]= int(self.num_disp.text())
        user_defined_parameters["uniquenessRatio"]= int(self.uniquenessRatio.text())
        user_defined_parameters["speckleWindowSize"]= int(self.speckleWindowSize.text())
        user_defined_parameters["disp12MaxDiff"]= int(self.disp12MaxDiff.text())
        user_defined_parameters["speckleRange"]= int(self.speckleRange.text())
        user_defined_parameters["lmbda"]= int(self.lmbda.text())
        user_defined_parameters["sigma"]= float(self.sigma.text())
        user_defined_parameters["visual_multiplier"]= int(self.visual_multiplier.text())
        user_defined_parameters["is_realsense"] = self.use_realsense_cameras.isChecked()
        user_defined_parameters["width"] = int(self.width.text())
        user_defined_parameters["height"] = int(self.height.text())
        user_defined_parameters["laser"]=self.laser.isChecked()
        
        self.life_take_worker.args=(user_defined_parameters,)
        self.life_take_worker.start()


    def _executeLifeTake_Pipeline(self, user_defined_parameters):
        depth_mapper = Depth_Mapper(user_defined_parameters, self.plotter_cv2)

        # We execute all the pipeline, notifying the user in between
        os.chdir(user_defined_parameters["working_directory"])
        self.barUpdate_Life.emit(1)
        if not self.use_taken_photos_life.isChecked(): # only then it is necesary a camera
            ret = depth_mapper.setCameras(int(self.cam_L_idx.value()),
                                                int(self.cam_R_idx.value())) #%5
            if (ret==1):
                logging.error("\nTry readjusting cameras and Start Calibration Again!")
                self.barUpdate_Life.emit(0)
                return 1

        self.barUpdate_Life.emit(10)

        ret = depth_mapper.clean_directories_build_new(self.remove_old_data_life.isChecked()) #%6
        if ret==1:
            logging.error("\n[ERROR] If old data is to use, there should exist an ./OUTPUT directory in working directory!")
            self.barUpdate_Life.emit(0)
            return 1
        self.barUpdate_Life.emit(15)

        depth_mapper.compute_Disparity_Maps()
        self.barUpdate_Life.emit(100)


    def executeEchoDepthTake(self):
        self.setUserInteraction(False)

        user_defined_parameters={}
        user_defined_parameters["working_directory"] = self.working_directory.text()
        user_defined_parameters["stereo_calibrate_param_path_echo_depth"] = self.stereo_calibrate_param_path_echo_depth.text()
        user_defined_parameters["stereo_rectify_param_path_echo_depth"] = self.stereo_rectify_param_path_echo_depth.text()

        user_defined_parameters["sample_num_echo_depth"]=int(self.sample_num_echo_depth.text())
        user_defined_parameters["time_between_shots"]=float(self.time_between_shots.text())
        user_defined_parameters["echo_depth_output_path"]=self.echo_depth_output_path.text()
        user_defined_parameters["frequency"]=float(self.frequency.text())
        user_defined_parameters["pulse_duration"]=float(self.pulse_duration.text())
        user_defined_parameters["broadcast_sample_rate"]=int(self.broadcast_sample_rate.text())
        user_defined_parameters["envelope_std"]=float(self.envelope_std.text())
        user_defined_parameters["gaussian_envelope"]=self.gaussian_envelope.isChecked()
        user_defined_parameters["recording_sample_rate"]=int(self.recording_sample_rate.text())
        user_defined_parameters["recording_duration"]=float(self.recording_duration.text())
        user_defined_parameters["use_average_image"]=self.use_average_image.isChecked()
        user_defined_parameters["delay_broadcast_recording"]=float(self.delay_broadcast_recording.text())

        user_defined_parameters["block_size"]= int(self.block_size.text())
        user_defined_parameters["min_disp"]= int(self.min_disp.text())
        user_defined_parameters["num_disp"]= int(self.num_disp.text())
        user_defined_parameters["uniquenessRatio"]= int(self.uniquenessRatio.text())
        user_defined_parameters["speckleWindowSize"]= int(self.speckleWindowSize.text())
        user_defined_parameters["disp12MaxDiff"]= int(self.disp12MaxDiff.text())
        user_defined_parameters["speckleRange"]= int(self.speckleRange.text())
        user_defined_parameters["lmbda"]= int(self.lmbda.text())
        user_defined_parameters["sigma"]= float(self.sigma.text())
        user_defined_parameters["visual_multiplier"]= int(self.visual_multiplier.text())

        self.echo_depth_worker.args=(user_defined_parameters,)
        self.echo_depth_worker.start()


    def _executeEchoDepthTake_Pipeline(self, user_defined_parameters):
        depth_mapper = Sound_Echo_and_Depth_Mapper(user_defined_parameters, self.plotter_cv2)

        # We execute all the pipeline, notifying the user in between
        os.chdir(user_defined_parameters["working_directory"])
        self.barUpdate_Life.emit(1)
        ret = depth_mapper.setCameras(int(self.cam_L_idx.value()),
                                            int(self.cam_R_idx.value())) #%5
        if (ret==1):
            logging.error("\nTry readjusting cameras and Start Calibration Again!")
            self.barUpdate_Life.emit(0)
            #return 1

        self.barUpdate_Life.emit(10)

        ret = depth_mapper.clean_directories_build_new(self.erase_old_data_echo_depth.isChecked())

        if ret==1:
            logging.error("\n[ERROR] If old data is to use, there should exist an ./OUTPUT directory in working directory!")
            self.barUpdate_Life.emit(0)
            return 1
        self.barUpdate_Life.emit(15)

        depth_mapper.take_samples()
        self.barUpdate_Life.emit(100)

    def selectEchoDepth(self):
        self.depthMaps=sorted(glob.glob(f"{self.echo_depth_output_path}/SOUND_ECHO_and_DEPTH/DEPTH_MAPS/FILTERED_DEPTH_MAPS/*.png"))
        self.audios=sorted(glob.glob(f"{self.echo_depth_output_path}/SOUND_ECHO_and_DEPTH/SOUND_RECORDINGS/*.wav"))


if __name__ == "__main__":

    # Initialize and execute app
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
