import camera
import database
import filesystem
import analyzer

if __name__ == "__main__":
    # ==============================
    # TIMING LOOP
    # ==============================

    # ==============================
    # CAMERA OPERATIONS
    # ==============================

    print ("[CAMERA] initializing camera..")
    camera_device = camera.initialize_camera()

    # calculate gain
    print ("[CAMERA] starting calibration..")
    calibration = camera.calibrate_gain(camera_device)
    print("[CAMERA] calibration values ", calibration)

    # capture n images every x frame
    print ("[CAMERA] capturing frame sequence..")
    frame_sequence, step = camera.capture_n_images (camera_device, step=2)
    last_image = frame_sequence[-1]

    print ("[CAMERA] done.")
    camera_device.release()

    # ==============================
    # ARCHIVE IMAGE
    # ==============================

    # save test image last
    print ("[FILE SYSTEM] saving last image..")
    image_path = filesystem.save_image_in_archive(last_image)

    # ==============================
    # MAKE OBSERVATION
    # ==============================
    
    single_image_analysis = analyzer.analyze_single_image(last_image)
    print("[ANALYZER] single image analysis: ", single_image_analysis)

    # ==============================
    # SAVE DATA
    # ==============================

    # SAVE OBSERVATION IN DB
    print ("[DATABASE] saving observation in DB..")

    last_observation = database.save_observation(
    image_path=image_path,
    camera_settings=calibration,
    analysis=single_image_analysis,
    analysis_version="V1",
    status="ok",
    error=None,
    timestamp=None)

    # retrieve observation from db
    print ("[DATABASE] retrieving observation from DB..")

    retrieved_observation = database.get_observation(last_observation["observation_id"])
    print (retrieved_observation)
