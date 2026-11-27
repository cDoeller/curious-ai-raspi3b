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
    camera_settings = camera.calibrate_gain(camera_device)
    print("[CAMERA] calibration values ", camera_settings)

    # capture n images every x frame
    print ("[CAMERA] capturing frame sequence..")
    frame_sequence, step = camera.capture_n_images (camera_device)
    
    last_image = frame_sequence[-1]

    # add step to camera data
    camera_settings["frame_interval_sequence"] = step

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
    
    overall_analysis = analyzer.analyze_single_image(last_image)
    print("[MAIN] [ANALYZER] overall analysis: ", overall_analysis)

    spatial_analysis = analyzer.analyze_n_fields(last_image)
    print("[MAIN] [ANALYZER] n fields analysis: ", spatial_analysis)

    motion_analysis = analyzer.analyze_movement (frame_sequence, True, True)
    print("[MAIN] [ANALYZER] motion analysis: ", motion_analysis)

    # ==============================
    # SAVE DATA
    # ==============================

    # SAVE OBSERVATION IN DB
    print ("[DATABASE] saving observation in DB..")

    last_observation = database.save_observation(
    image_path=image_path,
    camera_settings=camera_settings,
    analysis={"overall_analysis": overall_analysis,
              "spatial_analysis": spatial_analysis,
              "motion_analysis": motion_analysis},
    analysis_version="V1",
    status="ok",
    error=None,
    timestamp=None)

    # retrieve observation from db
    # print ("[DATABASE] retrieving observation from DB..")

    # retrieved_observation = database.get_observation(last_observation["observation_id"])
    # print (retrieved_observation)
