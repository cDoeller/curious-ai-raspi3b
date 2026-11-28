from camera import initialize_camera, calibrate_gain, capture_n_images
from database import save_observation, get_observation
from filesystem import save_image_in_archive
from analyzer import analyze_single_image, analyze_n_fields, analyze_movement
from objectdetection import detect_objects

if __name__ == "__main__":
    # ==============================
    # TIMING LOOP
    # ==============================

    # ==============================
    # CAMERA OPERATIONS
    # ==============================

    print ("[CAMERA] initializing camera..")
    camera_device = initialize_camera()

    # calculate gain
    print ("[CAMERA] starting calibration..")
    camera_settings = calibrate_gain(camera_device)
    print("\n[CAMERA] calibration values ", camera_settings)

    # capture n images every x frame
    print ("\n[CAMERA] capturing frame sequence..")
    frame_sequence, step = capture_n_images (camera_device)
    
    last_image = frame_sequence[-1]

    # add step to camera data
    camera_settings["frame_step"] = step
    camera_settings["frame_sequence_length"] = len(frame_sequence)

    print ("[CAMERA] done.")
    camera_device.release()

    # ==============================
    # ARCHIVE IMAGE
    # ==============================

    # save test image last
    print ("[FILE SYSTEM] saving last image..")
    image_path = save_image_in_archive(last_image)

    # ==============================
    # MAKE OBSERVATION
    # ==============================
    
    overall_analysis = analyze_single_image(last_image)
    print("\n[MAIN] [ANALYZER] overall analysis: ", overall_analysis)

    spatial_analysis = analyze_n_fields(last_image)
    print("\n[MAIN] [ANALYZER] n fields analysis: ", spatial_analysis)

    motion_analysis = analyze_movement (frame_sequence, True, True)
    print("\n[MAIN] [ANALYZER] motion analysis: ", motion_analysis)

    detected_objects = detect_objects(last_image, True)
    print ("\n[MAIN] [OBJ DETECT] detections: ", detected_objects)

    # ==============================
    # SAVE DATA
    # ==============================

    # SAVE OBSERVATION IN DB
    print ("\n[DATABASE] saving observation in DB..")

    last_observation = save_observation(
    image_path=image_path,
    camera_settings=camera_settings,
    analysis={"overall_analysis": overall_analysis,
              "spatial_analysis": spatial_analysis,
              "motion_analysis": motion_analysis,
              "object_detection": detected_objects},
    analysis_version="V1",
    status="ok",
    error=None,
    timestamp=None)

    # #retrieve observation from db
    # print ("\n[DATABASE] retrieving observation from DB..")

    # retrieved_observation = get_observation(last_observation["observation_id"])
    # print (retrieved_observation)
