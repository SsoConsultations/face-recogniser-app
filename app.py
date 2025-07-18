import streamlit as st
import face_recognition
import os
import numpy as np
import cv2
from PIL import Image # Import Pillow for image manipulation

# --- Configuration (relative path for deployment) ---
KNOWN_FACES_DIR = 'known_faces'

# --- Data Storage (use st.cache_resource for efficiency) ---
# Add a parameter to force reload the cache
@st.cache_resource
def load_known_faces(known_faces_dir, _=None): # Added _=None to allow manual cache invalidation
    st.info("Loading known faces... This might take a moment.")
    
    # Declare global variables here to modify the module-level lists
    global known_face_encodings, known_face_names
    
    # Initialize lists (important if cache is cleared or on first run)
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir) # Create the directory if it doesn't exist
        st.warning(f"'{known_faces_dir}' directory created. Please add face images.")
        return [], [] # Return empty lists if directory was just created

    # Iterate through subdirectories for each person
    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_dir, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_locations = face_recognition.face_locations(image)
                        face_encodings = face_recognition.face_encodings(image, face_locations)

                        if face_encodings:
                            known_face_encodings.append(face_encodings[0])
                            known_face_names.append(name)
                        # else:
                        #     st.warning(f"No face found in {filename} for {name}. Skipping.")
                    except Exception as e:
                        st.error(f"Error processing {filename} for {name}: {e}")
                        pass # Continue to next file even if one fails
    st.success(f"Finished loading known faces. Total known faces: {len(known_face_encodings)}")
    return known_face_encodings, known_face_names

# Initialize global variables at module level
known_face_encodings = []
known_face_names = []

# Load faces once when the app starts or is re-run due to cache invalidation
# The initial call to load_known_faces will populate the global lists.
known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)


# --- Function to process an image and draw boxes (reusable) ---
def process_frame_for_faces(frame_rgb, known_encodings, known_names):
    frame_rgb = np.copy(frame_rgb)
    
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if not face_locations:
        return frame_bgr # Return original frame if no faces found

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        if known_encodings: # Only compare if there are known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
            else:
                # Optional: If no exact match, consider the closest match if within a threshold
                if face_distances[best_match_index] < 0.6: # Adjust threshold as needed
                    name = known_names[best_match_index]

        # Drawing rectangles and labels
        box_padding = 15
        base_label_height = 25
        text_y_offset = 10
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        font_thickness = 1

        top_ext = max(0, top - box_padding)
        right_ext = min(frame_bgr.shape[1], right + box_padding)
        bottom_ext = min(frame_bgr.shape[0], bottom + box_padding)
        left_ext = max(0, left - box_padding)

        cv2.rectangle(frame_bgr, (left_ext, top_ext), (right_ext, bottom_ext), (0, 255, 0), 2)

        (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)
        label_width = text_width + (box_padding * 2)
        label_height = max(base_label_height, text_height + (text_y_offset * 2))

        label_top = bottom_ext
        label_bottom = label_top + label_height
        label_left = left_ext
        label_right = max(right_ext, left_ext + label_width) # Ensure label is at least as wide as box

        # Adjust label position if it goes out of bounds
        if label_bottom > frame_bgr.shape[0]:
            label_bottom = frame_bgr.shape[0]
            label_top = label_bottom - label_height
            if label_top < 0: # If still goes above, set to 0
                label_top = 0

        if label_right > frame_bgr.shape[1]:
            label_right = frame_bgr.shape[1]
            label_left = max(0, label_right - label_width) # Ensure label is not out of bounds left

        cv2.rectangle(frame_bgr, (label_left, label_top), (label_right, label_bottom), (0, 255, 0), cv2.FILLED)

        text_x = label_left + (label_right - label_left - text_width) // 2
        text_y = label_top + (label_height + text_height) // 2 - baseline

        cv2.putText(frame_bgr, name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return frame_bgr

# --- Streamlit UI ---
st.set_page_config(page_title="Dynamic Face Recognition App", layout="centered")

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home' # 'home', 'user_login', 'admin_login'

# --- Home Page ---
if st.session_state.page == 'home':
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        try:
            # Ensure 'image_f2baca.png' is in the same directory as your app.py
            st.image("sso_logo.jpg", width=300) 
        except FileNotFoundError:
            st.warning("Logo image 'sso_logo.jpg' not found. Please ensure it's in the same directory.")
            st.markdown("## SSO Consultants")

    st.markdown("<h2 style='text-align: center;'>SSO Consultants Face Recogniser 🕵️‍♂️</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Please choose your login type.</h3>", unsafe_allow_html=True)

    col1_btn, col2_btn, col3_btn, col4_btn = st.columns([1, 0.7, 0.7, 1])

    with col2_btn:
        if st.button("Login as User", key="user_login_btn", help="Proceed to face recognition for users"):
            st.session_state.page = 'user_login'
            st.rerun()

    with col3_btn:
        if st.button("Login as Admin", key="admin_login_btn", help="Proceed to admin functionalities"):
            st.session_state.page = 'admin_login'
            st.rerun()

# --- User Login (Face Recognition) Page ---
elif st.session_state.page == 'user_login':
    st.title("Face Recognition App with Dynamic Labels 🕵️‍♂️")
    st.markdown("""
    This application performs face recognition from your live webcam or an uploaded image.
    The name labels will dynamically adjust their size to fit the recognized name!
    """)

    if not known_face_encodings:
        st.error("No known faces loaded. Please ensure your `known_faces` directory "
                 "is correctly structured and contains images with faces for training.")

    st.sidebar.header("Choose Input Method")
    option = st.sidebar.radio("", ("Live Webcam Recognition", "Upload Image for Recognition"), key="user_input_option")

    if option == "Live Webcam Recognition":
        st.subheader("Live Webcam Face Recognition")
        st.info("Allow camera access. Take a picture, and the app will detect/recognize faces.")

        camera_image = st.camera_input("Click 'Take Photo' to capture an image:", key="user_camera_input")

        if camera_image is not None:
            with st.spinner("Processing live image..."):
                file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                processed_img_bgr = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names)
                processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)

            st.image(processed_img_rgb, caption="Processed Live Image", use_column_width=True)
            st.success("Face detection and recognition complete!")
        else:
            st.warning("Waiting for webcam input. Click 'Take Photo' above.")

    elif option == "Upload Image for Recognition":
        st.subheader("Upload Image for Face Recognition")
        uploaded_file = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png", "bmp", "gif"], key="user_file_uploader")

        if uploaded_file is not None:
            with st.spinner("Loading and processing image..."):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                st.image(img_rgb, caption="Original Uploaded Image", use_column_width=True)

                processed_img_bgr = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names)
                processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)

            st.image(processed_img_rgb, caption="Processed Image with Faces", use_column_width=True)
            st.success("Face detection and recognition complete!")
        else:
            st.info("Please upload an image file using the browser button.")

    if st.button("⬅ Back to Home", key="user_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

# --- Admin Login Page (Placeholder) ---
elif st.session_state.page == 'admin_login':
    st.title("Admin Panel 🔒")
    st.markdown("This section is for **administrators** only.")

    admin_password = st.text_input("Enter Admin Password:", type="password", key="admin_pass_input")

    if admin_password == "admin123": # **IMPORTANT: Replace with a more secure authentication method for production!**
        st.success("Welcome, Admin!")

        st.subheader("Add New Faces to Database ➕")
        st.markdown("Upload an image of a person and provide a name for recognition.")

        new_face_name = st.text_input("Enter Name/Description for the Face:", key="new_face_name_input")
        new_face_image = st.file_uploader("Upload Image of New Face:", type=["jpg", "jpeg", "png"], key="new_face_image_uploader")

        if st.button("Add Face to Database", key="add_face_btn"):
            if new_face_name and new_face_image:
                person_dir = os.path.join(KNOWN_FACES_DIR, new_face_name.replace(" ", "_").lower()) # Create a clean directory name
                os.makedirs(person_dir, exist_ok=True) # Create directory if it doesn't exist

                # Save the image
                # Generate a unique filename to avoid overwriting
                image_filename = f"{new_face_name.replace(' ', '_').lower()}_{len(os.listdir(person_dir)) + 1}.jpg"
                image_path = os.path.join(person_dir, image_filename)
                
                # Use PIL to save the image to ensure consistent format
                img = Image.open(new_face_image).convert("RGB")
                img.save(image_path, "JPEG") # Save as JPEG

                st.info(f"Analyzing {new_face_name}'s image...")
                
                try:
                    # Verify face can be encoded
                    image_to_encode = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image_to_encode)
                    
                    if face_locations:
                        # Clear the cache for load_known_faces to force a reload
                        load_known_faces.clear()
                        
                        # Re-load known faces; this will update the global lists
                        # The 'global' keyword is NOT needed here because known_face_encodings
                        # and known_face_names are already global variables at the module level.
                        known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR, _=np.random.rand())
                        
                        st.success(f"Successfully added '{new_face_name}' to the known faces database! ✅")
                        st.rerun() # Rerun to refresh the UI and known faces list
                    else:
                        st.error(f"No face found in the uploaded image for '{new_face_name}'. Please upload an image with a clear face.")
                        # Clean up the empty directory or file if no face was found
                        if os.path.exists(image_path):
                            os.remove(image_path)
                        if not os.listdir(person_dir): # Remove directory if it's empty after failed add
                            os.rmdir(person_dir)

                except Exception as e:
                    st.error(f"Error processing image for '{new_face_name}': {e}")
                    # Clean up if an error occurred during processing
                    if os.path.exists(image_path):
                        os.remove(image_path)
                    if not os.listdir(person_dir): # Remove directory if it's empty after failed add
                        os.rmdir(person_dir)

            else:
                st.warning("Please provide both a name and upload an image.")

        st.subheader("Current Known Faces 📋")
        if known_face_names:
            # Display current known faces with names
            # Using sorted(set(...)) to display unique names alphabetically
            for name in sorted(set(known_face_names)): 
                st.write(f"- **{name}**")
        else:
            st.info("No faces currently registered in the database.")


    else:
        if admin_password: # Only show error if user actually typed something
            st.error("Incorrect password.")

    if st.button("⬅ Back to Home", key="admin_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

st.markdown("---")
st.markdown("Developed with ❤️ using `face_recognition`, `OpenCV`, and `Streamlit`.")
