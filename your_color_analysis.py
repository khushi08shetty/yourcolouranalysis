
def analyze_image(image_path):
    import cv2
    # Load the image
    image = cv2.imread(image_path)
    # Your color nalysis logic here
    # Example output (replace this with your actual analysis)
#!/usr/bin/env python
# coding: utf-8
# In[1]:
# Cell 1: Import and define the handler class

    import warnings

    warnings.filterwarnings('ignore')

    import google.generativeai as genai

    class GeminiPromptHandler:
        def __init__(self, gemini_api_key: str):
            """
            Initialize the handler with Gemini API credentials.
            
            Args:
                gemini_api_key (str): Your Gemini API key
            """
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('gemini-pro')  # Replace with actual model name if different
            
        def send_prompt(self, prompt: str) -> str:
            """
            Send a prompt to Gemini and retrieve the response.
            
            Args:
                prompt (str): The prompt text to send
                
            Returns:
                str: The response text from Gemini
            """
            response = self.model.generate_content(prompt)
            return response.text


    # In[2]:


    # Cell 2: Initialize the handler

    # Replace with your actual API key
    GEMINI_API_KEY = "AIzaSyAeOoiNTkK6ZCu5ooYOnoCzLdpc-hHz3Yk"
    handler = GeminiPromptHandler(GEMINI_API_KEY)

    # In[4]:
    import cv2
    import mediapipe as mp
    import numpy as np

    # Initialize MediaPipe Face Mesh and Selfie Segmentation
    mp_face_mesh = mp.solutions.face_mesh
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    # Load image and convert to RGB
    # image_path = 'image_6209779.JPG'
    # image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define indices of face contour landmarks in MediaPipe Face Mesh
    face_contour_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67, 109
    ]

    # Process with Face Mesh and Selfie Segmentation
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:

            # Get the selfie mask from MediaPipe Selfie Segmentation
            segmentation_results = selfie_segmentation.process(rgb_image)
            selfie_mask = segmentation_results.segmentation_mask > 0.5  # True for foreground, False for background

            # Initialize a blank mask for the face
            face_mask = np.zeros_like(selfie_mask, dtype=np.uint8)

            # Get face landmarks
            face_results = face_mesh.process(rgb_image)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Extract only the face border points using the specified indices
                    face_border_points = [
                        (int(face_landmarks.landmark[i].x * image.shape[1]), int(face_landmarks.landmark[i].y * image.shape[0]))
                        for i in face_contour_indices
                    ]
                    # Create a face mask using the border points
                    cv2.fillConvexPoly(face_mask, np.array(face_border_points, dtype=np.int32), 255)

            # Convert boolean mask to uint8 format
            selfie_mask_uint8 = selfie_mask.astype(np.uint8) * 255

            # Mask out the background from the original image using the selfie mask
            foreground = cv2.bitwise_and(image, image, mask=selfie_mask_uint8)

            # Mask out the face region from the selfie mask to isolate hair
            hair = cv2.bitwise_and(foreground, foreground, mask=cv2.bitwise_not(face_mask))

            # Crop the hair mask to the top 40% of the image
            height, width, _ = hair.shape
            hair = hair[:int(0.2 * height), :, :]

            # Function to isolate features with transparent background
            def isolate_feature_with_transparency(indices):
                points = np.array([(int(face_landmarks.landmark[i].x * image.shape[1]),
                                    int(face_landmarks.landmark[i].y * image.shape[0])) for i in indices])
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, points, 255)
                # Extract feature and add alpha channel
                feature = cv2.bitwise_and(image, image, mask=mask)
                feature = cv2.cvtColor(feature, cv2.COLOR_BGR2BGRA)
                feature[:, :, 3] = mask  # Set alpha channel based on mask
                return feature

            # Define feature landmark indices
            left_eye_indices = [468,145,471,159]
            right_eye_indices = [473, 386, 474, 374]
            lip_indices = [181, 180, 178, 14, 317, 402, 403, 404, 405, 314, 17, 84]
            cheek_index = [123,117,119,142,205]

            # Isolate features with transparency
            left_eye = isolate_feature_with_transparency(left_eye_indices)
            right_eye = isolate_feature_with_transparency(right_eye_indices)
            lips = isolate_feature_with_transparency(lip_indices)
            cheek = isolate_feature_with_transparency(cheek_index)

            # Convert hair to BGRA and add transparency
            hair = cv2.cvtColor(hair, cv2.COLOR_BGR2BGRA)
            hair[:, :, 3] = np.where(hair[:, :, :3].any(axis=2), 255, 0)  # Set alpha to 255 for non-black pixels

            # Save isolated features with transparency
            cv2.imwrite("left_eye_transparent.png", left_eye)
            cv2.imwrite("right_eye_transparent.png", right_eye)
            cv2.imwrite("lips_transparent.png", lips)
            cv2.imwrite("cheek_transparent.png", cheek)
            cv2.imwrite("hair_transparent.png", hair)

    cv2.destroyAllWindows()


    # In[ ]:


    import cv2
    import pandas as pd
    import numpy as np

    # Load the colors.csv file, assuming it has columns: 'colour_name', 'hex', 'r', 'g', 'b'
    colors_df = pd.read_csv('colors.csv')

    # Function to crop the non-transparent region and return the RGB color of the center pixel
    def get_center_color(image_path):
        # Load the image with an alpha channel (transparency)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 4:  # Check for alpha channel
            alpha_channel = image[:, :, 3]
            y_coords, x_coords = np.where(alpha_channel > 0)
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)

            # Crop the bounding box region and get the center pixel's RGB color
            cropped_image = image[min_y:max_y+1, min_x:max_x+1, :3]  # Only RGB channels
            center_y, center_x = cropped_image.shape[0] // 2, cropped_image.shape[1] // 2
            center_rgb = cropped_image[center_y, center_x]  # Center pixel RGB color
            return center_rgb
        else:
            print(f"No alpha channel found in {image_path}")
            return None

    # Function to find the closest color in the CSV based on Euclidean distance
    def get_closest_color_name(center_rgb):
        distances = np.sqrt((colors_df["r"] - center_rgb[2]) ** 2 +
                            (colors_df["g"] - center_rgb[1]) ** 2 +
                            (colors_df["b"] - center_rgb[0]) ** 2)
        closest_index = distances.idxmin()
        return colors_df.loc[closest_index, "hex"]

    # Paths to each feature's isolated image
    features = {
        "lips": "lips_transparent.png",
        "left_eye": "left_eye_transparent.png",
        "right_eye": "right_eye_transparent.png",
        "cheek": "cheek_transparent.png",
        "hair": "hair_transparent.png"
    }

    # Process each feature and print the hex color
    output_colors = {}
    for feature, path in features.items():
        center_rgb = get_center_color(path)
        if center_rgb is not None:
            hex_color = get_closest_color_name(center_rgb)
            output_colors[feature] = hex_color

    # Print final output
    for feature, hex_color in output_colors.items():
        print(f"{feature}: {hex_color}")


    # In[3]:


    # Cell 3: Define and format the prompt dynamically with extracted hex codes

    # Ensure 'output_colors' contains hex codes for 'lips', 'left_eye', 'right_eye', 'cheek', and 'hair'
    lips_color = output_colors.get("lips", "#dea5a4")  # Default to #dea5a4 if not found
    left_eye_color = output_colors.get("left_eye", "#222222")
    right_eye_color = output_colors.get("right_eye", "#222222")
    cheek_color = output_colors.get("cheek", "#dea5a4")  # Default to skin tone color if not found
    hair_color = output_colors.get("hair", "#282828")

    # Format the prompt with extracted colors
    prompt = f"""
    My lips color is: {lips_color}, 
    eye colors are: {left_eye_color} (left) and {right_eye_color} (right), 
    skin tone is: {cheek_color}, 
    and hair color is: {hair_color}. 
    Please do a detailed color analysis and provide insights on:
    1. Suitable color palettes for me.
    2. Colors I should avoid.
    3. Makeup styles that would complement my features.
    """

    # Send the prompt to Gemini and retrieve the response
    response_text = handler.send_prompt(prompt)
    # print(response_text)

    import re
    def format_text(response_text):
        # Step 1: Add a newline before any instance of ' *' or ' **'
        response_text = re.sub(r' \*', '\n', response_text)
        response_text = re.sub(r' \*\*', '\n', response_text)

        # Step 2: Remove any remaining '*' characters
        response_text = response_text.replace('*', '')

        return response_text

    formatted_text = format_text(response_text)

    result = formatted_text

    return result

