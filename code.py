import logging
import os
import pickle
from typing import List, Tuple, Optional

import cv2
import face_recognition
import numpy as np

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

"""
    METHODS:
    1. save_encodings
    2. read_encodings
    3. create_encodings
    4. compare_face_encodings
    5. save_image
    6. process_known_people
    7. process_dataset
"""


class FaceRecognitionSeparator:
    def __init__(self, tolerance: float):

        self.tolerance = tolerance
        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    def save_encodings(self, encodings: List[np.ndarray], names: List[str],
                       fname: str = "encodings.pickle") -> None:
        try:
            data = [{"name": name, "encoding": enc} for name, enc in zip(names, encodings)]

            logger.info(f"Serializing {len(data)} encodings to {fname}")
            with open(fname, "wb") as f:
                pickle.dump(data, f)

        except Exception as e:
            logger.error(f"Failed to save encodings: {str(e)}")
            raise

    def read_encodings(self, fname: str) -> Tuple[List[np.ndarray], List[str]]:
        try:
            with open(fname, "rb") as f:
                data = pickle.load(f)

            data = np.array(data)
            encodings = [d["encoding"] for d in data]
            names = [d["name"] for d in data]

            logger.info(f"Loaded {len(encodings)} encodings from {fname}")
            return encodings, names

        except Exception as e:
            logger.error(f"Failed to read encodings from {fname}: {str(e)}")
            raise

    def create_encodings(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple]]:

        try:
            # convert BGR to RGB (face_recognition expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations)

            return face_encodings, face_locations

        except Exception as e:
            logger.error(f"Failed to create encodings: {str(e)}")
            raise

    def compare_face_encodings(self, unknown_encoding: np.ndarray,
                               known_encodings: List[np.ndarray],
                               known_names: List[str]) -> Tuple[bool, Optional[str], float]:

        try:
            matches = face_recognition.compare_faces(known_encodings, unknown_encoding,
                                                     tolerance=self.tolerance)
            face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)

            if not any(matches):
                return False, None, 1.0

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                return True, known_names[best_match_index], face_distances[best_match_index]

            return False, None, face_distances[best_match_index]

        except Exception as e:
            logger.error(f"Face comparison failed: {str(e)}")
            raise

    def save_image(self, image: np.ndarray, category: str,
                   image_name: str, output_dir: str = "./output") -> None:
        try:
            path = os.path.join(output_dir, category)
            os.makedirs(path, exist_ok=True)

            output_path = os.path.join(path, image_name)
            cv2.imwrite(output_path, image)
            logger.debug(f"Saved image to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save image {image_name}: {str(e)}")
            raise

    def process_known_people(self, input_dir: str,
                             output_pickle: str = "./known_encodings.pickle") -> None:
        known_encodings = []
        known_names = []

        logger.info(f"Processing known people from {input_dir}")

        for img_name in os.listdir(input_dir):
            if not img_name.lower().endswith(self.valid_extensions):
                continue

            img_path = os.path.join(input_dir, img_name)
            name = os.path.splitext(img_name)[0]

            try:
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"Could not read image: {img_path}")
                    continue

                # resize for faster processing
                # image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

                encodings, locations = self.create_encodings(image)

                if not encodings:
                    logger.warning(f"No faces found in {img_path}")
                    continue

                if len(encodings) > 1:
                    logger.warning(f"Multiple faces found in {img_path}, using first face")

                known_encodings.append(encodings[0])
                known_names.append(name)

                logger.info(f"Processed {name}: Found {len(encodings)} face(s)")

            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue

        #checks if known_encodings.pickle is empty or not
        if known_encodings:
            self.save_encodings(known_encodings, known_names, output_pickle)
        else:
            logger.error("No valid face encodings were created")

    def process_dataset(self, dataset_dir: str, known_encodings_file: str) -> None:

        try:
            known_encodings, known_names = self.read_encodings(known_encodings_file)
            logger.info(f"Processing dataset from {dataset_dir}")

            for img_name in os.listdir(dataset_dir):
                if not img_name.lower().endswith(self.valid_extensions):
                    continue

                img_path = os.path.join(dataset_dir, img_name)

                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        logger.warning(f"Could not read image: {img_path}")
                        continue

                    # keep original for saving
                    original = image.copy()

                    # resize for faster processing
                    image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

                    encodings, locations = self.create_encodings(image)

                    # Handle group photos
                    if len(locations) > 1:
                        # self.save_image(original, "Group", img_name)
                        # logger.info(f"Saved group photo: {img_name}")

                        logger.info(f"Skipping group photo : {img_name}")

                    # Process each face in the image
                    matches_found = False
                    for encoding in encodings:
                        matched, name, confidence = self.compare_face_encodings(
                            encoding, known_encodings, known_names
                        )

                        if matched:
                            self.save_image(original, name, img_name)
                            matches_found = True
                            logger.info(f"Match found in {img_name}: {name} (confidence: {1 - confidence:.2f})")

                    if not matches_found:
                        # self.save_image(original, "Unknown", img_name)
                        logger.info(f"No matches found in {img_name}")

                except Exception as e:
                    logger.error(f"Error processing dataset image {img_path}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Dataset processing failed: {str(e)}")
            raise


def main():
    try:
        # configuration
        separator = FaceRecognitionSeparator(tolerance=0.4)
        dataset_path = "./dataset/"
        people_path = "./people/"
        known_encodings_file = "./known_encodings.pickle"

        # process known people first
        separator.process_known_people(people_path, known_encodings_file)

        # process dataset images
        separator.process_dataset(dataset_path, known_encodings_file)

        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
