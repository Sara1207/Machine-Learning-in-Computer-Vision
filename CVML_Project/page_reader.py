from typing import Union, List, Dict

from enum import Enum
import argparse
import numpy as np
import utils
import os
# Environment variables to disable parallel execution
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
import tensorflow as tf
from keras import layers, models, optimizers
from difflib import SequenceMatcher
os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)
tf.random.set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--example", default=None,
                    type=int, help="Example argument.")

# paths
big_letters_data_path = "emnist_data/emnist_train_big.npz"
digits_data_path = "emnist_data/emnist_train_num.npz"


class SolveEnum(Enum):
    CLASSIFIED_TEXT = 1
    MATCHED_PHRASES = 2
    ORDERED_COMMANDS = 3


class PageReader:

    def __init__(self, note: str) -> None:
        self._phrases = utils.Phrases()
        self.emnist_mapping = self.load_emnist_mapping("emnist_mapping.txt")
        self.emnist_big_letters = utils.EMnistDataset(
            data=big_letters_data_path)
        self.emnist_digits = utils.EMnistDataset(data=digits_data_path)
        self.character_model = self._create_cnn_model_letters()
        self.digit_model = self._create_cnn_model_digits()

    @staticmethod
    def load_emnist_mapping(file_path):
        mapping = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.split()
                mapping[int(key)] = int(value)
        return mapping
    
    # CNN model for the digits
    def _create_cnn_model_digits(self):
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # layers.Dropout(0.25),

            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # layers.Dropout(0.25),

            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # layers.Dropout(0.25),

            # Flattening and Fully Connected Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            # layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
        ])

        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model


    # CNN model for the letters
    def _create_cnn_model_letters(self):
        model = models.Sequential()

        # First Convolutional Block
        model.add(layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        # Second Convolutional Block
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        # Third Convolutional Block
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        # Flattening and Fully Connected Layers
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        # Output Layer
        model.add(layers.Dense(26, activation='softmax'))  # 26 classes for A-Z
        # The output is from 0 - 25

        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        return model


    def train_and_save_digit_model(self, digit_data, digit_labels):

        # Preprocess data
        digit_data = digit_data.reshape(
            (-1, 28, 28, 1)) / 255.0

        model = self._create_cnn_model_digits()

        # Train digit model
        model.fit(digit_data, digit_labels,epochs=10, batch_size=32)

        self.digit_model = model
        self.digit_model.save('digit_model.h5')

    def train_and_save_letter_model(self, character_data, character_labels):

        # Remap character labels from EMNIST mapping to 0-25 for A-Z
        # Beacuse the model for the letters returns from 0 - 25
        # I need the letters from 10 - 35 to remap in the range from 0 - 25 (10 -> 0, 11 -> 1,....)
        character_labels = np.array(
            [self.remap_label(label) for label in character_labels])

        # Preprocess data
        character_data = character_data.reshape(
            (-1, 28, 28, 1)) / 255.0

        model = self._create_cnn_model_letters()

        # Train character model
        model.fit(
            character_data, character_labels, epochs=10, batch_size=32)

        self.character_model = model
        self.character_model.save('character_model.h5')

    def remap_label(self, label):
        # Check if the label is for an uppercase letter
        #  if is from 10 to 35 then i need to substract 10
        if 10 <= label <= 35:
            # Adjust label index to be in the range 0-25 for uppercase letters A-Z
            return label - 10
        else:
            raise ValueError(
                f"Label {label} is outside the expected range for uppercase letters.")

    def segment_page(self, page):
        segmented_page = []

        # Segment the page horizontally (in rows)
        i = 19  
        while i < 581:  
            row = page[i:i + 28, :]
            if not np.all(row == 0):
                
                segmented_row = []
                word = []
                # Vertical segment
                j = 19  

                
                next_row_height = i + 28 + 10

                if next_row_height > 581:
                    space = 0  
                else:
                    space = 10  

                while j < 581:  
                    column = row[:,j:j + 28]

                    # Check if the segment is not blank
                    if not np.all(column == 0):
                        word.append(column)
                    else:
                        if word:
                            segmented_row.append(word)
                            word = []

                    j += 28 + space

                if word:  
                    segmented_row.append(word)

                segmented_page.append(segmented_row)

            i += 28 + 10

        return segmented_page

    def decode_prediction(self, label, is_digit):
        if is_digit:
            # For digits, label corresponds to the digit itself
            return str(label)
        else:
            # For characters, use the EMNIST mapping to get the corresponding character
            for key, value in self.emnist_mapping.items():
                if key == label + 10:  # For A-Z (this is because we substract 10, in order to fit the model, now we need again to add 10)
                    return chr(value)
        return "Unknown"  # Unrecognized label

    # Matching phrases
    def find_most_probable_phrase(self, text_line):

        most_probable_phrase = None
        highest_similarity = 0.0

        text_words = text_line.split()

        for phrase in self._phrases.phrases:
            phrase_words = phrase.split()
            if len(phrase_words) == len(text_words):

                # Calculate similarity based on individual word lengths
                similar_words = [
                    SequenceMatcher(None, text_word, phrase_word).ratio()
                    for text_word, phrase_word in zip(text_words, phrase_words)
                    if len(text_word) == len(phrase_word)
                ]

                # Sum the similarities for an overall score
                similarity = sum(similar_words)

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_probable_phrase = phrase

        return most_probable_phrase

    def fit(self, training: bool) -> None:

        if training:
            # Split datasets into training and testing subsets
            character_train_data, character_train_labels = self.emnist_big_letters.imgs, self.emnist_big_letters.labels
            digit_train_data, digit_train_labels = self.emnist_digits.imgs, self.emnist_digits.labels

            # Train and save the models
            self.train_and_save_digit_model(
                digit_train_data, digit_train_labels)
            self.train_and_save_letter_model(
                character_train_data, character_train_labels)
        else:
            # Load the models if training is False
            self.character_model = tf.keras.models.load_model(
                'character_model.h5')
            self.digit_model = tf.keras.models.load_model('digit_model.h5')

    def solve(self, pages: np.ndarray) -> Dict[str, Union[List[List[str]], List[str]]]:
        # Models
        character_model = self.character_model
        digit_model = self.digit_model
        # Phrases
        phrase_utils = self._phrases

        classified_text = []
        matched_phrases = []
        ordered_commands = []

        page_order = []
        temp_page_data = []

        for page in pages:
            segmented_page = self.segment_page(page)
            page_text = []
            page_phrases = []

            for row_index, row in enumerate(segmented_page):
                row_words = []
                is_last_row = (row_index == len(segmented_page) - 1)

                for word in row:
                    word_text = ""
                    for segment in word:
                        # Preprocess the segment
                        segment = segment.reshape(1, 28, 28, 1)/255.0

                        # Check if it is a digit
                        if is_last_row:
                            prediction = digit_model.predict(segment)
                            label = np.argmax(prediction)
                            decoded_character = self.decode_prediction(
                                label, is_digit=True)
                        else:
                            prediction = character_model.predict(segment)
                            label = np.argmax(prediction)
                            decoded_character = self.decode_prediction(
                                label, is_digit=False)
                        word_text += decoded_character
                    row_words.append(word_text)

                # we store the words(2 words in one row) with space between them
                line = " ".join(row_words)
                if is_last_row:
                    page_text.append(line)
                    page_order.append(int(line))
                else:
                    page_text.append(line)
                    matched_phrase = self.find_most_probable_phrase(line)
                    page_phrases.append(matched_phrase)

            temp_page_data.append((page_text, page_phrases))
            classified_text.append(page_text)
            matched_phrases.append(page_phrases)

        # Sort pages based on page numbers
        sorted_pages = [x for _, x in sorted(
            zip(page_order, temp_page_data), key=lambda pair: pair[0])]

        for page_text, page_phrases in sorted_pages:

            for phrase in page_phrases:
                # Map the command
                command = phrase_utils.toCommand(phrase)
                if command is not None:  
                    ordered_commands.append(command)

        return {
            SolveEnum.CLASSIFIED_TEXT: classified_text,
            SolveEnum.MATCHED_PHRASES: matched_phrases,
            SolveEnum.ORDERED_COMMANDS: ordered_commands,
        }


def main(args: argparse.Namespace) -> None:
    # Initialize PageReader
    pr = PageReader(note="")

    # Load maze run pages
    mr_path = "page_data/python_train/001.npz"
    maze_run = utils.MazeRunLoader.fromFile(mr_path)
    pages = maze_run.pages

    pr.fit(training=True)

    # Process the pages
    results = pr.solve(pages)

    # Get results
    # classified_text = results[SolveEnum.CLASSIFIED_TEXT]
    # matched_phrases = results[SolveEnum.MATCHED_PHRASES]
    # ordered_commands = results[SolveEnum.ORDERED_COMMANDS]

    # Print the classified text and matched phrases from each page
    # print("Classified Text and Matched Phrases:")
    # for page_num, (page_text, page_phrases) in enumerate(zip(classified_text, matched_phrases), start=1):
    #     print(f"Page {page_num}:")
    #     for line, phrase in zip(page_text, page_phrases):
    #         print(f"Text: {line}")
    #         print(f"Matched Phrase: {phrase}")
    #         print()
    #     print()

    # # Print the ordered commands
    # print("Ordered Commands:")
    # for command in ordered_commands:
    #     print(command)
    # print()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
