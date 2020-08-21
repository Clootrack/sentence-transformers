import csv
import json
import os
import pickle
import re
import shutil
from collections import defaultdict
from os.path import join, isfile, isdir, basename, normpath
from pathlib import Path
from shutil import rmtree, copyfile
from typing import List, Sequence, Any, Callable, Tuple

csv.field_size_limit(100000000)


def ensure_directory(file_path):
    directory = os.path.dirname(file_path)
    if not directory:
        # file_path is a file, no need to create directory.
        return
    if not os.path.exists(directory):
        os.makedirs(directory)


def chunks(list_name, chunk_size):
    """
    :param list_name: The name of the list which needs to be sliced into chunks
    :param chunk_size: The size of each chunk
    This method is used to break a list into chunks of "n" size
    """
    for count in range(0, len(list_name), chunk_size):
        yield list_name[count:count + chunk_size]


# Read from csv file and return row_list
def read_csv_file(csv_file):
    row_list = []
    with open(csv_file, 'r', newline='', encoding='utf-8') as bf:
        csvf = csv.DictReader(bf, delimiter=',', quotechar='"')
        for row in csvf:
            row_list.append(row)
    return row_list


def write_csv_to_file_dictwriter(file_name, header, rows, extrasaction="raise"):
    ensure_directory(file_name)
    with open(file_name, 'w', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=header, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                extrasaction=extrasaction)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def csv_row_size(file_name: str) -> int:
    rows = read_csv_file(file_name)
    return len(rows)


class AnyKeywordMention:
    def __init__(self, keywords):
        self.keywords = set(self._remove_blank_keywords(keywords))
        self.regex = self._make_keyword_regex()

    def _remove_blank_keywords(self, keywords):
        return list(filter(lambda x: x, keywords))

    def _make_keyword_regex(self):
        regex = re.compile(r'|'.join(self.keywords), re.IGNORECASE)
        return regex

    def has_mention(self, text):
        res = self.regex.search(text)
        return True if res else False


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def read_object(filename):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj


class IOPost:
    def __init__(self, input_file, output_file):
        self.post_file = input_file
        self.output_file = output_file
        self.fieldnames = None

    def read_posts(self):
        posts = []
        with open(self.post_file, 'r', encoding='utf-8') as in_file:
            csv_reader = csv.DictReader(in_file, delimiter=',', quotechar='"')
            self.fieldnames = csv_reader.fieldnames
            for row in csv_reader:
                posts.append(row)
        return read_csv_file(self.post_file)

    def write_posts(self, posts):
        with open(self.output_file, 'w', encoding='utf-8') as out_file:
            csv_writer = csv.DictWriter(out_file, fieldnames=self.fieldnames, delimiter=',', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
            csv_writer.writeheader()
            for post in posts:
                csv_writer.writerow(post)


def load_json_data_from_file(input_file, default_dict=False, defaultdict_type=dict):
    with open(input_file) as file:
        data_dict = json.load(file)
        if default_dict:
            return defaultdict(defaultdict_type, data_dict)
        return data_dict


def write_json_data_to_file(file, data_dict):
    ensure_directory(file)
    with open(file, 'w') as outfile:
        json.dump(data_dict, outfile)


def format_float(number, decimal_places=2):
    float_format = "{:05." + str(decimal_places) + "f}"
    return float_format.format(number)


def read_text_file(input_file):
    """
    This method will return you a list of lines in a text file after applying a strip()
    on each line also making sure the line is not None after the strip() operation
    :param input_file: the input file to be read
    :return: list of lines contained in the input file
    """
    lines_list = list()
    with open(input_file, 'r') as inp_file:
        for line in inp_file:
            line = line.strip()
            if line:
                lines_list.append(line)
    return lines_list


def path_exists(file_path):
    my_file = Path(file_path)
    return my_file.exists()


class ReviewFileHandler:
    def load_reviews(self, reviewfile: str):
        with open(reviewfile, encoding='utf-8') as datafile:
            reader = csv.DictReader(datafile)
            rows = [r for r in reader]
            for row in rows:
                row['Review'] = row['Review'].lower()
        return rows


def split_on_condition(seq: Sequence[Any], condition: Callable[..., bool]) -> Tuple[List[Any], List[Any]]:
    a, b = [], []
    for item in seq:
        (a if condition(item) else b).append(item)
    return a, b


class OSFileOperations:
    @staticmethod
    def get_all_files(path, recursive=True):
        files = []
        for file_name in os.listdir(path):
            file_path = join(path, file_name)
            if isfile(file_path):
                files.append(file_path)
            elif isdir(file_path):
                if recursive:
                    files += OSFileOperations.get_all_files(file_path)
        return files

    @staticmethod
    def get_all_csv_files(path, recursive=True):
        files = OSFileOperations.get_all_files(path, recursive=recursive)
        files = [file for file in files if file.endswith("csv")]
        return files

    @staticmethod
    def get_all_dirs(path, recursive=True):
        dirs = []
        for dir_path in os.listdir(path):
            dir_path = join(path, dir_path)
            if isfile(dir_path):
                continue
            elif isdir(dir_path):
                dirs.append(dir_path)
                if recursive:
                    dirs += OSFileOperations.get_all_dirs(dir_path)
        return dirs

    @staticmethod
    def get_base_name(path):
        base_name = basename(normpath(path))
        return base_name

    @staticmethod
    def get_base_name_without_extension(path):
        base_name_with_extension = OSFileOperations.get_base_name(path)
        base_name = os.path.splitext(base_name_with_extension)[0]
        return base_name

    @staticmethod
    def ensure_directory(file_path):
        directory = OSFileOperations.get_dir_path(file_path)
        if not directory:
            # file_path is a file, no need to create directory.
            return
        if not OSFileOperations.entity_exists(directory):
            os.makedirs(directory)

    @staticmethod
    def clear_path(entity_path):
        if not OSFileOperations.entity_exists(entity_path):
            return
        if isfile(entity_path):
            os.remove(entity_path)
        else:
            rmtree(entity_path)

    @staticmethod
    def entity_exists(file_path):
        return os.path.exists(file_path)

    @staticmethod
    def get_all_sub_directories(directory_path: str) -> List[str]:
        sub_directories = list()
        files_and_folder_names = os.listdir(directory_path)
        for item in files_and_folder_names:
            path = '{}{}'.format(directory_path, item)
            if os.path.isdir(path):
                sub_directories.append(item)
        return sub_directories

    @staticmethod
    def copy_file(source_file_path, destination):
        OSFileOperations.ensure_directory(destination)
        copyfile(source_file_path, destination)

    @staticmethod
    def get_dir_path(file_path):
        directory = os.path.dirname(file_path)
        return directory

    @staticmethod
    def copy_directory(source, destination, overwrite=False):
        if overwrite:
            OSFileOperations.clear_path(destination)
        shutil.copytree(source, destination)

    @staticmethod
    def move_directory(source, destination, overwrite=False):
        if overwrite:
            OSFileOperations.clear_path(destination)
        OSFileOperations.copy_directory(source, destination)
        OSFileOperations.clear_path(source)

    @staticmethod
    def rename_entity(source, destination):
        if not OSFileOperations.entity_exists(source):
            return
        os.rename(source, destination)

    @staticmethod
    def remove_directory(path):
        shutil.rmtree(path)

    @staticmethod
    def get_all_log_files(path: str, recursive: bool = False) -> List[str]:
        try:
            files = OSFileOperations.get_all_files(path, recursive=recursive)
            files = [file for file in files if file.endswith("log")]
        except FileNotFoundError:
            files = []
        return files
