#!/usr/bin/env python
import itertools
import multiprocessing
import os
import pickle
import re
import sys
from pathlib import Path

import click
import face_recognition.api as face_recognition
import numpy as np
import PIL.Image


DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

# def scan_known_people(known_people_folder=None,model: str = "cnn"):


def scan_known_people(known_people_folder=None, model: str = "hog"):
    known_names = []
    known_face_encodings = []

    if known_people_folder:
        for file in image_files_in_folder(known_people_folder):
            basename = os.path.splitext(os.path.basename(file))[0]

            img = face_recognition.load_image_file(file)

            face_locations = face_recognition.face_locations(img, model=model)
            face_encodings = face_recognition.face_encodings(img, face_locations)

            if len(face_encodings) > 1:
                click.echo(
                    "WARNING: More than one face found in {}. Only considering the first face.".format(
                        file
                    )
                )

            if len(face_encodings) == 0:
                click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
            else:
                known_names.append(basename)
                known_face_encodings.append(face_encodings[0])

        print(f"Generados encodings de {len(known_names)} nuevas caras!")

        name_encodings = {"names": known_names, "encodings": known_face_encodings}
        with DEFAULT_ENCODINGS_PATH.open(mode="wb") as f:
            pickle.dump(name_encodings, f)
    else:
        with DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
            name_encodings = pickle.load(f)
            known_names = name_encodings["names"]
            known_face_encodings = name_encodings["encodings"]
        print(f"Cargadas {len(known_names)} nuevas caras!")

    return known_names, known_face_encodings


def print_result(filename, name, distance, show_distance=True):
    if show_distance:
        print("{},{},{}".format(filename, name, distance))
    else:
        print("{},{}".format(filename, name))


def test_image(
    image_to_check,
    known_names,
    known_face_encodings,
    tolerance=0.6,
    show_distance=False,
):
    unknown_image = face_recognition.load_image_file(image_to_check)

    # Scale down image if it's giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)

    unknown_encodings = face_recognition.face_encodings(unknown_image)

    for unknown_encoding in unknown_encodings:
        distances = face_recognition.face_distance(
            known_face_encodings, unknown_encoding
        )
        # cambio 20240826
        result = list(distances > 0.0)
        # result = list(distances <= tolerance)

        if True in result:
            [
                print_result(image_to_check, name, distance, show_distance)
                for is_match, name, distance in zip(result, known_names, distances)
                if is_match
            ]
        else:
            print_result(image_to_check, "unknown_person", None, show_distance)

    if not unknown_encodings:
        # print out fact that no faces were found in image
        print_result(image_to_check, "no_persons_found", None, show_distance)


def image_files_in_folder(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if re.match(r".*\.(jpg|jpeg|png)", f, flags=re.I)
    ]


def process_images_in_process_pool(
    images_to_check,
    known_names,
    known_face_encodings,
    number_of_cpus,
    tolerance,
    show_distance,
):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(known_names),
        itertools.repeat(known_face_encodings),
        itertools.repeat(tolerance),
        itertools.repeat(show_distance),
    )

    pool.starmap(test_image, function_parameters)


@click.command()
@click.argument("image_to_check")
@click.option("--known_people_folder", default=None, help="")
@click.option(
    "--cpus",
    default=1,
    help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"',
)
@click.option(
    "--tolerance",
    default=0.6,
    help="Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.",
)
@click.option(
    "--show-distance",
    default=True,
    type=bool,
    help="Output face distance. Useful for tweaking tolerance setting.",
)
def main(image_to_check, known_people_folder, cpus, tolerance, show_distance):
    if not known_people_folder:
        known_names, known_face_encodings = scan_known_people()
    else:
        known_names, known_face_encodings = scan_known_people(known_people_folder)

    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo(
            "WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!"
        )
        cpus = 1

    if os.path.isdir(image_to_check):
        if cpus == 1:
            [
                test_image(
                    image_file,
                    known_names,
                    known_face_encodings,
                    tolerance,
                    show_distance,
                )
                for image_file in image_files_in_folder(image_to_check)
            ]
        else:
            process_images_in_process_pool(
                image_files_in_folder(image_to_check),
                known_names,
                known_face_encodings,
                cpus,
                tolerance,
                show_distance,
            )
    else:
        test_image(
            image_to_check, known_names, known_face_encodings, tolerance, show_distance
        )


if __name__ == "__main__":
    main()
