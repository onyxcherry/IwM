from math import pi, sqrt, floor, ceil
import os

from PIL import Image
from PIL.ImageFile import ImageFile
from skimage.draw import line_nd
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from time import perf_counter


def load_bitmap(filename: str) -> ImageFile:
    image = Image.open("./obrazy/" + filename)
    return image


def find_coords_inside_image(
    pixels: NDArray,
    im_size: tuple,
) -> tuple:
    # 3. b) wyznacz linie przejścia tylko przez obraz (wersja z wyszukiwaniem przez połowienie)
    width = im_size[0]
    height = im_size[1]

    line_start = 0
    line_end = pixels.shape[1] - 1
    mid_line_point = (line_start + line_end) // 2

    pixel_x, pixel_y = pixels[:, mid_line_point]
    while not (0 <= pixel_x < width and 0 <= pixel_y < height):
        return (-1, -1)

    end = mid_line_point
    # print(f"1. {line_start} : {end}")
    while line_start != end:
        mid_point = (line_start + end) // 2
        pixel_x, pixel_y = pixels[:, mid_point]
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            end = mid_point
        else:
            line_start = mid_point + 1
        # print(f"1. {line_start} : {end}")

    start = mid_line_point
    # print(f"2. {start} : {line_end}")
    while start != line_end:
        mid_point = ((start + line_end) // 2) + 1
        pixel_x, pixel_y = pixels[:, mid_point]
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            start = mid_point
        else:
            line_end = mid_point - 1

    return (line_start, line_end)


def calculate_scan_rays(
    image_filename: str,
    scans_count: int,
    alfa_step: float,
    detectors_count: int,
    detectors_angular_aperture: float,
    ifCreateOptionalPartialImages: bool = False,
    ifCreateOptionalFullImage: bool = False,
) -> NDArray:
    image = load_bitmap(image_filename)
    image.load()
    im_size = image.size
    image.close()

    # 1. Opisz okrąg na obrazie
    diagonal_length = sqrt(pow(im_size[0], 2) + pow(im_size[1], 2))
    radius = diagonal_length / 2

    im_left_x = -im_size[0] // 2
    im_top_y = -im_size[1] // 2

    # 2. Wyznacz położenie n emiterów i n detektorów na okręgu

    # Arrays of all emitters and detectors positions set to (radius, radius)
    emitters_positions = np.full(
        (scans_count, detectors_count, 2), radius, dtype=np.float64
    )
    detectors_positions = np.full(
        (scans_count, detectors_count, 2), radius, dtype=np.float64
    )

    # Arrays of all emitters angles set to 0 rad
    emitters_angles = np.zeros((scans_count, detectors_count), dtype=np.float64)

    # Arrays of all detectors angles set to pi rad
    detectors_angles = np.full((scans_count, detectors_count), pi, dtype=np.float64)

    # Array of angle differences between scans
    scan_start_angles = np.arange(
        0, scans_count * alfa_step, alfa_step, dtype=np.float64
    )

    # Update emitters and detectors angles to include difference in angle between scans
    emitters_angles += scan_start_angles[:, np.newaxis]
    detectors_angles += scan_start_angles[:, np.newaxis]

    # Array of angle differences between detectors/emitters
    angles_dif = np.linspace(
        -detectors_angular_aperture / 2,
        detectors_angular_aperture / 2,
        detectors_count,
        dtype=np.float64,
    )

    # Update emitters and detectors angles to include difference in angle between detectors/emitters
    emitters_angles -= angles_dif
    detectors_angles += angles_dif

    # Calculate cordinates of emitters and detectors
    emitters_positions_x = np.cos(emitters_angles)
    emitters_positions_y = np.sin(emitters_angles)
    emitters_positions_xy = np.stack(
        (emitters_positions_x, emitters_positions_y), axis=2, dtype=np.float64
    )
    emitters_positions *= emitters_positions_xy

    detectors_positions_x = np.cos(detectors_angles)
    detectors_positions_y = np.sin(detectors_angles)
    detectors_positions_xy = np.stack(
        (detectors_positions_x, detectors_positions_y), axis=2, dtype=np.float64
    )
    detectors_positions_xy = np.stack(
        (np.cos(detectors_angles), np.sin(detectors_angles)), axis=2, dtype=np.float64
    )
    detectors_positions *= detectors_positions_xy

    all_scans_rays = np.empty((scans_count, detectors_count), dtype=np.object_)
    for scan_number in range(scans_count):
        # print(f"Create lines in {scan_number} scan")
        for detector_number in range(detectors_count):
            # 3. Wyznacz linie przejścia
            # 3. a) wyznacz linie przejścia od emiterów do detektorów
            emitter_coords = emitters_positions[scan_number][detector_number]
            detector_coords = detectors_positions[scan_number][detector_number]

            pixels_coords = line_nd(emitter_coords, detector_coords, endpoint=True)

            # Change coordinate system to image coordinate system where point (0,0) means upper left corner of image
            pixels_coords -= np.array([im_left_x, im_top_y])[:, np.newaxis]

            # 3. b) wyznacz linie przejścia tylko przez obraz (wersja z wyszukiwaniem przez połowienie)
            line_start, line_end = find_coords_inside_image(pixels_coords, im_size)
            if line_start == -1:
                # Alternative way to calculate similiar time requirements
                width = im_size[0]
                height = im_size[1]
                filter = (
                    (0 <= pixels_coords[0, :])
                    & (pixels_coords[0, :] < width)
                    & (0 <= pixels_coords[1, :])
                    & (pixels_coords[1, :] < height)
                )
                all_scans_rays[scan_number][detector_number] = tuple(
                    pixels_coords[:, filter]
                )
            else:
                all_scans_rays[scan_number][detector_number] = tuple(
                    pixels_coords[:, line_start : line_end + 1]
                )

        # Optional partial rays image
        if ifCreateOptionalPartialImages:
            plt.figure(figsize=(8, 8))
            plt.axis("equal")
            plt.xlim((-floor(radius * 1.1), ceil(radius * 1.1)))
            plt.ylim((ceil(radius * 1.1), -floor(radius * 1.1)))
            x, y = zip(*(emitters_positions[scan_number]))
            plt.plot(x, y, "ro", markersize=2)
            x, y = zip(*(detectors_positions[scan_number]))
            plt.plot(x, y, "go", markersize=2)
            all_ray_pixels_x = []
            all_ray_pixels_y = []
            for ray in all_scans_rays[scan_number]:
                x, y = ray
                all_ray_pixels_x.extend(np.array(x) + im_left_x)
                all_ray_pixels_y.extend(np.array(y) + im_top_y)
            x, y = all_ray_pixels_x, all_ray_pixels_y
            plt.plot(x, y, "bo", markersize=1, alpha=0.1)
            plt.imshow(
                plt.imread(f"obrazy/{image_filename}"),
                extent=(
                    im_left_x,
                    im_left_x + im_size[0],
                    im_top_y + im_size[1],
                    im_top_y,
                ),
            )
            if not os.path.exists(f"{image_filename.split(".")[0]}/rays"):
                if not os.path.exists(f"{image_filename.split(".")[0]}"):
                    os.mkdir(f"{image_filename.split(".")[0]}")
                os.mkdir(f"{image_filename.split(".")[0]}/rays")
            plt.savefig(f"{image_filename.split(".")[0]}/rays/pixels_{scan_number}.png")

    # Optional full rays image
    if ifCreateOptionalFullImage:
        diagonal_length = sqrt(pow(im_size[0], 2) + pow(im_size[1], 2))
        radius = diagonal_length / 2
        plt.figure(figsize=(8, 8))
        plt.axis("equal")
        plt.xlim((-floor(radius * 1.1), ceil(radius * 1.1)))
        plt.ylim((ceil(radius * 1.1), -floor(radius * 1.1)))
        all_emitters_pixels_x = []
        all_emitters_pixels_y = []
        for scan_emitters_positions in emitters_positions:
            for emitter_position in scan_emitters_positions:
                x, y = emitter_position
                all_emitters_pixels_x.append(x)
                all_emitters_pixels_y.append(y)
        plt.plot(all_emitters_pixels_x, all_emitters_pixels_y, "go", markersize=2)
        plt.plot(x, y, "ro", markersize=2)
        all_detectors_pixels_x = []
        all_detectors_pixels_y = []
        for scan_detectors_positions in detectors_positions:
            for detector_position in scan_detectors_positions:
                x, y = detector_position
                all_detectors_pixels_x.append(x)
                all_detectors_pixels_y.append(y)
        plt.plot(all_detectors_pixels_x, all_detectors_pixels_y, "go", markersize=2)
        all_ray_pixels_x = []
        all_ray_pixels_y = []
        for scan_rays in all_scans_rays:
            for ray in scan_rays:
                x, y = ray
                all_ray_pixels_x.extend(np.array(x) + im_left_x)
                all_ray_pixels_y.extend(np.array(y) + im_top_y)
        plt.plot(all_ray_pixels_x, all_ray_pixels_y, "bo", markersize=1, alpha=0.1)
        plt.imshow(
            plt.imread(f"obrazy/{image_filename}"),
            extent=(
                im_left_x,
                im_left_x + im_size[0],
                im_top_y + im_size[1],
                im_top_y,
            ),
        )
        if not os.path.exists(f"{image_filename.split(".")[0]}/rays"):
            if not os.path.exists(f"{image_filename.split(".")[0]}"):
                os.mkdir(f"{image_filename.split(".")[0]}")
            os.mkdir(f"{image_filename.split(".")[0]}/rays")
        plt.savefig(f"{image_filename.split(".")[0]}/rays/pixels.png")

    return all_scans_rays


def calculate_sinogram(
    image_filename: str,
    all_scans_rays: NDArray,
) -> NDArray:
    # Create array for sinogram data
    scans_count, detectors_count = all_scans_rays.shape
    data = np.zeros((scans_count, detectors_count), dtype=np.float64)

    image = load_bitmap(image_filename)
    image.load()
    im_size = image.size
    im_pixels = np.asarray(image, dtype=np.float64)
    image.close()

    im_pixels1 = im_pixels.mean(axis=2)

    for scan_number in range(scans_count):
        for detector_number in range(detectors_count):
            ray_matrix = np.zeros(im_size, dtype=np.uint8)
            ray_matrix[all_scans_rays[scan_number, detector_number]] = 1

            ray_pixel_number = ray_matrix.sum()

            ray_sum = np.sum(ray_matrix * im_pixels1)

            data[scan_number, detector_number] = ray_sum / ray_pixel_number

    if not os.path.exists(f"{image_filename.split(".")[0]}"):
        os.mkdir(f"{image_filename.split(".")[0]}")
    save_matrix_to_grayscale_image(data, f"{image_filename.split(".")[0]}/sinogram.png")

    # 5. Zwróć sinogram (bez filtracji)
    return data


def save_matrix_to_grayscale_image(matrix: NDArray, output_filename: str):

    height, width = matrix.shape

    image = Image.new("L", (width, height))  # "L" mode for grayscale
    image.putdata(matrix.flatten())
    image.save(output_filename)


def make_image(
    filename: str,
    all_scans_rays: NDArray,
    sinogram: NDArray,
):
    scans_count, detectors_count = sinogram.shape

    image = load_bitmap(filename)
    im_size = image.size
    image.close()

    image_data = np.zeros(im_size)
    for scan_number in range(scans_count):
        for detector_number in range(detectors_count):
            ray_matrix = np.zeros(im_size, dtype=np.uint8)
            ray_matrix[all_scans_rays[scan_number, detector_number]] = 1

            image_data += ray_matrix * sinogram[scan_number, detector_number]

    image_data = (
        (image_data - np.min(image_data))
        / (np.max(image_data) - np.min(image_data))
        * (255)
    )

    if not os.path.exists(f"{filename.split(".")[0]}"):
        os.mkdir(f"{filename.split(".")[0]}")
    save_matrix_to_grayscale_image(image_data, f"{filename.split(".")[0]}/obraz.png")


def main():
    filename = "Kropka.jpg"
    scans_count = 180
    alfa_step = 2 / scans_count * pi
    detectors_count = 180
    detectors_angular_aperture = 0.25 * pi

    start = perf_counter()
    scan_rays = calculate_scan_rays(
        filename, scans_count, alfa_step, detectors_count, detectors_angular_aperture
    )
    exit()
    sinogram = calculate_sinogram(filename, scan_rays)
    make_image(filename, scan_rays, sinogram)
    stop = perf_counter()
    print(f"Script finished in {stop-start} seconds")


# Aplikacja powinna móc pozwolić konfigurować następujące elementy:
# Krok ∆α układu emiter/detektor.
# Dla jednego układu emiter/detektor liczbę detektorów (n).
# Rozwartość/rozpiętość układu emiter/detektor (l).


if __name__ == "__main__":
    main()
