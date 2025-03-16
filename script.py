from math import cos, pi, sin, sqrt
from statistics import mean

from PIL import Image
from PIL.ImageFile import ImageFile
from skimage.draw import line_nd
import numpy as np

import matplotlib.pyplot as plt
from math import floor, ceil

def load_bitmap(filename: str) -> ImageFile:
    image = Image.open(filename)
    return image


def scan(
    image_filename,
    scans_count: int,
    alfa_step: float,
    detectors_count: int,
    detectors_angular_aperture: float,
) -> dict[float, list[float]]:
    # angle --> list of data
    data: dict[float, list[float]] = {}

    image = load_bitmap(image_filename)

    # 1. Opisz okrąg na obrazie
    im_size = image.size
    diagonal_length = sqrt(pow(im_size[0], 2) + pow(im_size[1], 2))
    radius = diagonal_length / 2
    
    im_left_x = -im_size[0] // 2
    # im_right_x = im_size[0] / 2
    im_right_x = im_left_x + im_size[0] - 1
    im_top_y = -im_size[1] // 2
    # im_bottom_y = im_size[1] / 2
    im_bottom_y = im_top_y + im_size[1] - 1

    # 2. Wyznacz położenie n emiterów i n detektorów na okręgu

    all_pixels_lines_within_image: list[tuple[int, int]] = []
    all_emitters: list[tuple[int, int]] = []
    all_detectors: list[tuple[int, int]] = []
    alfa = alfa_step
    for scan_number in range(scans_count):
        print(f"Scan {scan_number}")
        data[alfa] = []
        emitters: list[tuple[int, int]] = []
        detectors: list[tuple[int, int]] = []

        pixels_on_scan_lines_within_image: list[tuple[int, int]] = []
        
        mid_angle = alfa + pi

        arg_diff = detectors_angular_aperture / (detectors_count - 1)

        d_0_arg = mid_angle - detectors_angular_aperture / 2
        d_i_arg = d_0_arg

        e_0_arg = alfa + detectors_angular_aperture / 2
        e_i_arg = e_0_arg

        for detector_number in range(detectors_count):
            if(detector_number > 0):
                d_i_arg += arg_diff
                e_i_arg -= arg_diff
            
            x_d_i = radius * cos(d_i_arg)
            y_d_i = radius * sin(d_i_arg)
            detector_coords = (x_d_i, y_d_i)

            x_e_i = radius * cos(e_i_arg)
            y_e_i = radius * sin(e_i_arg)
            emitter_coords = (x_e_i, y_e_i)

            # 3. Wyznacz linie przejścia
            # 3. a) wyznacz linie przejścia od emiterów do detektorów
            pixels_coords = line_nd(emitter_coords, detector_coords, endpoint=True)
            pixels = list(map(list,zip(pixels_coords[0], pixels_coords[1])))
            pixels_count = len(pixels)
            
            # 3. b) wyznacz linie przejścia tylko przez obraz (wersja z wyszukiwaniem przez połowienie)
            line_start = 0
            line_end = pixels_count-1
            mid_line_point = (line_end - line_start) // 2

            pixel_x, pixel_y = pixels[mid_line_point]
            if(not(
                im_left_x <= pixel_x <= im_right_x
                and im_top_y <= pixel_y <= im_bottom_y
            )):
                # raczej nie możliwe
                print("coś się zepsuło")
                exit()
            
            end = mid_line_point
            # print(f"1. {line_start} : {end}")
            while(line_start != end):
                mid_point = (end + line_start) // 2
                pixel_x, pixel_y = pixels[mid_point]
                if(
                    im_left_x <= pixel_x <= im_right_x
                    and im_top_y <= pixel_y <= im_bottom_y
                ):
                    end = mid_point
                else:
                    line_start = mid_point + 1
                # print(f"1. {line_start} : {end}")

            start = mid_line_point
            # print(f"2. {start} : {line_end}")
            while(start != line_end):
                mid_point = ((line_end + start) // 2) +1
                pixel_x, pixel_y = pixels[mid_point]
                if(
                    im_left_x <= pixel_x <= im_right_x
                    and im_top_y <= pixel_y <= im_bottom_y
                ):
                    start = mid_point
                else:
                    line_end = mid_point - 1

            pixels_on_line_within_image = pixels[line_start:line_end+1]

            
            # for pixel_x, pixel_y in zip(pixels_coords[0], pixels_coords[1]):
            #     # 3. b) wyznacz linie przejścia tylko przez obraz
            #     if (
            #         im_left_x <= pixel_x <= im_right_x
            #         and im_top_y <= pixel_y <= im_bottom_y
            #     ):
            #         pixels_on_scan_lines_within_image.append((pixel_x, pixel_y))


            # 4. Wyznacz (średnią) wartość przejścia sygnału
            im_pixels = image.load()
            pixels_greyscale_values: list[float] = []
            for pixel_x, pixel_y in pixels_on_line_within_image:
                rgb = im_pixels[pixel_x-im_left_x, pixel_y-im_top_y]
                # val = mean(rgb) * 5
                val = mean(rgb)
                pixels_greyscale_values.append(val)
            whole_line_mean = mean(pixels_greyscale_values)
            data[alfa].append(whole_line_mean)

            # Opcjonalne dane do wykresów
            # detectors.append(detector_coords)
            # all_detectors.append(detector_coords)
            # emitters.append(emitter_coords)
            # all_emitters.append(emitter_coords)
            # pixels_on_scan_lines_within_image.extend(pixels[line_start:line_end+1])
            # all_pixels_lines_within_image.extend(pixels[line_start:line_end+1])

        alfa += alfa_step

        # Opcjonalnhy wykres
        # plt.figure(figsize=(8, 8))
        # plt.axis("equal")
        # plt.xlim((-floor(radius * 1.1), ceil(radius * 1.1)))
        # plt.ylim((ceil(radius * 1.1), -floor(radius * 1.1)))
        # x, y = zip(*(emitters))
        # plt.plot(x, y, "ro", markersize=2)
        # x, y = zip(*(detectors))
        # plt.plot(x, y, "go", markersize=2)
        # x, y = zip(*(pixels_on_scan_lines_within_image))
        # plt.plot(x, y, "bo", markersize=1, alpha=0.1)
        # plt.imshow(plt.imread(image_filename), extent=(im_left_x, im_right_x, im_bottom_y, im_top_y))
        # plt.savefig(f"pixels_{scan_number}.png")


    # 5. Zwróć sinogram (bez filtracji)

    # Pokaż linie skanowania --> w pętli for (zewnętrznej) dodać break na końcu,
    # by pokazać linie tylko dla jednego skanu
    # Opcjonalnhy wykres
    # plt.figure(figsize=(8, 8))
    # plt.axis("equal")
    # plt.xlim((-floor(radius * 1.1), ceil(radius * 1.1)))
    # plt.ylim((ceil(radius * 1.1), -floor(radius * 1.1)))
    # x, y = zip(*(all_emitters))
    # plt.plot(x, y, "ro", markersize=2)
    # x, y = zip(*(all_detectors))
    # plt.plot(x, y, "go", markersize=2)
    # x, y = zip(*(all_pixels_lines_within_image))
    # plt.plot(x, y, "bo", markersize=1, alpha=0.1)
    # plt.imshow(plt.imread(image_filename), extent=(im_left_x, im_right_x, im_bottom_y, im_top_y))
    # plt.savefig(f"pixels.png")

    return data


def write_matrix_to_grayscale_file(matrix, output_filename: str) -> None:
    height = len(matrix)
    width = len(matrix[0])
    flat_pixels = sum(matrix, [])

    image = Image.new("L", (width, height))  # "L" mode for grayscale
    image.putdata(flat_pixels)

    image.save(output_filename)

def make_image(
    image_filename: str,
    scans_count: int, 
    detectors_count: int,
    detectors_angular_aperture: float,
    sinogram: dict[float, list[float]],
    output_filename: str
):
    image = load_bitmap(image_filename)

    # 1. Opisz okrąg na obrazie
    im_size = image.size
    diagonal_length = sqrt(pow(im_size[0], 2) + pow(im_size[1], 2))
    radius = diagonal_length / 2
    
    im_left_x = -im_size[0] // 2
    im_right_x = im_left_x + im_size[0] - 1
    im_top_y = -im_size[1] // 2
    im_bottom_y = im_top_y + im_size[1] - 1

    matrix = np.zeros((im_size[0],im_size[1]),float)

    # 2. Wyznacz położenie n emiterów i n detektorów na okręgu

    for alfa, values in sinogram.items():
        print(f"alpha: {(alfa*360.0)/(2.0*pi)}")
        
        mid_angle = alfa + pi

        arg_diff = detectors_angular_aperture / (detectors_count - 1)

        d_0_arg = mid_angle - detectors_angular_aperture / 2
        d_i_arg = d_0_arg

        e_0_arg = alfa + detectors_angular_aperture / 2
        e_i_arg = e_0_arg

        for detector_number, value in enumerate(values):
            if(detector_number > 0):
                d_i_arg += arg_diff
                e_i_arg -= arg_diff
            
            x_d_i = radius * cos(d_i_arg)
            y_d_i = radius * sin(d_i_arg)
            detector_coords = (x_d_i, y_d_i)

            x_e_i = radius * cos(e_i_arg)
            y_e_i = radius * sin(e_i_arg)
            emitter_coords = (x_e_i, y_e_i)

            # 3. Wyznacz linie przejścia
            # 3. a) wyznacz linie przejścia od emiterów do detektorów
            pixels_coords = line_nd(emitter_coords, detector_coords, endpoint=True)
            pixels = list(map(list,zip(pixels_coords[0], pixels_coords[1])))
            pixels_count = len(pixels)
            
            # 3. b) wyznacz linie przejścia tylko przez obraz (wersja z wyszukiwaniem przez połowienie)
            line_start = 0
            line_end = pixels_count-1
            mid_line_point = (line_end - line_start) // 2

            pixel_x, pixel_y = pixels[mid_line_point]
            if(not(
                im_left_x <= pixel_x <= im_right_x
                and im_top_y <= pixel_y <= im_bottom_y
            )):
                # raczej nie możliwe
                print("coś się zepsuło")
                exit()
            
            end = mid_line_point
            while(line_start != end):
                mid_point = (end + line_start) // 2
                pixel_x, pixel_y = pixels[mid_point]
                if(
                    im_left_x <= pixel_x <= im_right_x
                    and im_top_y <= pixel_y <= im_bottom_y
                ):
                    end = mid_point
                else:
                    line_start = mid_point + 1

            start = mid_line_point
            while(start != line_end):
                mid_point = ((line_end + start) // 2) +1
                pixel_x, pixel_y = pixels[mid_point]
                if(
                    im_left_x <= pixel_x <= im_right_x
                    and im_top_y <= pixel_y <= im_bottom_y
                ):
                    start = mid_point
                else:
                    line_end = mid_point - 1

            pixels_on_line_within_image = pixels[line_start:line_end+1]

            # 4. Wyznacz (średnią) wartość przejścia sygnału
            for pixel_x, pixel_y in pixels_on_line_within_image:
                matrix[pixel_x-im_left_x, pixel_y-im_top_y] += value
                # matrix[pixel_x-im_left_x, pixel_y-im_top_y] += (value/scans_count)

    image = Image.new("L", image.size)  # "L" mode for grayscale
    print(matrix.max())
    print(matrix.min())
    matrix = (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix))*(255)
    print(matrix.max())
    print(matrix.min())
    image.putdata(matrix.flatten())

    image.save(output_filename)


def main():
    inputFilename = "./obrazy/Kropka.jpg"
    # scans_count = 10
    scans_count = 80
    alfa_step = 2 / scans_count * pi
    # detectors_count = 36
    detectors_count = 80
    # detectors_angular_aperture = 0.15 * pi
    detectors_angular_aperture = 0.2 * pi
    sinogramFilename = "kropka_sinogram.png"
    outputFilename = "obraz.png"

    # sinogram = scan(image, 90, 1 / 90 * pi, 180, 0.2 * pi)
    sinogram = scan(inputFilename, scans_count, alfa_step, detectors_count, detectors_angular_aperture)
    sinogram_matrix = list(sinogram.values())
    write_matrix_to_grayscale_file(sinogram_matrix, sinogramFilename)
    make_image(inputFilename, scans_count, detectors_count, detectors_angular_aperture, sinogram, outputFilename)


# Aplikacja powinna móc pozwolić konfigurować następujące elementy:
# Krok ∆α układu emiter/detektor.
# Dla jednego układu emiter/detektor liczbę detektorów (n).
# Rozwartość/rozpiętość układu emiter/detektor (l).


if __name__ == "__main__":
    main()
